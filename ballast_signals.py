"""
Ballast Signal API v1 — structured, read-only snapshot over a UserTBG.

Mode 2/3: this module ONLY reads engine-computed fields. It never mutates the
graph and never calls an LLM. `get_insight()` (tbg_engine) is untouched; this is
a parallel, machine-readable output.

--------------------------------------------------------------------------------
Design decisions (what is IN, what is deliberately OUT, and why):

IN — `core` tier (mechanistically grounded, engine-computed):
  • beliefs        — label/category/confidence + evidence_mass (pos+neg),
                     anchored (mass ≥ ANCHOR_MASS), ambivalence
                     (tanh(min(pos,neg)/AMBIV_SCALE) — engine's own formula &
                     constant, imported, not duplicated).
  • conflicts      — ALL opposition edges, rank-ordered by edge confidence.
                     Confidence is an ORDINAL LLM signal, not a probability.
  • trajectories   — confidence_history as-is + a sign-of-diff direction.
                     NO velocity/acceleration (see OUT).
  • turning_points — with a `caveat` field: may include decay-induced duplicates
                     unless TBG_FIX_DECAY_TP=1 (engine E1 fix, default off).
  • oscillating    — Thompson (1995) oscillation signature re-derived read-only
                     from confidence_history (≥1 up AND ≥1 down over the last
                     THOMPSON_MIN_HIST+2 points) — the same test the engine uses
                     to clamp oscillating beliefs, exposed as a flag.
IN — `experimental` tier (separate key; each carries a validation marker):
  • graph          — (schema 1.2, moved here from core) deterministic structural
                     metrics: node/edge counts, connectivity (edges/node),
                     coherence (mean edge confidence — ordinal), relation
                     distribution, top hubs by degree. History-INDEPENDENT.
                     Marked "deterministic_unvalidated": the numbers are exact,
                     but their usefulness as a signal has not been measured.
  • domain_profile / stance_profile / type_profile — LLM-assigned taxonomies,
    marked "llm_assigned_unvalidated" (never validated against ground truth).
  • archive_summary — resolved (contradicted) vs faded (decayed) split, by a
    documented heuristic on evidence sign. Marked heuristic_unvalidated.

OUT — deliberately NOT included:
  • velocity / acceleration derivatives of confidence — noisy on a 5-point
    capped history; no evidence they carry signal. Direction-only is honest.
  • conviction gap — undefined/unvalidated construct.
  • any risk / crisis / clinical field — unvalidated and unsafe to surface.
  • AMF (Ambivalence Momentum Filter, amf_filter.compute_graph_amf) — GATED OUT.
    Pre-build check on the frozen demo_tbg_opref1.json graph (21 nodes): the
    headline signal amf_ambiv is ≈0 (max 0.032; 17/21 exactly 0.0) because
    confidence_history is capped at 5 and rarely accrues variance; amf_conf
    largely reflects history length/smoothness, not ambivalence. p8 could not be
    checked (the gate persisted only render strings, no BeliefNodes). Verdict:
    AMF carries no usable signal on this data → excluded from the API. Re-run the
    gate and revisit if a longer/uncapped history is available.
--------------------------------------------------------------------------------
"""
import math
import os
from collections import Counter

# Reuse engine constants — do NOT duplicate.
from tbg_engine import AMBIV_SCALE, THOMPSON_MIN_HIST

SCHEMA_VERSION = "1.2"  # 1.2: graph tier moved core -> experimental (deterministic
                        # but its usefulness is unvalidated; tier by our own discipline).

# Heuristic anchor threshold on accumulated evidence mass (pos+neg). A concept
# with this much evidence has been reinforced/contradicted enough to be "held",
# not a one-off mention. Not an engine constant — a display heuristic.
ANCHOR_MASS = 1.0

_OPPOSITION = ("contradicts", "conflicts_with", "blocks")


def _beliefs(tbg):
    out = []
    for n in sorted(tbg.nodes.values(), key=lambda n: n.confidence, reverse=True):
        mass = n.pos_evidence + n.neg_evidence
        out.append({
            "label": n.label,
            "category": n.category,
            "confidence": round(n.confidence, 3),
            "evidence_mass": round(mass, 3),
            "anchored": mass >= ANCHOR_MASS,
            "ambivalence": round(math.tanh(min(n.pos_evidence, n.neg_evidence) / AMBIV_SCALE), 4),
        })
    return out


def _conflicts(tbg):
    edges = [e for e in tbg.edges.values() if e.relation in _OPPOSITION]
    edges.sort(key=lambda e: e.confidence, reverse=True)
    out = []
    for e in edges:
        s = tbg.nodes.get(e.source_id); t = tbg.nodes.get(e.target_id)
        if s and t:
            out.append({
                "source": s.label,
                "target": t.label,
                "relation": e.relation,
                "confidence": round(e.confidence, 3),
                "confidence_kind": "ordinal_llm_signal",
            })
    return out


def _direction(points):
    # sign of net change over the history — no velocity/acceleration.
    if len(points) < 2:
        return "flat"
    delta = points[-1][1] - points[0][1]
    if delta > 0.02:
        return "up"
    if delta < -0.02:
        return "down"
    return "flat"


def _trajectories(tbg):
    out = []
    for n in tbg.nodes.values():
        pts = [[int(mc), round(c, 3)] for mc, c in n.confidence_history]
        if len(pts) >= 2:
            out.append({"label": n.label, "points": pts, "direction": _direction(pts)})
    return out


def _turning_points(tbg):
    fix_on = os.getenv("TBG_FIX_DECAY_TP", "0") == "1"
    return {
        "points": [
            {"message_count": tp.message_count,
             "cascade_magnitude": round(tp.cascade_magnitude, 3),
             "top_nodes": list(tp.top_nodes)}
            for tp in tbg.turning_points
        ],
        "caveat": "may include decay-induced duplicates unless TBG_FIX_DECAY_TP=1",
        "decay_fix_active": fix_on,
    }


def _oscillating(tbg):
    # Thompson (1995) oscillation signature: recent history shows both an up and
    # a down move. Same test the engine applies before clamping. Read-only.
    out = []
    for n in tbg.nodes.values():
        ch = [c for _, c in n.confidence_history]
        if len(ch) >= THOMPSON_MIN_HIST:
            ups = sum(1 for i in range(1, len(ch)) if ch[i] > ch[i - 1])
            downs = sum(1 for i in range(1, len(ch)) if ch[i] < ch[i - 1])
            if ups >= 1 and downs >= 1:
                out.append({"label": n.label, "ups": ups, "downs": downs,
                            "pos_evidence": round(n.pos_evidence, 3),
                            "neg_evidence": round(n.neg_evidence, 3)})
    return out


def _profile(nodes, attr):
    counts = Counter((getattr(n, attr) or "unknown") for n in nodes)
    return {"counts": dict(counts), "validation": "llm_assigned_unvalidated"}


def _archive_summary(tbg):
    arch = list(getattr(tbg, "archive_nodes", {}).values())
    # resolved (contradicted) vs faded (decayed), by evidence sign:
    #   contradiction leaves negative evidence dominant; passive decay does not.
    resolved = [n for n in arch if n.neg_evidence > 0 and n.neg_evidence >= n.pos_evidence]
    faded = [n for n in arch if n not in resolved]
    return {
        "total": len(arch),
        "resolved_contradicted": len(resolved),
        "faded_decayed": len(faded),
        "heuristic": "resolved = neg_evidence>0 and neg_evidence>=pos_evidence (contradiction "
                     "dominant); otherwise faded (passive decay)",
        "validation": "heuristic_unvalidated",
    }


def _graph(tbg):
    """Graph tier (1.1): deterministic structural metrics from nodes/edges only —
    history-independent, so NOT throttled by the confidence_history cap. Edge
    confidence is an ORDINAL LLM signal (same caveat as conflicts), not a probability."""
    nodes = tbg.nodes
    edges = list(tbg.edges.values())
    n, e = len(nodes), len(edges)
    deg = {}
    for x in edges:
        deg[x.source_id] = deg.get(x.source_id, 0) + 1
        deg[x.target_id] = deg.get(x.target_id, 0) + 1
    hubs = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "nodes": n,
        "edges": e,
        "connectivity": round(e / n, 4) if n else 0.0,          # edges/node; 0 on empty, no div-by-zero
        "coherence": (round(sum(x.confidence for x in edges) / e, 4) if e else None),  # mean edge conf (ordinal); None if no edges
        "relation_distribution": dict(Counter(x.relation for x in edges)),
        "top_hubs": [{"label": nodes[nid].label, "degree": d}
                     for nid, d in hubs if nid in nodes],
        "validation": "deterministic_unvalidated",  # metrics deterministic; utility unmeasured
    }


def snapshot(tbg) -> dict:
    """Structured, read-only snapshot of a UserTBG. 0 LLM. Does not mutate `tbg`."""
    nodes = list(tbg.nodes.values())
    return {
        "version": SCHEMA_VERSION,
        "user_id": getattr(tbg, "user_id", None),
        "message_count": getattr(tbg, "message_count", None),
        "core": {
            "beliefs": _beliefs(tbg),
            "conflicts": _conflicts(tbg),
            "trajectories": _trajectories(tbg),
            "turning_points": _turning_points(tbg),
            "oscillating": _oscillating(tbg),
        },
        "experimental": {
            "graph": _graph(tbg),   # 1.2: deterministic structural metrics; utility unvalidated
            "domain_profile": _profile(nodes, "domain"),
            "stance_profile": _profile(nodes, "stance"),
            "type_profile": _profile(nodes, "node_type"),
            "archive_summary": _archive_summary(tbg),
            # AMF intentionally absent — see module header gate verdict.
        },
    }
