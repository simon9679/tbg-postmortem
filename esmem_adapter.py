"""
TBG adapter over EvoEmo / ES-MemEval.

Ingests a conversation's sessions (one extraction per session, op/ref +
evidence_type ON, const memory MAX_NODES=50) and renders the resulting belief-
state for the answering LLM: current beliefs + confidence trajectories + conflicts
+ turning points. This is arm E (LLM + TBG) and the demo timeline core.

The engine is NOT modified — this is a pure adapter.
"""
import os
from typing import List, Dict, Optional

from tbg_schema import UserTBG
from tbg_engine import TBGEngine


# Equalized per-session INPUT budget — equalized pre-run, not tuned (set
# 2026-06-26 from dataset measurements, BEFORE any QA score existed). Every arm
# (TBG ingest via session_texts AND trunc/summary/rag/tracker via render_session)
# sees the SAME seeker + supporter content per session. SEEKER_BUDGET covers the
# dataset max seeker length (2292) so the user's disclosures are never truncated
# for ANY arm; SUPPORTER_BUDGET ~ p90 of supporter length. Removes the prior
# asymmetry where competitors were cut to 1400 chars/session (median 2590) while
# TBG saw the full seeker.
SEEKER_BUDGET = 2400
SUPPORTER_BUDGET = 1600


def _session_parts(session: dict) -> tuple:
    seeker, supporter = [], []
    for turn in session.get("dialogue", []):
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        (seeker if role == "seeker" else supporter).append(content)
    head = f"[{session.get('timestamp','')}] (emotion: {session.get('emotion','')}; " \
           f"topic: {session.get('topic','')})"
    return head, " ".join(seeker).strip(), " ".join(supporter).strip()


def session_texts(session: dict) -> tuple:
    """(seeker_text, supporter_text) for TBG extraction. Seeker = user (belief
    source), supporter = assistant. Same budgets as render_session -> equal input."""
    head, sk, sp = _session_parts(session)
    return (head + " " + sk[:SEEKER_BUDGET]).strip(), sp[:SUPPORTER_BUDGET].strip()


def render_session(session: dict) -> str:
    """Competitor-facing per-session text (trunc/summary/rag/tracker). Exposes the
    SAME seeker+supporter content (same budgets) that TBG ingests -> equal input."""
    head, sk, sp = _session_parts(session)
    return f"{head}\nseeker: {sk[:SEEKER_BUDGET]}\nsupporter: {sp[:SUPPORTER_BUDGET]}"


# Per-turn sign diagnostic. Default OFF -> zero behavior change (byte-identical).
# When ON, ingest writes tbg_signs_<cid>.txt as a FREE byproduct of a normal
# ingest: per session it records the reply and, for every node the turn touched,
# old_conf -> new_conf, Δ and direction (and the extraction decision: reinforce /
# contradict). Lets the sign behaviour of gpt-oss extraction be read without any
# extra LLM call. No re-ingest needed: enable the flag on the next scheduled ingest.
_DUMP_SIGNS = os.getenv("TBG_DUMP_SIGNS", "0") == "1"


def _log_turn(out, turn, user_text, delta, before, tbg):
    out.append(f"[turn {turn}] {user_text[:120]}")
    out.append(f"  delta: +{len(delta.add_nodes)}n "
               f"reinforce={len(delta.reinforce_ids)} contradict={len(delta.contradict_ids)} "
               f"strong_contradict={len(delta.strong_contradict_ids)}")
    contra = set(delta.contradict_ids) | set(delta.strong_contradict_ids)
    reinf = set(delta.reinforce_ids)
    touched = False
    for nid, n in tbg.nodes.items():
        old = before.get(nid)
        new = n.confidence
        if old is None:
            out.append(f'    + "{n.label}" NEW @ {new:.0%}')
            touched = True
        elif abs(new - old) >= 0.01:
            arrow = "UP " if new > old else "DOWN"
            tag = " [contradict]" if nid in contra else (" [reinforce]" if nid in reinf else "")
            out.append(f'    {arrow} "{n.label}" {old:.0%} -> {new:.0%}  d{new-old:+.0%}{tag}')
            touched = True
    if not touched:
        out.append("    (no confidence change)")
    out.append("")


async def ingest(conversation: dict, llm_fn, *, engine: Optional[TBGEngine] = None,
                 max_sessions: Optional[int] = None) -> UserTBG:
    """Ingest sessions in order -> belief-state. Requires op/ref + evidence_type ON
    (set by the caller via env). Returns the UserTBG."""
    os.environ.setdefault("TBG_OPREF", "1")
    os.environ.setdefault("TBG_EVIDENCE_TYPE", "1")
    from tbg_extractor import extract_tbg_delta

    engine = engine or TBGEngine(db_pool=_MockDB())
    tbg = UserTBG(user_id=str(conversation.get("id", "evo")))
    sessions = conversation["dialog_history"]
    if max_sessions:
        sessions = sessions[:max_sessions]
    sign_log = [] if _DUMP_SIGNS else None
    turn = 0
    for sess in sessions:
        user_text, assistant_text = session_texts(sess)  # already budget-limited
        if not user_text:
            tbg.message_count += 1
            continue
        delta = await extract_tbg_delta(
            user_text=user_text, assistant_text=assistant_text,
            existing_tbg_summary=tbg.summary(), existing_label_to_uuid={},
            llm_call_fn=llm_fn, tbg=tbg,
        )
        if delta:
            before = ({nid: n.confidence for nid, n in tbg.nodes.items()}
                      if _DUMP_SIGNS else None)
            tbg = engine.apply_delta(tbg, delta)
            if _DUMP_SIGNS:
                turn += 1
                _log_turn(sign_log, turn, user_text, delta, before, tbg)
        else:
            tbg.message_count += 1
    if _DUMP_SIGNS:
        with open(f"tbg_signs_{tbg.user_id}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sign_log))
    return tbg


def _traj(node) -> str:
    hist = node.confidence_history or []
    pts = [c for _, c in hist][-6:]
    if len(pts) < 2:
        return ""
    return " → ".join(f"{int(p*100)}%" for p in pts)


# max_beliefs equalized pre-run, not tuned: TBG render is capped at REPR_BUDGET
# (in esmem_gate) like every other arm; 40 lets the dense belief list fill toward
# that shared budget so TBG's representation VOLUME is comparable to the tracker D
# digest (~5200 chars) instead of the prior ~1700 (16 beliefs) handicap. Graph
# holds up to MAX_NODES=50.
def render_state(tbg: UserTBG, *, max_beliefs: int = 40) -> str:
    """Compact belief-state for the answering LLM: beliefs, trajectories,
    conflicts, turning points. Deterministic, no LLM."""
    nodes = sorted(tbg.active_nodes(0.3), key=lambda n: n.confidence, reverse=True)[:max_beliefs]
    lines: List[str] = []

    lines.append("CURRENT USER BELIEFS / STATES (confidence):")
    if not nodes:
        lines.append("  (none extracted)")
    for n in nodes:
        t = _traj(n)
        traj = f"  [trajectory: {t}]" if t else ""
        lines.append(f"  - {n.label} [{n.category}] {n.confidence:.0%}{traj}")

    # Conflicts. Default (TBG_RANK_RENDER=1) = rank-based, no absolute floor,
    # confidence shown ordinally (v1.1). TBG_RANK_RENDER=0 restores the v1.0 gate
    # format (absolute >=0.5 floor, no confidence number) — the single toggled
    # variable for the rank-OFF ablation. Read dynamically; default byte-identical.
    _OPP = ("blocks", "contradicts", "conflicts_with")
    _rank = os.getenv("TBG_RANK_RENDER", "1") == "1"
    if _rank:
        conflicts = sorted((e for e in tbg.edges.values() if e.relation in _OPP),
                           key=lambda e: e.confidence, reverse=True)
    else:
        conflicts = [e for e in tbg.edges.values()
                     if e.relation in _OPP and e.confidence >= 0.5]
    if conflicts:
        lines.append("\nCONFLICTS / TENSIONS:")
        for e in conflicts[:6]:
            s = tbg.nodes.get(e.source_id); t = tbg.nodes.get(e.target_id)
            if s and t:
                line = f"  - {s.label}  <-X->  {t.label}"
                if _rank:
                    line += f"  (conf {e.confidence:.0%})"
                lines.append(line)

    # Turning points (state shifts)
    if tbg.turning_points:
        lines.append("\nTURNING POINTS (where the user's state shifted):")
        for tp in tbg.turning_points:
            top = ", ".join(tp.top_nodes[:3])
            lines.append(f"  - around update #{tp.message_count}: {top} "
                         f"(magnitude {tp.cascade_magnitude:.2f})")

    return "\n".join(lines)


class _MockDB:
    async def fetchrow(self, *a, **kw): return None
    async def execute(self, *a, **kw): pass
    async def fetch(self, *a, **kw): return []
