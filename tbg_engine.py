"""
TBG Engine v5.0.0
- Log-odds Bayesian update
- Single-pass graph propagation (no cycle runaway)
- Cascade propagation: multi-hop BFS when delta > CASCADE_THRESHOLD
- Delta computed in logit space (mathematically consistent)
- Explicit nodes survive prune over inferred at equal confidence
- Forced snapshots after long idle period
- Cromwell's Rule: CONFIDENCE_MAX = 0.92 hard ceiling
- Priester & Petty (1996): ambivalence cap via pos/neg evidence accumulation
- De Finetti coherence: conflict pairs normalized to C_max sum
- get_insight: "Recent shift" sorted by delta magnitude; "Ambivalent" via pos/neg evidence
"""
import asyncio
import math
import logging
import os
import sys
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from asyncpg import Pool

from tbg_schema import UserTBG, BeliefNode, BeliefEdge, TBGDelta, ConfidenceSnapshot, TurningPoint
from tbg_telemetry import emit as _tele_emit, telemetry_enabled

logger = logging.getLogger(__name__)

# =============================================================================
# MATH
# =============================================================================

EPS = 1e-8

def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))

def sigmoid(x: float) -> float:
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# =============================================================================
# CONFIG
# =============================================================================

HALF_LIFE_DAYS = 30
CATEGORY_HALF_LIFE_DAYS = {
    "mood": 7,
    "fears": 14,
    "goals": 21,
    "relationships": 30,
    "finances": 30,
    "career": 40,
    "values": 60,
    "identity": 90,
}
BASELINE = 0.5
BASELINE_LOGIT = logit(BASELINE)
MIN_DECAY_INTERVAL_DAYS = 0.01

# R3: deterministic decay via logical clock instead of real-world time.
_DECAY_USE_LOGICAL_CLOCK = os.getenv("TBG_DECAY_USE_LOGICAL_CLOCK", "0") == "1"
# Turns equivalent to MIN_DECAY_INTERVAL_DAYS=0.01: assuming 1 turn ≈ 1 minute,
# 0.01 days ≈ 14 turns. Calibrated: don't decay within a moderate-length session.
MIN_DECAY_INTERVAL_TURNS = int(os.getenv("TBG_DECAY_MIN_INTERVAL_TURNS", "14"))
# Conversion: how many "logical days" one turn represents for decay rate math.
# Default: 1 turn = 0.001 logical days (~1.5 minutes equivalent for half-life math).
TURNS_TO_LOGICAL_DAYS = float(os.getenv("TBG_DECAY_TURNS_PER_DAY", "0.001"))

# R7: Group-aware deduplication for opposition relations.
# contradicts/conflicts_with/blocks describe the same logical relation
# at different abstraction levels. Without this, LLM-raw "contradicts"
# and SDL-generated "conflicts_with" coexist between same node pair,
# inflating dissonance scores in dissonance_engine.
_OPPOSITION_RELATIONS = frozenset({"contradicts", "conflicts_with", "blocks"})
_FIX_DOUBLE_EDGES = os.getenv("TBG_FIX_DOUBLE_EDGES", "0") == "1"

# E1: turning-point detection must ignore pure passive decay. _apply_decay sets
# confidence_prev on EVERY node, so a decay-only turn otherwise counts drift-to-
# baseline as "cascade" and can fabricate a turning point. When ON, only nodes
# that received direct evidence this turn seed the turning-point magnitude.
# Default OFF -> gate path byte-identical until a v1.1 decision.
_FIX_DECAY_TP = os.getenv("TBG_FIX_DECAY_TP", "0") == "1"

GRAPH_DAMPING = 0.7

EVIDENCE_WEIGHTS = {
    "strong_pos":  2.2,
    "medium_pos":  0.85,
    "neutral":     0.0,
    "medium_neg": -0.85,
    "strong_neg": -2.2,
}

# --- Cromwell's Rule ---
# No belief may reach certainty. Hard ceiling on all evidence updates.
CONFIDENCE_MAX = 0.92

# --- Priester & Petty (1996) ambivalence ---
# When both sides have accumulated evidence, max confidence is capped:
#   effective_max = CONFIDENCE_MAX * (1 - AMBIV_WEIGHT * ambivalence)
# At pos=neg=AMBIV_SCALE: ambivalence≈0.76 → effective_max≈0.71
AMBIV_SCALE  = 3.0
AMBIV_WEIGHT = 0.30

# --- De Finetti coherence ---
# Competing beliefs must satisfy sum ≤ C_max (proportional normalization).
C_MAX_CONTRADICTS    = 1.0
C_MAX_CONFLICTS_WITH = 1.3

EDGE_WEIGHTS = {
    "causes":       0.6,
    "supports":     0.5,
    "motivates":    0.6,
    "blocks":      -0.7,
    "contradicts": -1.2,
}

MIN_CONFIDENCE = 0.1

# --- Thompson et al. (1995) oscillation clamping ---
# When a node has been BOTH reinforced and contradicted (oscillating belief),
# repeated contradictions should pull confidence toward moderate (0.5) rather
# than dragging it monotonically toward zero.
# Reference: Priester & Petty (1996) Gradual Threshold Model — subjective
# ambivalence rises when conflicting reactions are similar in magnitude.
#
# THOMPSON_BLEND: weight of raw update vs pull-to-moderate.
#   0.7 → 70% raw update, 30% pull toward THOMPSON_TARGET.
#   Lower → stronger pull, faster convergence to moderate.
# THOMPSON_TARGET: equilibrium confidence for oscillating beliefs.
#   0.5 = pure center (maximum ambivalence). Calibrate if needed.
# THOMPSON_MIN_HISTORY: minimum history points before oscillation is detectable.
THOMPSON_BLEND    = 0.7
THOMPSON_TARGET   = 0.5
THOMPSON_MIN_HIST = 3
MIN_EDGE_CONFIDENCE = 0.20  # edges below this are pruned — prevents conflict soup
MAX_NODES = 50
MAX_ARCHIVE_NODES = 150
MAX_EDGES = 60              # hard cap on total edges in graph
STALE_DAYS = 45

SNAPSHOT_EVERY = 10
FORCED_SNAPSHOT_DAYS = 7

LLM_TIMEOUT = 12.0

INSIGHT_MIN_CONFIDENCE = 0.55

# --- Turning point detection ---
# A turn is a "turning point" when the sum of |Δconfidence| across all nodes
# exceeds this threshold — captures moments of genuine cognitive cascade.
TURNING_POINT_THRESHOLD = 1.5
# Keep only the top-N most impactful turning points (by cascade magnitude).
TURNING_POINT_MAX_STORED = 3

# Cascade propagation config
# Trigger: logit delta > this threshold (≈ confidence delta > 0.17 at midrange)
CASCADE_THRESHOLD = 0.7
# Max BFS hops — prevents runaway in dense graphs
CASCADE_MAX_HOPS = 3
# Damping per hop — each hop attenuates the signal
CASCADE_HOP_DAMPING = 0.5
# Numerical epsilon for cascade frontier pruning
CASCADE_MIN_INFLUENCE = 1e-6


CONFIDENCE_HISTORY_MAX = int(os.getenv("TBG_HISTORY_CAP", "5"))  # points per node;
# env-tunable for v1.2 experiments (AMF/trajectory classes are throttled by the cap).
# Unset/"5" -> byte-identical to v1.0/v1.1.

# Turning point: measured as average Δconf per node (scale-invariant).
# 0.07 means: on average each node shifted 7% confidence in one turn.
TURNING_POINT_THRESHOLD_PER_NODE = 0.07

def _append_history(node: "BeliefNode", message_count: int, confidence: float):
    node.confidence_history.append([float(message_count), round(confidence, 3)])
    if len(node.confidence_history) > CONFIDENCE_HISTORY_MAX:
        node.confidence_history.pop(0)


# =============================================================================
# ENGINE
# =============================================================================

def _add_edge_dedup(tbg: UserTBG, edge: BeliefEdge) -> None:
    """Add edge to tbg.edges with opposition-group deduplication (R7)."""
    if not _FIX_DOUBLE_EDGES or edge.relation not in _OPPOSITION_RELATIONS:
        tbg.upsert_edge(edge)
        return

    # Check for existing opposition-group edge between same pair
    for existing_key, existing_edge in list(tbg.edges.items()):
        if (existing_edge.source_id == edge.source_id
                and existing_edge.target_id == edge.target_id
                and existing_edge.relation in _OPPOSITION_RELATIONS):
            # Keep the stronger one (highest confidence)
            if edge.confidence > existing_edge.confidence:
                del tbg.edges[existing_key]
                tbg.edges[edge.key] = edge
                logger.info(
                    f"R7 dedup: replaced {existing_edge.relation} "
                    f"(conf={existing_edge.confidence:.2f}) with {edge.relation} "
                    f"(conf={edge.confidence:.2f}) for "
                    f"{edge.source_id}->{edge.target_id}"
                )
            else:
                logger.info(
                    f"R7 dedup: kept {existing_edge.relation} "
                    f"(conf={existing_edge.confidence:.2f}), skipped {edge.relation} "
                    f"(conf={edge.confidence:.2f}) for "
                    f"{edge.source_id}->{edge.target_id}"
                )
            return
    # No conflict — add normally
    tbg.upsert_edge(edge)


class TBGEngine:
    def __init__(self, db_pool: Pool):
        self.db = db_pool

    # -------------------------------------------------------------------------
    # DB
    # -------------------------------------------------------------------------

    async def load(self, user_id: str) -> UserTBG:
        row = await self.db.fetchrow(
            "SELECT nodes_data, edges_data, message_count, last_sync, last_decay, "
            "turning_points_data, archive_data FROM user_tbg WHERE user_id = $1",
            user_id
        )
        if not row:
            return UserTBG(user_id=user_id)

        tbg = UserTBG(user_id=user_id, message_count=row["message_count"] or 0)

        if row["nodes_data"]:
            nodes_raw = dict(row["nodes_data"])
            # Unpack concept_registry packed alongside nodes (no schema change needed).
            concept_registry = nodes_raw.pop("__concept_registry__", {})
            if isinstance(concept_registry, dict):
                tbg.concept_registry = concept_registry
            # Unpack concept_aliases (Step 4 closed-vocab cache).
            concept_aliases = nodes_raw.pop("__concept_aliases__", {})
            if isinstance(concept_aliases, dict):
                tbg.concept_aliases = concept_aliases
            # Unpack R3 logical clock counters.
            tbg.turn_counter = int(nodes_raw.pop("__turn_counter__", 0))
            tbg.last_decay_turn = int(nodes_raw.pop("__last_decay_turn__", 0))
            for node_id, node_data in nodes_raw.items():
                node = BeliefNode(**node_data)
                tbg.nodes[node.id] = node

        if row["edges_data"]:
            for edge_key, edge_data in row["edges_data"].items():
                edge = BeliefEdge(**edge_data)
                tbg.edges[edge.key] = edge

        if row.get("archive_data"):
            for node_id, node_data in row["archive_data"].items():
                node = BeliefNode(**node_data)
                tbg.archive_nodes[node.id] = node

        if row["last_sync"]:
            tbg.last_sync = row["last_sync"]
        if row.get("last_decay"):
            tbg.last_decay = row["last_decay"]

        if row.get("turning_points_data"):
            tbg.turning_points = [TurningPoint(**tp) for tp in row["turning_points_data"]]

        tbg.rebuild_index()

        # Derived AMF cache. apply_delta() refreshes _amf_state after every
        # mutation, but it is NOT persisted (private attr, absent from model_dump).
        # A freshly loaded graph must therefore recompute it — otherwise readers
        # of a loaded-but-unmutated graph (e.g. /cognitive/directive) always see an
        # empty map and fall back to neutral defaults, permanently disabling the
        # grounding/calibration regimes. Cheap O(nodes) CPU pass.
        from amf_filter import compute_graph_amf
        tbg._amf_state = compute_graph_amf(tbg.nodes)

        return tbg

    async def save(self, tbg: UserTBG):
        tbg.last_sync = datetime.now(timezone.utc)

        nodes_data = {k: v.model_dump(mode="json") for k, v in tbg.nodes.items()}
        # Pack concept_registry into nodes_data — avoids DB schema change.
        if tbg.concept_registry:
            nodes_data["__concept_registry__"] = tbg.concept_registry
        # Pack concept_aliases (Step 4 closed-vocab cache) — same approach.
        if tbg.concept_aliases:
            nodes_data["__concept_aliases__"] = tbg.concept_aliases
        # Pack R3 logical clock counters.
        nodes_data["__turn_counter__"] = tbg.turn_counter
        nodes_data["__last_decay_turn__"] = tbg.last_decay_turn
        edges_data = {k: v.model_dump(mode="json") for k, v in tbg.edges.items()}
        turning_points_data = [tp.model_dump(mode="json") for tp in tbg.turning_points]
        archive_data = {k: v.model_dump(mode="json") for k, v in tbg.archive_nodes.items()}

        await self.db.execute(
            """
            INSERT INTO user_tbg (user_id, nodes_data, edges_data, message_count, last_sync, last_decay, turning_points_data, archive_data)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (user_id) DO UPDATE
            SET nodes_data = $2, edges_data = $3,
                message_count = $4, last_sync = $5, last_decay = $6,
                turning_points_data = $7, archive_data = $8
            """,
            tbg.user_id,
            nodes_data,
            edges_data,
            tbg.message_count,
            tbg.last_sync,
            tbg.last_decay,
            turning_points_data,
            archive_data,
        )

    async def save_snapshot(self, tbg: UserTBG, force: bool = False):
        snapshot = ConfidenceSnapshot.from_tbg(tbg)

        if force:
            await self.db.execute(
                """
                INSERT INTO tbg_history (user_id, message_count, snapshot_data, created_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, message_count) DO UPDATE
                SET snapshot_data = EXCLUDED.snapshot_data,
                    created_at = EXCLUDED.created_at
                """,
                tbg.user_id,
                snapshot.message_count,
                snapshot.model_dump(mode="json"),
                snapshot.timestamp
            )
        else:
            await self.db.execute(
                """
                INSERT INTO tbg_history (user_id, message_count, snapshot_data, created_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, message_count) DO NOTHING
                """,
                tbg.user_id,
                snapshot.message_count,
                snapshot.model_dump(mode="json"),
                snapshot.timestamp
            )

    async def load_last_snapshots(self, user_id: str, limit: int = 2) -> List[ConfidenceSnapshot]:
        rows = await self.db.fetch(
            """
            SELECT snapshot_data FROM tbg_history
            WHERE user_id = $1
            ORDER BY message_count DESC
            LIMIT $2
            """,
            user_id, limit
        )
        return [ConfidenceSnapshot(**row["snapshot_data"]) for row in rows]

    # -------------------------------------------------------------------------
    # CORE
    # -------------------------------------------------------------------------

    def apply_delta(self, tbg: UserTBG, delta: TBGDelta) -> UserTBG:
        now = datetime.now(timezone.utc)
        tbg.message_count += 1
        tbg.turn_counter += 1  # R3: logical clock

        self._apply_decay(tbg, now)

        # Track which nodes received direct evidence this step.
        # Only these can seed cascade — decay-touched nodes must not.
        evidence_modified: set = set()

        for node in delta.add_nodes:
            nid = self._update_node(tbg, node, now)
            if nid:
                evidence_modified.add(nid)

        for edge in delta.add_edges:
            if edge.source_id in tbg.nodes and edge.target_id in tbg.nodes:
                _add_edge_dedup(tbg, edge)

        for nid in delta.reinforce_ids:
            node = tbg.nodes.get(nid)
            if node:
                self._apply_evidence(node, "medium_pos", now, tbg.message_count)
                evidence_modified.add(nid)

        for nid in delta.contradict_ids:
            node = tbg.nodes.get(nid)
            if node:
                self._apply_evidence(node, "medium_neg", now, tbg.message_count)
                evidence_modified.add(nid)

        for nid in delta.strong_contradict_ids:
            node = tbg.nodes.get(nid)
            if node:
                self._apply_evidence(node, "strong_neg", now, tbg.message_count)
                evidence_modified.add(nid)

        # Cascade seeds come exclusively from evidence-modified nodes.
        # This prevents long-idle nodes (large decay delta) from triggering
        # a cascade wave as if they were new evidence.
        evidence_strong: Dict[str, float] = {}
        for nid in evidence_modified:
            node = tbg.nodes.get(nid)
            if node is None or node.confidence_prev is None:
                continue
            d = logit(node.confidence) - logit(node.confidence_prev)
            if abs(d) >= CASCADE_THRESHOLD:
                evidence_strong[nid] = d

        # Single-pass for all changes — restricted to evidence-modified sources only
        self._apply_graph_influence(tbg, now, evidence_modified)

        # Cascade for strong changes only — starts from targets of evidence nodes
        self._cascade_propagate(tbg, now, evidence_strong)

        # De Finetti: normalize conflict pairs so sum ≤ C_max.
        # Runs after all evidence + propagation so we see final values.
        _definetti_applied, _definetti_skipped = self._normalize_conflicts(tbg, now, evidence_modified)

        # Turning point: average |Δconf| per node (scale-invariant).
        # Per-node average prevents large graphs from triggering on every turn.
        if _FIX_DECAY_TP:
            # Only evidence-modified nodes count — pure-decay drift is excluded.
            changed_nodes = [
                tbg.nodes[nid] for nid in evidence_modified
                if nid in tbg.nodes and tbg.nodes[nid].confidence_prev is not None
            ]
        else:
            changed_nodes = [
                n for n in tbg.nodes.values() if n.confidence_prev is not None
            ]
        cascade_magnitude = sum(abs(n.confidence - n.confidence_prev) for n in changed_nodes)
        cascade_per_node = cascade_magnitude / max(1, len(changed_nodes))
        if cascade_per_node >= TURNING_POINT_THRESHOLD_PER_NODE:
            top_changed = sorted(
                changed_nodes,
                key=lambda n: abs(n.confidence - n.confidence_prev),
                reverse=True,
            )[:3]
            tp = TurningPoint(
                message_count=tbg.message_count,
                cascade_magnitude=round(cascade_magnitude, 3),
                top_nodes=[n.label for n in top_changed],
            )
            tbg.turning_points.append(tp)
            tbg.turning_points.sort(key=lambda t: t.cascade_magnitude, reverse=True)
            tbg.turning_points = tbg.turning_points[:TURNING_POINT_MAX_STORED]
            logger.info(
                f"TBG turning point msg#{tbg.message_count}: "
                f"cascade={cascade_magnitude:.3f} top={[n.label for n in top_changed[:2]]}"
            )

        self._prune(tbg, now)

        from amf_filter import compute_graph_amf
        tbg._amf_state = compute_graph_amf(tbg.nodes)

        # Telemetry: per-turn dynamics (free side-product; guarded so OFF does zero
        # extra work — no label loop, no emit). Reads only already-computed locals.
        if telemetry_enabled():
            _ev_labels = []
            _ev_pruned = 0
            for _nid in evidence_modified:
                _n = tbg.nodes.get(_nid)
                if _n is not None:
                    _ev_labels.append(_n.label)
                else:
                    _ev_pruned += 1  # id evicted by prune this turn — count, don't silently drop
            _tele_emit({
                "event": "turn_dynamics",
                "user_id": tbg.user_id,
                "msg_count": tbg.message_count,
                "evidence_modified": _ev_labels,
                "evidence_modified_pruned": _ev_pruned,
                "evidence_strong_count": len(evidence_strong),
                "cascade_magnitude": round(cascade_magnitude, 3),
                "definetti_applied": _definetti_applied,
                "definetti_skipped": _definetti_skipped,
            })

        return tbg

    def _apply_decay(self, tbg: UserTBG, now: datetime):
        if _DECAY_USE_LOGICAL_CLOCK:
            # R3: logical clock path — deterministic, wall-clock independent
            dt_turns = tbg.turn_counter - tbg.last_decay_turn
            if dt_turns < MIN_DECAY_INTERVAL_TURNS:
                return
            dt = dt_turns * TURNS_TO_LOGICAL_DAYS
            tbg.last_decay_turn = tbg.turn_counter
        else:
            # Default: real-world clock
            dt = (now - tbg.last_decay).total_seconds() / 86400
            if dt < MIN_DECAY_INTERVAL_DAYS:
                return
            tbg.last_decay = now

        for node in tbg.nodes.values():
            half_life = CATEGORY_HALF_LIFE_DAYS.get(node.category, HALF_LIFE_DAYS)
            decay_factor = math.exp(-math.log(2) / half_life)
            decay = decay_factor ** dt
            old_conf = node.confidence
            l = logit(old_conf)
            l_new = l * decay + BASELINE_LOGIT * (1 - decay)
            new_conf = sigmoid(l_new)

            node.confidence_prev = old_conf
            node.confidence = new_conf

            # Decay accumulated evidence — old evidence should gradually
            # lose its grip on ambivalence, same half-life as confidence.
            node.pos_evidence *= decay
            node.neg_evidence *= decay
            # updated_at is intentionally NOT touched here.
            # Decay is passive forgetting — it must not appear as "recently active"
            # in get_insight() recency filters or cascade seeding.

    def _update_node(self, tbg: UserTBG, node: BeliefNode, now: datetime) -> Optional[str]:
        """Apply one belief node from a delta. Returns the node ID that was changed."""
        # concept_id lookup takes priority over label lookup:
        # if the concept already exists under a different label, reinforce it in-place.
        existing = (
            tbg.get_node_by_concept_id(node.concept_id)
            if node.concept_id
            else None
        ) or tbg.get_node_by_label(node.label)

        if not existing:
            # Check archive
            archived = (
                tbg.get_archived_node_by_concept_id(node.concept_id)
                if node.concept_id
                else tbg.get_archived_node_by_label(node.label)
            )
            if archived:
                tbg.archive_nodes.pop(archived.id, None)
                archived.confidence_prev = archived.confidence
                tbg.set_node(archived)
                existing = archived

        if existing:
            w = EVIDENCE_WEIGHTS.get(node.evidence_type or "neutral", 0.0)
            if w > 0:
                existing.pos_evidence += w
            elif w < 0:
                existing.neg_evidence += abs(w)

            ambivalence = math.tanh(
                min(existing.pos_evidence, existing.neg_evidence) / AMBIV_SCALE
            )
            effective_max = CONFIDENCE_MAX * (1.0 - AMBIV_WEIGHT * ambivalence)

            old_conf = existing.confidence
            new_conf = min(sigmoid(logit(old_conf) + w), effective_max)

            existing.confidence_prev = old_conf
            existing.confidence = new_conf
            existing.updated_at = now
            _append_history(existing, tbg.message_count, new_conf)
            return existing.id
        else:
            node.confidence_prev = BASELINE
            tbg.set_node(node)
            _append_history(node, tbg.message_count, node.confidence)
            return node.id

    def _apply_evidence(self, node: BeliefNode, evidence_type: str, now: datetime, message_count: int = 0):
        weight = EVIDENCE_WEIGHTS.get(evidence_type, 0.0)
        if weight == 0.0:
            return

        if weight > 0:
            node.pos_evidence += weight
        else:
            node.neg_evidence += abs(weight)

        ambivalence = math.tanh(min(node.pos_evidence, node.neg_evidence) / AMBIV_SCALE)
        effective_max = CONFIDENCE_MAX * (1.0 - AMBIV_WEIGHT * ambivalence)

        old_conf = node.confidence
        new_conf = min(sigmoid(logit(old_conf) + weight), effective_max)

        # Thompson et al. (1995) oscillation clamping — contradiction path only.
        # When a node has been both pushed up and pulled down (oscillating),
        # blend the raw update toward THOMPSON_TARGET (0.5) to prevent
        # monotonic decay to the floor.  Fixes OCS confidence over-reduction:
        # "wants to leave NYC" reaching conf=0.13 after 12 turns of alternating
        # reinforce/contradict pairs.
        #
        # Condition: weight < 0 (contradiction) AND history shows ≥1 up + ≥1 down.
        if weight < 0 and len(node.confidence_history) >= THOMPSON_MIN_HIST:
            recent_conf = [c for _, c in node.confidence_history[-(THOMPSON_MIN_HIST + 2):]]
            ups   = sum(1 for i in range(1, len(recent_conf)) if recent_conf[i] > recent_conf[i - 1])
            downs = sum(1 for i in range(1, len(recent_conf)) if recent_conf[i] < recent_conf[i - 1])
            if ups >= 1 and downs >= 1:
                # Oscillating belief: pull toward moderate instead of flooring
                new_conf = THOMPSON_BLEND * new_conf + (1.0 - THOMPSON_BLEND) * THOMPSON_TARGET
                # Still respect effective_max ceiling and absolute floor
                new_conf = max(MIN_CONFIDENCE, min(new_conf, effective_max))

        node.confidence_prev = old_conf
        node.confidence = new_conf
        node.updated_at = now
        _append_history(node, message_count, new_conf)

    # -------------------------------------------------------------------------
    # GRAPH INFLUENCE — SINGLE-PASS
    # -------------------------------------------------------------------------

    def _apply_graph_influence(self, tbg: UserTBG, now: datetime, evidence_modified: set):
        """
        Single-pass propagation.
        Only evidence-modified nodes act as sources — decay-touched nodes are excluded.
        Delta is in logit space — consistent with all other updates.
        One pass only: no iterative amplification in cycles.
        """

        # Step 1: collect evidence-modified nodes with their logit delta.
        # Restricting to evidence_modified prevents decay (passive forgetting)
        # from propagating through the graph as if it were real evidence.
        pending: Dict[str, float] = {}

        for node_id in evidence_modified:
            node = tbg.nodes.get(node_id)
            if node is None or node.confidence_prev is None:
                continue
            delta = logit(node.confidence) - logit(node.confidence_prev)
            if abs(delta) > 1e-6:
                pending[node_id] = delta

        if not pending:
            return

        # Step 2: one pass from sources to targets
        updates: Dict[str, float] = {}

        for edge in tbg.edges.values():
            if edge.source_id not in pending:
                continue
            if edge.target_id not in tbg.nodes:
                continue
            if edge.target_id == edge.source_id:
                continue

            delta = pending[edge.source_id]
            w = EDGE_WEIGHTS.get(edge.relation, 0.0)
            influence = w * delta * GRAPH_DAMPING

            if abs(influence) > 1e-6:
                updates[edge.target_id] = updates.get(edge.target_id, 0.0) + influence

        # Step 3: apply updates
        for node_id, delta_logit in updates.items():
            node = tbg.nodes.get(node_id)
            if not node:
                continue

            old_conf = node.confidence
            new_conf = min(sigmoid(logit(old_conf) + delta_logit), CONFIDENCE_MAX)

            node.confidence_prev = old_conf
            node.confidence = new_conf
            node.updated_at = now

    # -------------------------------------------------------------------------
    # CASCADE PROPAGATION — MULTI-HOP BFS
    # -------------------------------------------------------------------------

    def _cascade_propagate(
        self,
        tbg: UserTBG,
        now: datetime,
        evidence_strong: Dict[str, float],
    ):
        """
        Multi-hop cascade for strong belief changes.

        Single-pass (_apply_graph_influence) already covered hop-1:
            evidence_node → direct_neighbor

        Cascade picks up from hop-2 onward by seeding the frontier at the
        direct neighbors of evidence nodes (not the evidence nodes themselves).
        This prevents the same A→B edge from being applied twice.

        Diffusion formula:
            influence^(k+1) = influence^(k) * edge_weight * CASCADE_HOP_DAMPING
        Truncated at CASCADE_MAX_HOPS total hops from evidence nodes.
        """
        if not evidence_strong:
            return

        # Build outgoing adjacency: source_id -> [(target_id, normalized_edge_w)].
        # Normalize by outgoing L1 so propagation stays bounded.
        adjacency: Dict[str, List[tuple]] = {}
        outgoing_l1: Dict[str, float] = {}
        for edge in tbg.edges.values():
            w = EDGE_WEIGHTS.get(edge.relation, 0.0)
            if w == 0.0:
                continue
            if edge.source_id not in tbg.nodes or edge.target_id not in tbg.nodes:
                continue
            edge_w = w * edge.confidence
            adjacency.setdefault(edge.source_id, []).append((edge.target_id, edge_w))
            outgoing_l1[edge.source_id] = outgoing_l1.get(edge.source_id, 0.0) + abs(edge_w)

        for source_id, edges in adjacency.items():
            norm = max(1.0, outgoing_l1.get(source_id, 0.0))
            adjacency[source_id] = [(tid, ew / norm) for tid, ew in edges]

        # L1 clamp — guarantees bounded propagation on any graph topology.
        for source_id, edges in adjacency.items():
            l1 = sum(abs(ew) for _, ew in edges)
            if l1 > 1.0 + 1e-9:
                logger.warning(
                    f"TBG cascade: L1 norm {l1:.4f} > 1.0 for node {source_id[:8]}, clamping"
                )
                adjacency[source_id] = [(tid, ew / l1) for tid, ew in edges]

        # Build hop-1 frontier: targets of evidence nodes.
        # Single-pass already applied evidence→target; cascade continues from there.
        frontier: Dict[str, float] = {}
        for source_id, source_delta in evidence_strong.items():
            for target_id, edge_w in adjacency.get(source_id, []):
                influence = source_delta * edge_w * CASCADE_HOP_DAMPING
                if abs(influence) >= CASCADE_MIN_INFLUENCE:
                    frontier[target_id] = frontier.get(target_id, 0.0) + influence

        if not frontier:
            return

        logger.debug(
            f"TBG cascade: {len(evidence_strong)} evidence sources, "
            f"{len(frontier)} hop-1 frontier nodes"
        )

        # BFS for hops 2..CASCADE_MAX_HOPS (frontier is already at hop-1).
        accumulated: Dict[str, float] = {}
        for _hop in range(1, CASCADE_MAX_HOPS):
            if not frontier:
                break

            next_frontier: Dict[str, float] = {}
            for source_id, influence in frontier.items():
                for target_id, edge_w in adjacency.get(source_id, []):
                    propagated = influence * edge_w * CASCADE_HOP_DAMPING
                    if abs(propagated) < CASCADE_MIN_INFLUENCE:
                        continue
                    accumulated[target_id] = accumulated.get(target_id, 0.0) + propagated
                    next_frontier[target_id] = next_frontier.get(target_id, 0.0) + propagated

            frontier = next_frontier

        if not accumulated:
            return

        cascade_count = 0
        for node_id, delta_logit in accumulated.items():
            node = tbg.nodes.get(node_id)
            if not node:
                continue

            old_conf = node.confidence
            new_conf = min(sigmoid(logit(old_conf) + delta_logit), CONFIDENCE_MAX)

            if abs(new_conf - old_conf) < CASCADE_MIN_INFLUENCE:
                continue

            node.confidence_prev = old_conf
            node.confidence = new_conf
            node.updated_at = now
            cascade_count += 1

        if cascade_count:
            logger.debug(f"TBG cascade: updated {cascade_count} nodes")

    # -------------------------------------------------------------------------
    # DE FINETTI COHERENCE — conflict pair normalization
    # -------------------------------------------------------------------------

    def _normalize_conflicts(self, tbg: UserTBG, now: datetime, evidence_modified: set):
        """
        Enforce De Finetti coherence: competing beliefs must sum ≤ C_max.

        For each conflict/contradicts edge, if src.confidence + tgt.confidence > C_max,
        scale both proportionally — preserving relative strength, fixing the sum.

        C_max differs by relation semantics:
          contradicts    → 1.0 (mutually exclusive hypotheses)
          conflicts_with → 1.3 (partially independent, can both be somewhat true)
        """
        edges_total = sum(
            1 for e in tbg.edges.values()
            if e.relation in ("contradicts", "conflicts_with")
        )
        skipped_missing = 0
        skipped_evidence = 0
        skipped_below_c_max = 0
        applied = 0

        for edge in tbg.edges.values():
            if edge.relation not in ("contradicts", "conflicts_with"):
                continue
            if edge.confidence < 0.4:
                continue  # low-confidence edges don't enforce coherence

            if (edge.source_id not in evidence_modified
                    and edge.target_id not in evidence_modified):
                skipped_evidence += 1
                continue

            src = tbg.nodes.get(edge.source_id)
            tgt = tbg.nodes.get(edge.target_id)
            if not src or not tgt:
                skipped_missing += 1
                continue

            c_max = (
                C_MAX_CONTRADICTS if edge.relation == "contradicts"
                else C_MAX_CONFLICTS_WITH
            )
            total = src.confidence + tgt.confidence
            if total <= c_max:
                skipped_below_c_max += 1
                continue

            scale = c_max / total
            src.confidence_prev = src.confidence
            tgt.confidence_prev = tgt.confidence
            src.confidence = round(src.confidence * scale, 4)
            tgt.confidence = round(tgt.confidence * scale, 4)
            src.updated_at = now
            tgt.updated_at = now
            applied += 1
            logger.debug(
                f"De Finetti [{edge.relation}]: "
                f"{src.label}[{src.category}]({src.confidence_prev:.2f}→{src.confidence:.2f}) + "
                f"{tgt.label}[{tgt.category}]({tgt.confidence_prev:.2f}→{tgt.confidence:.2f}) "
                f"sum={total:.2f}→{c_max:.2f}"
            )

        if applied > 0 or skipped_missing > 0:
            logger.info(
                f"De Finetti summary: applied={applied} "
                f"skipped_missing_node={skipped_missing} "
                f"skipped_no_evidence={skipped_evidence} "
                f"skipped_below_c_max={skipped_below_c_max}"
            )
        # Return the counts this method already computes so apply_delta can surface
        # them as telemetry. Private, single-caller (apply_delta) — additive; the
        # graph-mutation logic above is unchanged.
        return applied, (skipped_missing + skipped_evidence + skipped_below_c_max)

    # -------------------------------------------------------------------------
    # PRUNE
    # -------------------------------------------------------------------------

    def _prune(self, tbg: UserTBG, now: datetime):
        if tbg.message_count < 5:
            return  # cold start: no pruning, let the graph build

        cutoff = now - timedelta(days=STALE_DAYS)
        # identity/values nodes survive confidence-floor pruning.
        # Belief perseverance: a contradicted identity remains latent, not erased.
        # These nodes are still removed if stale (no updates in STALE_DAYS).
        _PERSISTENT_CATS = frozenset(("identity", "values"))

        def _archive_node(n: BeliefNode):
            existing = tbg.archive_nodes.get(n.id)
            if existing and existing.confidence > n.confidence:
                return
            tbg.archive_nodes[n.id] = n

        filtered_nodes = {}
        for node_id, node in tbg.nodes.items():
            if node.confidence <= MIN_CONFIDENCE:
                if node.category not in _PERSISTENT_CATS:
                    _archive_node(node)
                    continue
            if node.updated_at <= cutoff and node.confidence < 0.6:
                _archive_node(node)
                continue
            filtered_nodes[node_id] = node

        # concept_id dedup: nodes sharing the same concept_id are the same concept.
        # Keep highest-confidence node, accumulate evidence from duplicates.
        # This fires when: two labels merged at different turns, or LLM ignored LABEL REUSE.
        by_concept: Dict[str, List[str]] = {}
        for nid, node in filtered_nodes.items():
            if node.concept_id:
                by_concept.setdefault(node.concept_id, []).append(nid)

        for concept_id, nids in by_concept.items():
            if len(nids) <= 1:
                continue
            winner_id = max(nids, key=lambda nid: filtered_nodes[nid].confidence)
            winner = filtered_nodes[winner_id]
            for nid in nids:
                if nid == winner_id:
                    continue
                loser = filtered_nodes.pop(nid)
                winner.pos_evidence += loser.pos_evidence
                winner.neg_evidence += loser.neg_evidence
                _archive_node(loser)
                logger.info(
                    f"concept_id merge: {loser.label!r} → {winner.label!r} [{concept_id}]"
                )

        if len(filtered_nodes) > MAX_NODES:
            # E5: confidence is primary; "explicit" only breaks ties at equal
            # confidence (documented tie-break, not a blanket override of conf).
            sorted_nodes = sorted(
                filtered_nodes.items(),
                key=lambda x: (x[1].confidence, x[1].source == "explicit"),
                reverse=True
            )
            for node_id, node in sorted_nodes[MAX_NODES:]:
                _archive_node(node)
            filtered_nodes = dict(sorted_nodes[:MAX_NODES])

        tbg.nodes = filtered_nodes
        tbg.rebuild_index()

        # Trim archive if it exceeds limit
        if len(tbg.archive_nodes) > MAX_ARCHIVE_NODES:
            sorted_archive = sorted(
                tbg.archive_nodes.values(),
                key=lambda n: n.confidence,
                reverse=True
            )
            tbg.archive_nodes = {n.id: n for n in sorted_archive[:MAX_ARCHIVE_NODES]}

        valid_ids = set(tbg.nodes.keys())

        # Step 1: update edge confidence to weakest endpoint
        surviving_edges = {}
        for k, e in tbg.edges.items():
            if e.source_id not in valid_ids or e.target_id not in valid_ids:
                continue
            src = tbg.nodes[e.source_id]
            tgt = tbg.nodes[e.target_id]
            e.confidence = min(e.confidence, src.confidence, tgt.confidence)
            if e.confidence >= MIN_EDGE_CONFIDENCE:
                surviving_edges[k] = e

        # Step 2: hard cap — keep strongest edges if over MAX_EDGES
        if len(surviving_edges) > MAX_EDGES:
            sorted_edges = sorted(
                surviving_edges.items(),
                key=lambda x: x[1].confidence,
                reverse=True,
            )
            surviving_edges = dict(sorted_edges[:MAX_EDGES])

        tbg.edges = surviving_edges

    # -------------------------------------------------------------------------
    # INSIGHT
    # -------------------------------------------------------------------------

    @staticmethod
    def get_drift_labels(tbg: UserTBG, snapshots: List[ConfidenceSnapshot]) -> List[str]:
        """
        Compare current TBG against the most recent snapshot.
        Returns labels of beliefs that shifted >= 0.15 in absolute confidence,
        sorted by magnitude, up to 3.
        Pure Python — no IO. Returns [] if no snapshots or empty graph.
        """
        if not snapshots or not tbg.nodes:
            return []

        latest = snapshots[0]
        shifted = []

        for node in tbg.nodes.values():
            if node.confidence < INSIGHT_MIN_CONFIDENCE:
                continue
            snap_conf = latest.state.get(node.category, {}).get(node.label)
            if snap_conf is None:
                continue
            delta = abs(node.confidence - snap_conf)
            if delta >= 0.15:
                arrow = "↑" if node.confidence > snap_conf else "↓"
                shifted.append((delta, f"{node.label}{arrow}"))

        shifted.sort(reverse=True)
        return [label for _, label in shifted[:3]]

    def get_insight(self, tbg: UserTBG, snapshots: Optional[List[ConfidenceSnapshot]] = None) -> str:
        """Build a belief-state summary from the TBG graph.
        Pure Python — no IO, no LLM call. Safe to call synchronously.

        Args:
            tbg:       the user's current belief graph
            snapshots: optional list of recent ConfidenceSnapshots (newest first).
                       When provided, a "Shifting:" section is added with beliefs
                       that changed significantly since the last snapshot.
        """
        if not tbg.nodes:
            return ""

        parts = []

        strong = sorted(
            [n for n in tbg.nodes.values() if n.confidence >= INSIGHT_MIN_CONFIDENCE],
            key=lambda n: n.confidence,
            reverse=True
        )[:4]

        if strong:
            by_cat: Dict[str, List[str]] = {}
            for n in strong:
                by_cat.setdefault(n.category, []).append(n.label)
            cat_str = "; ".join(
                f"{cat}: {', '.join(labels[:2])}" for cat, labels in by_cat.items()
            )
            parts.append(f"Key beliefs: {cat_str}")

        # "Recent shift": beliefs that changed most in the last update pass
        # Sort by absolute confidence delta — not just recency
        recently_shifted = [
            n for n in tbg.nodes.values()
            if n.confidence_prev is not None
            and abs(n.confidence - n.confidence_prev) > 0.05
            and n.confidence >= INSIGHT_MIN_CONFIDENCE
        ]
        recently_shifted.sort(key=lambda n: abs(n.confidence - n.confidence_prev), reverse=True)
        if recently_shifted:
            shift_labels = ", ".join(n.label for n in recently_shifted[:3])
            parts.append(f"Recent shift: {shift_labels}")

        conflicts = []
        for edge in tbg.edges.values():
            if edge.relation in ("blocks", "contradicts", "conflicts_with"):
                if edge.confidence >= INSIGHT_MIN_CONFIDENCE:
                    src = tbg.nodes.get(edge.source_id)
                    tgt = tbg.nodes.get(edge.target_id)
                    if src and tgt:
                        if src.confidence >= INSIGHT_MIN_CONFIDENCE and tgt.confidence >= INSIGHT_MIN_CONFIDENCE:
                            conflicts.append(f"Conflict: {src.label} vs {tgt.label}")

        if conflicts:
            parts.extend(conflicts[:2])

        # Ambivalent: beliefs with substantial evidence on both sides
        ambivalent = [
            n for n in tbg.nodes.values()
            if n.pos_evidence > 0.5 and n.neg_evidence > 0.5 and n.confidence >= 0.30
        ]
        if ambivalent:
            ambivalent.sort(key=lambda n: min(n.pos_evidence, n.neg_evidence), reverse=True)
            amb_labels = ", ".join(n.label for n in ambivalent[:2])
            parts.append(f"Ambivalent: {amb_labels}")

        # Snapshot-based drift: beliefs that shifted significantly since last snapshot
        if snapshots:
            drift = self.get_drift_labels(tbg, snapshots)
            if drift:
                parts.append(f"Shifting: {', '.join(drift)}")

        # Turning points: report the largest cascade on record
        if tbg.turning_points:
            tp = tbg.turning_points[0]  # highest cascade_magnitude
            nodes_str = ", ".join(tp.top_nodes[:2])
            parts.append(f"Turning point @msg{tp.message_count}: {nodes_str}")

        return " | ".join(parts) if parts else ""


# =============================================================================
# BACKGROUND TASK
# =============================================================================

async def update_tbg_background(
    user_id: str,
    user_text: str,
    assistant_text: str,
    engine: TBGEngine,
    llm_call_fn
):
    try:
        from tbg_extractor import extract_tbg_delta

        tbg = await engine.load(user_id)

        if tbg.last_sync:
            now = datetime.now(timezone.utc)
            dt = (now - tbg.last_sync).total_seconds() / 86400
            if dt > FORCED_SNAPSHOT_DAYS:
                await engine.save_snapshot(tbg, force=True)
                logger.info(f"TBG: forced snapshot for {user_id[:8]}")

        existing_summary = tbg.summary()
        existing_label_to_uuid = {
            node.label.lower(): node_id
            for node_id, node in tbg.nodes.items()
        }

        delta = await asyncio.wait_for(
            extract_tbg_delta(
                user_text=user_text,
                assistant_text=assistant_text,
                existing_tbg_summary=existing_summary,
                existing_label_to_uuid=existing_label_to_uuid,
                llm_call_fn=llm_call_fn,
                tbg=tbg
            ),
            timeout=LLM_TIMEOUT
        )

        if delta is None:
            return

        tbg = engine.apply_delta(tbg, delta)
        await engine.save(tbg)

        if tbg.message_count % SNAPSHOT_EVERY == 0:
            await engine.save_snapshot(tbg)

        logger.info(
            f"TBG ok user={user_id[:8]} "
            f"nodes={len(tbg.nodes)} edges={len(tbg.edges)} "
            f"msg#{tbg.message_count}"
        )

    except asyncio.TimeoutError:
        logger.warning(f"TBG: LLM timeout for user {user_id[:8]}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"TBG update failed for {user_id[:8]}: {e}", exc_info=True)
