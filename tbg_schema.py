"""
TBG Schema v5.0.0
- BeliefNode: pos_evidence / neg_evidence — Priester & Petty (1996) ambivalence model
- TBGDelta: strong_contradict_ids — explicit reversals of high-confidence beliefs
- UserTBG: _label_index via PrivateAttr — no shared mutable class default
"""
from datetime import datetime, timezone
from typing import Dict, Optional, List
from pydantic import BaseModel, Field, PrivateAttr
import uuid

from user_state import UserAxisState


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def short_uuid() -> str:
    return str(uuid.uuid4())[:12]


class BeliefNode(BaseModel):
    id: str = Field(default_factory=short_uuid)
    label: str
    category: str
    # Stable semantic identifier: survives label rewording.
    # Format: "{category}:{slug}", e.g. "career:corporate_lawyer".
    # Nodes sharing concept_id are the same concept — merged on prune.
    concept_id: Optional[str] = None
    # Verbatim source span stored by DeterministicExtractor for span-vs-span matching.
    # Not used by the LLM path. None on all nodes created via SemanticDecisionLayer.
    signal_span: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence_prev: Optional[float] = None
    source: str = "inferred"
    evidence_type: Optional[str] = None
    # Accumulated evidence weights (Priester & Petty 1996 ambivalence model).
    pos_evidence: float = 0.0
    neg_evidence: float = 0.0
    # Confidence trajectory: last 5 updates [[message_count, confidence], ...]
    confidence_history: List[List[float]] = Field(default_factory=list)
    # Semantic axis projections — computed once on node creation, immutable after.
    # Keys: "evaluation", "potency", "activity", "autonomy", "self_other"
    # Values: float in [-1.0, 1.0]. Empty dict = not yet computed (legacy nodes).
    axis_projection: Dict[str, float] = Field(default_factory=dict)
    # Ontological type — set by extractor, used for conflict gating.
    # "fact" | "state" | "value" | "intention" | "" (unknown/legacy)
    node_type: str = Field(default="")
    # Wheel-of-life domain — set by extractor routing (TBG_HER_ROUTING).
    # career|money|health|relationships|family|lifestyle|meaning|self|other | "" (legacy/off)
    # Used by her_resolver cross-domain VETO. Independent from `category` (8-way legacy taxonomy).
    domain: str = Field(default="")
    # Approach/avoid orientation — stored for diagnostics and L1; NOT used in SDL decisions.
    # approach|avoid|neutral | "" (legacy/off)
    stance: str = Field(default="")
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)

    model_config = {"extra": "ignore"}


class BeliefEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    confidence_prev: Optional[float] = None
    updated_at: datetime = Field(default_factory=now_utc)

    model_config = {"extra": "ignore"}

    @property
    def key(self) -> str:
        return f"{self.source_id}:{self.relation}:{self.target_id}"


class TBGDelta(BaseModel):
    add_nodes: List[BeliefNode] = []
    add_edges: List[BeliefEdge] = []
    reinforce_ids: List[str] = []
    contradict_ids: List[str] = []
    strong_contradict_ids: List[str] = []  # explicit reversal of high-confidence belief
    reasoning: str = ""

    model_config = {"extra": "ignore"}


class ConfidenceSnapshot(BaseModel):
    message_count: int = 0
    timestamp: datetime = Field(default_factory=now_utc)
    state: Dict[str, Dict[str, float]] = {}

    model_config = {"extra": "ignore"}

    @classmethod
    def from_tbg(cls, tbg: "UserTBG") -> "ConfidenceSnapshot":
        state: Dict[str, Dict[str, float]] = {}
        for node in tbg.nodes.values():
            state.setdefault(node.category, {})[node.label] = node.confidence
        return cls(message_count=tbg.message_count, state=state)

    def diff(self, earlier: "ConfidenceSnapshot") -> Dict[str, float]:
        result: Dict[str, float] = {}
        for cat, labels in self.state.items():
            for label, conf in labels.items():
                prev = earlier.state.get(cat, {}).get(label)
                if prev is not None:
                    d = conf - prev
                    if abs(d) > 0.05:
                        result[label] = d
                else:
                    result[label] = conf
        for cat, labels in earlier.state.items():
            for label, conf in labels.items():
                if label not in self.state.get(cat, {}):
                    result[label] = -conf
        return result


class TurningPoint(BaseModel):
    message_count: int
    cascade_magnitude: float        # sum of |Δconfidence| across all nodes
    top_nodes: List[str]            # top-3 node labels by individual delta

    model_config = {"extra": "ignore"}


class UserTBG(BaseModel):
    user_id: str
    nodes: Dict[str, BeliefNode] = Field(default_factory=dict)
    edges: Dict[str, BeliefEdge] = Field(default_factory=dict)
    message_count: int = 0
    last_sync: datetime = Field(default_factory=now_utc)
    last_decay: datetime = Field(default_factory=now_utc)
    # R3: logical clock for deterministic decay (TBG_DECAY_USE_LOGICAL_CLOCK=1).
    # When enabled, decay is applied based on turn count rather than wall-clock
    # time, ensuring identical results across reruns of the same dialogue.
    turn_counter: int = Field(default=0)
    last_decay_turn: int = Field(default=0)
    turning_points: List[TurningPoint] = Field(default_factory=list)
    # Canonical concept registry: concept_id → canonical_label.
    # Survives node pruning — enables cross-turn semantic deduplication.
    # Populated from existing nodes on load; grows as new concepts appear.
    concept_registry: Dict[str, str] = Field(default_factory=dict)
    # Phase 4: raw_label.lower() → canonical short form.
    # Populated by _canonicalize_labels() before SDL.resolve().
    # Survives turns and pruning. One LLM call/turn on cold cache, 0 on warm.
    label_aliases: Dict[str, str] = Field(default_factory=dict)
    # Step 4 (closed-vocab): raw_label.lower() → concept_id of existing concept.
    # Populated by _lookup_or_register and SDL MERGE/FLAG paths.
    # Independent from label_aliases (different semantics, different write-sites).
    # Per LINK-KG (2025) Prompt Cache pattern: deterministic O(1) match for
    # exact label repeats — bypasses cosine, eliminates path-dependence on repeats.
    concept_aliases: Dict[str, str] = Field(default_factory=dict)
    archive_nodes: Dict[str, BeliefNode] = Field(default_factory=dict)
    # Aggregate semantic axis state — recomputed on every apply_delta cycle.
    # Serialized into nodes_data as "__axis_state__" (no DB schema change needed).
    axis_state: UserAxisState = Field(default_factory=UserAxisState)

    _label_index: Dict[str, str] = PrivateAttr(default_factory=dict)
    _concept_index: Dict[str, str] = PrivateAttr(default_factory=dict)

    model_config = {"extra": "ignore"}

    def model_post_init(self, __context):
        self._rebuild_label_index()

    def _rebuild_label_index(self):
        self._label_index = {
            node.label.lower(): node_id
            for node_id, node in self.nodes.items()
        }
        self._concept_index = {
            node.concept_id: node_id
            for node_id, node in self.nodes.items()
            if node.concept_id
        }
        # Backfill registry from existing nodes (backward compat for old graphs).
        for node in self.nodes.values():
            if node.concept_id and node.concept_id not in self.concept_registry:
                self.concept_registry[node.concept_id] = node.label

    def rebuild_index(self):
        """Public alias for backward compatibility."""
        self._rebuild_label_index()

    # -- Nodes -----------------------------------------------------------------

    def set_node(self, node: BeliefNode):
        """Add or update node, keeping both indexes in sync."""
        if node.id in self.nodes:
            old = self.nodes[node.id]
            self._label_index.pop(old.label.lower(), None)
            if old.concept_id:
                self._concept_index.pop(old.concept_id, None)
        self.nodes[node.id] = node
        self._label_index[node.label.lower()] = node.id
        if node.concept_id:
            self._concept_index[node.concept_id] = node.id

    def remove_node(self, node_id: str):
        """Remove node and clean up both indexes."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            self._label_index.pop(node.label.lower(), None)
            if node.concept_id:
                self._concept_index.pop(node.concept_id, None)
            del self.nodes[node_id]

    def get_node_by_label(self, label: str) -> Optional[BeliefNode]:
        """O(1) lookup by label."""
        node_id = self._label_index.get(label.lower())
        return self.nodes.get(node_id) if node_id else None

    def get_node_id_by_label(self, label: str) -> Optional[str]:
        """O(1) id lookup by label."""
        return self._label_index.get(label.lower())

    def get_node_by_concept_id(self, concept_id: str) -> Optional[BeliefNode]:
        """O(1) lookup by concept_id."""
        node_id = self._concept_index.get(concept_id)
        return self.nodes.get(node_id) if node_id else None

    def get_node_id_by_concept_id(self, concept_id: str) -> Optional[str]:
        """O(1) id lookup by concept_id."""
        return self._concept_index.get(concept_id)

    def get_archived_node_by_concept_id(self, concept_id: str) -> Optional[BeliefNode]:
        for node in self.archive_nodes.values():
            if node.concept_id == concept_id:
                return node
        return None

    def get_archived_node_by_label(self, label: str) -> Optional[BeliefNode]:
        label_lower = label.lower()
        for node in self.archive_nodes.values():
            if node.label.lower() == label_lower:
                return node
        return None

    # -- Edges -----------------------------------------------------------------

    def upsert_edge(self, edge: BeliefEdge):
        """Add edge or update confidence if already exists."""
        existing = self.edges.get(edge.key)
        if existing:
            existing.confidence_prev = existing.confidence
            existing.confidence = edge.confidence
            existing.updated_at = edge.updated_at
        else:
            self.edges[edge.key] = edge

    # -- Queries ---------------------------------------------------------------

    def active_nodes(self, min_confidence: float = 0.3) -> List[BeliefNode]:
        return [n for n in self.nodes.values() if n.confidence >= min_confidence]

    def summary(self) -> str:
        if not self.nodes:
            return ""
        by_category: Dict[str, List[BeliefNode]] = {}
        for node in self.active_nodes(0.4):
            by_category.setdefault(node.category, []).append(node)
        parts = []
        for category, nodes in by_category.items():
            top = sorted(nodes, key=lambda n: n.confidence, reverse=True)[:2]
            labels = ", ".join(f"{n.label}({n.confidence:.0%})" for n in top)
            parts.append(f"{category}: {labels}")
        conflicts = [
            e for e in self.edges.values()
            if e.relation in ("contradicts", "conflicts_with") and e.confidence > 0.6
        ]
        if conflicts:
            parts.append(f"conflicts: {len(conflicts)}")
        return " | ".join(parts) if parts else ""
