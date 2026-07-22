"""
TBG Extractor v5.1.0

Architecture change vs v4.9.0:
- LLM prompt extracts ONLY raw facts (label, category, confidence, source)
  and optional causal edges. Zero decision logic in the prompt.
- SemanticDecisionLayer (Python class) owns all epistemic decisions:
    • deduplication (merge / flag / new)
    • reversal detection (explicit language → strong contradict + uncertain node)
    • opposition inference (polarity analysis — fast, no model needed)
    • oscillation pair linking (auto conflicts_with edges)
    • performative action sweeps (quit job → contradict career domain)
    • NLI-based semantic contradiction detection (optional, TBG_NLI_ENABLED=1)
      catches oppositions that polarity dict misses ("wants to be artist" vs
      "values job stability" — no polarity words, but semantically opposed)
- Result: deterministic, testable, debuggable belief update pipeline.
  Prompt bugs are now prompt bugs; logic bugs are now Python bugs with stack traces.
"""
import asyncio
import json
import os
import re
import logging
from typing import Optional, Dict, List, FrozenSet

from tbg_schema import TBGDelta, BeliefNode, BeliefEdge, UserTBG
from tbg_telemetry import emit as _tele_emit

logger = logging.getLogger(__name__)

# ── provenance instrumentation (logging-only) ───────────────────────────────
# Tags WHICH resolve() branch produced each contradict/reinforce/edge so an
# offline analysis can attribute belief decay to the LLM extractor vs SDL logic.
# STRICTLY no-op unless _PROV_ON is True (set by the attribution runner only).
# With _PROV_ON False, every _prov(...) call returns immediately and changes
# nothing -> golden E0 snapshot stays byte-identical (proof of no-op).
# Env-gated (default OFF): TBG_PROV_ON=1 turns on raw-edge/branch provenance so
# the demo ingest can dump _PROV for offline raw-vs-resolved analysis.
_PROV_ON = os.getenv("TBG_PROV_ON", "0") == "1"
# WARNING: when ON this module-level list accumulates across ALL users/ingests in
# the process (no per-user reset) — for single-process offline analysis only, not
# concurrent/multi-user serving. (New telemetry uses no module state; see tbg_telemetry.)
_PROV: List[dict] = []


def _prov(tag: str, kind: str, key) -> None:
    if _PROV_ON:
        _PROV.append({"tag": tag, "kind": kind, "key": key})

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

VALID_CATEGORIES: FrozenSet[str] = frozenset({
    "career", "mood", "goals", "values", "fears",
    "relationships", "finances", "identity",
})

VALID_RELATIONS: FrozenSet[str] = frozenset({
    "causes", "blocks", "motivates", "contradicts",
    "supports", "conflicts_with",
})

CONFIDENCE_CAP_BY_SOURCE: Dict[str, float] = {
    "explicit": 0.85,
    "inferred": 0.65,
}

CONFIDENCE_CAP_BY_CATEGORY: Dict[str, float] = {
    "mood": 0.50,
}

_CANONICAL_LABEL_PROMPT = """You normalize psychological/behavioral concept labels to canonical short forms for a knowledge graph.

Rules:
- Output 2 to 4 words, lowercase, no articles ("the", "a"), no first-person prefixes ("I feel", "I am", "I'm"), no auxiliary verbs ("feels", "is", "am").
- Preserve the conceptual meaning AND polarity (positive vs negative valence).
- Use compact psychology/HR vocabulary when applicable: "burnout", "career drift", "imposter syndrome", "identity crisis".
- If input is already canonical, return it unchanged.
- Same input must always produce same output.

Examples:
  "I feel exhausted" → "exhaustion"
  "values personal freedom" → "values freedom"
  "I'm a disciplined person" → "disciplined person"
  "my job is meaningless" → "meaningless work"

Input pairs:
{items_json}

Return strict JSON: {{"items": [{{"input": "<original>", "canonical": "<normalized>"}}, ...]}}
Preserve order and length. No commentary."""


def _generate_concept_id(category: str, label: str) -> str:
    """Generate a stable concept_id from category + label slug.
    The same label always produces the same concept_id.
    After canonicalization (Qwen, future), concept_ids will be reassigned
    to a canonical form — this is the MVP deterministic fallback.
    """
    slug = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')[:24]
    return f"{category}:{slug}"


def _lookup_or_register(label: str, category: str, tbg: UserTBG) -> str:
    """
    Cross-turn concept unification via tbg.concept_registry.

    The active graph loses context when nodes are pruned. The registry keeps
    every concept_id ever seen, mapped to its canonical label — so a paraphrase
    arriving several turns later can still be matched to the original id.

    Contract:
      - Returns a concept_id. The id is guaranteed to exist in
        tbg.concept_registry after this call.
      - Exact label repeats hit the concept_aliases cache (closed-vocab) for
        an O(1) match.
      - Otherwise generates a fresh concept_id (category:slug) and registers
        it with `label` as the canonical form.
    """
    # Step 4 (closed-vocab Stage 2): deterministic alias cache lookup.
    # If the LLM returned the exact same label as before, instant O(1) match — no
    # threshold, no path-dependence. Per LINK-KG (2025) Prompt Cache pattern.
    if _CLOSED_VOCAB:
        cached_id = tbg.concept_aliases.get(label.lower().strip())
        if cached_id:
            logger.info(f"concept_aliases hit: '{label}' -> id={cached_id}")
            return cached_id

    registry = tbg.concept_registry

    # Mint a fresh concept_id and register it. Preserve the first
    # canonical label seen under a given id (no overwrite on slug collision).
    new_id = _generate_concept_id(category, label)
    if new_id not in registry:
        registry[new_id] = label
    # Step 4: populate alias cache for next turns (idempotent).
    if _CLOSED_VOCAB:
        tbg.concept_aliases[label.lower().strip()] = new_id
    return new_id


async def _canonicalize_labels(
    raw_facts: List[dict],
    tbg: UserTBG,
    llm_call_fn,
) -> None:
    """
    In-place canonicalization of raw_facts['label'] before SDL.resolve.

    Phase 4: collapses LLM lexical variation ("values freedom" / "values personal
    freedom") to a stable canonical form, so _generate_concept_id produces
    matching slugs across runs.

    Strategy:
      - Pass 1: apply tbg.label_aliases cache, collect unmapped pairs
      - Pass 2: one batched LLM call for unmapped pairs
      - Pass 3: write cache, mutate raw_facts in place

    Safety: ANY error → no mutation, raw_facts unchanged. The pipeline
    falls through to the existing slug-based concept_id path.
    """
    if not raw_facts:
        return

    aliases = tbg.label_aliases  # mutable reference

    # Pass 1: cache hits + collect todo
    todo: List[tuple] = []  # (raw_label, category)
    seen: set = set()
    for raw in raw_facts:
        label = str(raw.get("label", "")).strip()
        category = str(raw.get("category", "")).strip()
        if not label:
            continue
        key = label.lower()
        if key in aliases:
            cached = aliases[key]
            if cached and len(cached) >= 2:
                raw["label"] = cached
            continue
        if key in seen:
            continue
        seen.add(key)
        todo.append((label, category))

    if not todo:
        return

    # Pass 2: batched LLM call
    items_json = json.dumps(
        [{"input": l, "category": c} for (l, c) in todo],
        ensure_ascii=False,
    )
    prompt = _CANONICAL_LABEL_PROMPT.format(items_json=items_json)

    try:
        raw_response = await asyncio.wait_for(llm_call_fn(prompt), timeout=15.0)
        parsed = json.loads(_clean_json(raw_response))
        items = parsed.get("items", [])
        if not isinstance(items, list) or len(items) != len(todo):
            logger.warning(
                f"canonicalize: length mismatch "
                f"(got {len(items) if isinstance(items, list) else '?'}, "
                f"expected {len(todo)}); skipping"
            )
            return
    except asyncio.TimeoutError:
        logger.warning("canonicalize: LLM timeout, skipping")
        return
    except Exception as e:
        logger.warning(f"canonicalize: parse error ({type(e).__name__}: {e}); skipping")
        return

    # Pass 3: validate, write cache, build apply-map
    label_to_canonical: Dict[str, str] = {}
    for (orig_label, _), response_item in zip(todo, items):
        if not isinstance(response_item, dict):
            continue
        canonical = str(response_item.get("canonical", "")).strip().lower()
        if not canonical or len(canonical) < 2 or len(canonical) > 60:
            continue
        key = orig_label.lower()
        label_to_canonical[key] = canonical
        aliases[key] = canonical

    # Apply to all raw_facts (covers within-turn duplicates)
    for raw in raw_facts:
        key = str(raw.get("label", "")).strip().lower()
        if key in label_to_canonical:
            raw["label"] = label_to_canonical[key]


# ---------------------------------------------------------------------------
# SEMANTIC DECISION LAYER
# ---------------------------------------------------------------------------

_IDENTITY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bI\s+am\s+a?\s*([^.,!?\n]{3,40})", re.I),
    re.compile(r"\bI'm\s+a?\s*([^.,!?\n]{3,40})", re.I),
    re.compile(r"\bI\s+see\s+myself\s+as\s+([^.,!?\n]{3,40})", re.I),
    re.compile(r"\bI'?ve\s+always\s+been\s+([^.,!?\n]{3,40})", re.I),
    re.compile(r"\bI\s+consider\s+myself\s+([^.,!?\n]{3,40})", re.I),
]


def _ensure_identity_fallback(
    user_text: str,
    delta: TBGDelta,
    tbg: "UserTBG",
) -> TBGDelta:
    # Condition 1: LLM already created identity in this delta
    if any(n.category == "identity" for n in delta.add_nodes):
        return delta

    # Condition 2: detect explicit self-reference pattern
    candidates = []
    for pattern in _IDENTITY_PATTERNS:
        m = pattern.search(user_text)
        if m:
            phrase = re.sub(r'\s+', ' ', m.group(1).strip())[:40]
            if len(phrase) >= 3:
                candidates.append(phrase)
    if not candidates:
        return delta

    # Condition 3: no existing identity node already covers this concept
    existing_identity_labels = {
        n.label.lower() for n in tbg.nodes.values()
        if n.category == "identity" and n.confidence >= 0.25
    }

    new_nodes = []
    for phrase in candidates[:2]:
        if phrase.lower() in existing_identity_labels:
            continue  # already covered
        node = BeliefNode(
            label=phrase,
            category="identity",
            confidence=0.20,
            source="inferred",
            concept_id=_generate_concept_id("identity", phrase),
            node_type="state",
            # Identity statements ("I am X") are inherently the self life-area.
            domain="self",
            stance="neutral",
        )
        new_nodes.append(node)
        logger.debug(f"identity_fallback: '{phrase}' conf=0.20")

    delta = delta.model_copy(update={"add_nodes": delta.add_nodes + new_nodes})
    return delta


# Performative actions: irreversible events that change world-state
_PERFORMATIVE: List[tuple] = [
    (re.compile(r'\b(quit|quitted|resigned|was fired|got fired|left.*?job|walked out)\b', re.I),
     frozenset({'career', 'identity', 'values', 'finances', 'goals'})),
    (re.compile(r'\b(moved|relocated)\b', re.I),
     frozenset({'relationships', 'identity', 'values'})),
    (re.compile(r'\b(broke up|divorced|separated)\b', re.I),
     frozenset({'relationships', 'mood', 'identity'})),
    (re.compile(r'\b(married|got engaged|engaged)\b', re.I),
     frozenset({'relationships', 'identity', 'values'})),
    (re.compile(r'\b(bought|purchased|took.*?loan|went into debt)\b', re.I),
     frozenset({'finances', 'values'})),
    (re.compile(r'\b(enrolled|started school|graduated|got accepted)\b', re.I),
     frozenset({'career', 'identity', 'goals'})),
]


class SemanticDecisionLayer:
    """
    Deterministic epistemic decision engine.

    Takes raw facts extracted by the LLM + the current UserTBG and returns
    a TBGDelta — no LLM calls, fully testable.

    Decision tree per fact:
        → create new node (concept_id resolved via concept_registry / slug).
          Exact-label dedup against the live graph is handled downstream by
          the engine's _update_node.

    After all facts: performative sweep links irreversible world-state events
    to contradictions; LLM-provided edges are resolved against the graph.
    """

    def resolve(
        self,
        raw_facts: List[dict],
        user_text: str,
        tbg: UserTBG,
        raw_edges: Optional[List[dict]] = None,
    ) -> TBGDelta:

        perf_domains  = self._detect_performative(user_text)

        add_nodes:       List[BeliefNode] = []
        reinforce_ids:   List[str]        = []
        contradict_ids:  List[str]        = []
        label_to_new_id: Dict[str, str]   = {}

        existing = tbg.nodes

        # op/ref closed vocabulary: label.lower() -> concept_id of an existing
        # concept, restricted to exactly the candidates shown in the prompt.
        # Empty (and never consulted) when the flag is OFF -> byte-identical.
        opref_index: Dict[str, str] = {}
        if _opref_enabled():
            for n in _opref_candidate_nodes(tbg):
                opref_index[n.label.lower().strip()] = (
                    n.concept_id or _generate_concept_id(n.category, n.label)
                )

        et_on = _evidence_type_enabled()

        if _PROV_ON:
            import copy as _copy
            _PROV.append({"tag": "RAW", "kind": "input",
                          "raw_facts": _copy.deepcopy(raw_facts),
                          "raw_edges": _copy.deepcopy(raw_edges or [])})

        for raw in raw_facts:
            category = raw.get("category")
            if category not in VALID_CATEGORIES:
                continue
            label = str(raw.get("label", ""))[:50].strip()
            if not label:
                continue

            source     = raw.get("source", "inferred")
            raw_conf   = float(raw.get("confidence", 0.5))
            confidence = min(
                raw_conf,
                CONFIDENCE_CAP_BY_SOURCE.get(source, 0.65),
                CONFIDENCE_CAP_BY_CATEGORY.get(category, 1.0),
            )
            # Ontological type — validated against known types, empty for unknown
            _raw_type = str(raw.get("type", "")).strip().lower()
            node_type = _raw_type if _raw_type in ("fact", "state", "value", "intention") else ""

            # Wheel-of-life routing fields. Validated; empty if unknown.
            _raw_domain = str(raw.get("domain", "")).strip().lower()
            node_domain = _raw_domain if _raw_domain in VALID_DOMAINS else ""
            _raw_stance = str(raw.get("stance", "")).strip().lower()
            node_stance = _raw_stance if _raw_stance in VALID_STANCES else ""

            # Evidence strength of THIS assertion (closed set). Drives the engine's
            # weighted update on repeat confirmations. Invalid/missing -> medium_pos
            # (treat as an ordinary positive assertion). None when the flag is OFF.
            evidence_type = None
            if et_on:
                _raw_et = str(raw.get("evidence_type", "")).strip().lower()
                evidence_type = _raw_et if _raw_et in VALID_EVIDENCE_TYPES else "medium_pos"

            # op/ref: if the LLM tagged this fact with a verbatim `ref` to an
            # existing concept (closed vocab), reuse that concept's concept_id so
            # the engine accumulates the anchor across paraphrases. Exact /
            # normalized (lower+strip) match only — a hallucinated ref outside the
            # candidate list is ignored and the normal mint path runs. No fuzzy.
            concept_id = None
            ref = raw.get("ref") if opref_index else None
            if ref:
                matched_cid = opref_index.get(str(ref).lower().strip())
                if matched_cid:
                    concept_id = matched_cid

            # Concept id via registry (cross-turn) / deterministic slug. Exact-label
            # dedup against the live graph is handled downstream by the engine's
            # _update_node (it reinforces an existing same-label/concept node).
            if concept_id is None:
                concept_id = _lookup_or_register(label, category, tbg)
            node = self._make_node(label, category, confidence, source,
                                   concept_id=concept_id, node_type=node_type,
                                   domain=node_domain, stance=node_stance,
                                   evidence_type=evidence_type)
            add_nodes.append(node)
            label_to_new_id[label.lower()] = node.id

        # Performative sweep: contradict old-state nodes in affected domains
        if perf_domains:
            new_ids = set(label_to_new_id.values())
            for node in existing.values():
                if (node.category in perf_domains
                        and node.confidence >= 0.45
                        and node.id not in contradict_ids
                        and node.id not in new_ids
                        and node.id not in reinforce_ids):
                    contradict_ids.append(node.id)
                    _prov("SDL_PERFORMATIVE", "contradict", node.id)

        # A3 no-double-count: the performative sweep contradicts existing nodes
        # (engine applies a strong_contradict). If a new fact this turn targets one
        # of those SAME concepts with a NEGATIVE evidence_type, that is one doubt
        # arriving on two channels. Keep the contradict channel; drop the node's
        # negative evidence_type so the engine does not punish the concept twice.
        # (A positive evidence_type on a contradicted concept is left intact —
        # different signal, the engine reconciles it via ambivalence.)
        if et_on and contradict_ids:
            contra_concepts = {
                existing[nid].concept_id for nid in contradict_ids if nid in existing
            }
            for node in add_nodes:
                if (node.evidence_type in ("medium_neg", "strong_neg")
                        and node.concept_id in contra_concepts):
                    node.evidence_type = None

        add_edges = self._resolve_edges(raw_edges or [], tbg, label_to_new_id)

        return TBGDelta(
            add_nodes=add_nodes,
            add_edges=add_edges,
            reinforce_ids=list(dict.fromkeys(reinforce_ids)),
            contradict_ids=list(dict.fromkeys(contradict_ids)),
        )

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _detect_performative(text: str) -> Optional[frozenset]:
        for pattern, domains in _PERFORMATIVE:
            if pattern.search(text):
                return domains
        return None

    @staticmethod
    def _make_node(
        label: str,
        category: str,
        confidence: float,
        source: str,
        concept_id: Optional[str] = None,
        node_type: str = "",
        domain: str = "",
        stance: str = "",
        evidence_type: Optional[str] = None,
    ) -> BeliefNode:
        return BeliefNode(
            label=label,
            category=category,
            confidence=confidence,
            source=source,
            concept_id=concept_id or _generate_concept_id(category, label),
            node_type=node_type,
            domain=domain,
            stance=stance,
            evidence_type=evidence_type,
        )

    @staticmethod
    def _resolve_edges(
        raw_edges: List[dict],
        tbg: UserTBG,
        label_to_new_id: Dict[str, str],
    ) -> List[BeliefEdge]:
        if _fix_label_collision():
            existing_by_label = {}
            for _nid, _n in tbg.nodes.items():
                _k = _n.label.lower()
                _prev = existing_by_label.get(_k)
                if _prev is None:
                    existing_by_label[_k] = _nid
                    continue
                _pn = tbg.nodes[_prev]
                logger.warning(f"TBG label collision {_k!r}: resolving by higher confidence")
                if (_n.confidence > _pn.confidence
                        or (_n.confidence == _pn.confidence and _n.created_at < _pn.created_at)):
                    existing_by_label[_k] = _nid
        else:
            existing_by_label = {n.label.lower(): nid for nid, n in tbg.nodes.items()}
        all_ids           = set(tbg.nodes.keys()) | set(label_to_new_id.values())
        edges             = []

        for e in raw_edges:
            if e.get("relation") not in VALID_RELATIONS:
                continue
            src_key = str(e.get("source", "")).lower().strip()
            tgt_key = str(e.get("target", "")).lower().strip()
            src = label_to_new_id.get(src_key) or existing_by_label.get(src_key)
            tgt = label_to_new_id.get(tgt_key) or existing_by_label.get(tgt_key)
            if not src or not tgt or src not in all_ids or tgt not in all_ids or src == tgt:
                continue
            edges.append(BeliefEdge(
                source_id=src,
                target_id=tgt,
                relation=e["relation"],
                confidence=float(e.get("confidence", 0.6)),
            ))
            _prov("LLM_EDGE", "edge", (src, tgt, e["relation"]))
        return edges


# Module-level singleton — shared across coroutines.
_sdl = SemanticDecisionLayer()



# Phase 4: enable via TBG_LABEL_CANONICALIZATION=1.
_LABEL_CANONICALIZATION_ENABLED = os.getenv("TBG_LABEL_CANONICALIZATION", "0") == "1"

# Step 2: disable the extraction_confidence penalty — the LLM self-rating is
# not calibrated against ground truth and adds variance to BDA.
_DISABLE_EXTRACTION_CONF_PENALTY = os.getenv("TBG_DISABLE_EXTRACTION_CONF_PENALTY", "0") == "1"

# Step 4: closed-vocabulary extraction + deterministic alias cache.
# Stage 1: prompt passes concept_id next to the label, requires exact copying.
# Stage 2: deterministic lookup in tbg.concept_aliases before matching.
# Basis: GenIE (EPFL 2022), LLM StructCore (CL4Health 2026), LINK-KG (2025).
_CLOSED_VOCAB = os.getenv("TBG_CLOSED_VOCAB", "0") == "1"

# op/ref: closed-vocab concept reference. When enabled, the extraction prompt
# lists existing concepts and the LLM may tag a fact with a verbatim `ref` to
# one of them; the extractor then reuses that concept's concept_id so the engine
# accumulates the anchor across paraphrases. Read dynamically (not a module
# constant) so a single process can compare ON vs OFF. Default OFF → byte-identical.
def _opref_enabled() -> bool:
    return os.getenv("TBG_OPREF", "0") == "1"


def _fix_label_collision() -> bool:
    # v1.2 flag: ON -> edge-label resolution keeps the HIGHER-confidence node on a
    # duplicate lowercased label (tie -> older created_at) and warns; OFF -> the
    # original last-wins dict-comprehension. Read dynamically. Default OFF -> byte-identical.
    return os.getenv("TBG_FIX_LABEL_COLLISION", "0") == "1"


# Active concepts offered to the LLM as the closed op/ref vocabulary, and matched
# against `ref` in resolve(). Same set on both sides so matching is exactly what
# was shown. Up to 50 active nodes (confidence >= 0.3), strongest first.
def _opref_candidate_nodes(tbg: UserTBG) -> List[BeliefNode]:
    return sorted(tbg.active_nodes(0.3), key=lambda n: n.confidence, reverse=True)[:50]


def _build_opref_block(tbg: UserTBG) -> str:
    nodes = _opref_candidate_nodes(tbg)
    if not nodes:
        return "(none yet)"
    return "\n".join(
        f'- "{n.label}" [{n.concept_id or _generate_concept_id(n.category, n.label)}]'
        for n in nodes
    )


# evidence_type: the LLM classifies how strongly the user asserts each belief, so
# the engine's evidence-weighted update can move confidence / pos_evidence on
# repeat confirmations (engine reads node.evidence_type -> EVIDENCE_WEIGHTS). Read
# dynamically so one process can compare ON vs OFF. Default OFF -> field not in the
# prompt, not parsed, evidence_type stays None -> byte-identical.
def _evidence_type_enabled() -> bool:
    return os.getenv("TBG_EVIDENCE_TYPE", "0") == "1"


VALID_EVIDENCE_TYPES: FrozenSet[str] = frozenset({
    "strong_pos", "medium_pos", "medium_neg", "strong_neg",
})

EVIDENCE_TYPE_CLAUSE = """\
EVIDENCE STRENGTH (evidence_type) — for each fact, classify how the user asserts it:
- strong_pos = emphatic / decisive assertion or re-affirmation of the belief
- medium_pos = ordinary assertion of the belief (DEFAULT for a positive assertion)
- medium_neg = the user doubts or partially walks back this belief
- strong_neg = the user decisively rejects / contradicts this belief
Value strictly from this set; default medium_pos. Use strong_* only for explicit
emphasis or categorical rejection — keep it rare."""


def _inject_evidence_type_clause(prompt: str) -> str:
    """Add the evidence_type field + instruction to an already-built extraction
    prompt via string injection on stable anchors. Runs only when the flag is ON,
    so the OFF path leaves the base templates byte-identical."""
    prompt = prompt.replace(
        '      "source": "explicit|inferred",\n',
        '      "source": "explicit|inferred",\n'
        '      "evidence_type": "strong_pos|medium_pos|medium_neg|strong_neg",\n',
        1,
    )
    prompt = prompt.replace(
        "Return ONLY valid JSON. No markdown. No code fences.",
        EVIDENCE_TYPE_CLAUSE + "\n\nReturn ONLY valid JSON. No markdown. No code fences.",
        1,
    )
    return prompt


VALID_DOMAINS: FrozenSet[str] = frozenset({
    "career", "money", "health", "relationships",
    "family", "lifestyle", "meaning", "self", "other",
})
VALID_STANCES: FrozenSet[str] = frozenset({"approach", "avoid", "neutral"})


def validate_delta(
    delta: TBGDelta,
    extraction_confidence: float,
) -> tuple:
    """
    Deterministic post-extraction validation.
    Catches LLM hallucinations: invalid categories, uncalibrated confidence,
    low-quality extractions.  Returns (cleaned_delta, warnings_list).
    """
    warnings = []
    clean_nodes = []

    for node in delta.add_nodes:
        # Hard check: category must be from closed set
        if node.category not in VALID_CATEGORIES:
            warnings.append(f"INVALID_CATEGORY: '{node.category}' for '{node.label}' -> dropped")
            continue

        # Confidence cap by source + category
        cap = CONFIDENCE_CAP_BY_SOURCE.get(node.source, 0.65)
        cat_cap = CONFIDENCE_CAP_BY_CATEGORY.get(node.category, 1.0)
        effective_cap = min(cap, cat_cap)
        if node.confidence > effective_cap:
            warnings.append(f"CONFIDENCE_CAPPED: '{node.label}' {node.confidence:.2f} -> {effective_cap:.2f}")
            node = node.model_copy(update={"confidence": effective_cap})

        # If extraction_confidence is low, penalize inferred nodes.
        # Disabled via TBG_DISABLE_EXTRACTION_CONF_PENALTY=1 — penalty depends
        # on LLM self-reported value, which is uncalibrated and noisy.
        if not _DISABLE_EXTRACTION_CONF_PENALTY:
            if extraction_confidence < 0.6 and node.source == "inferred":
                penalized = round(node.confidence * extraction_confidence, 3)
                warnings.append(f"LOW_EXTRACTION_CONF: '{node.label}' {node.confidence:.2f} -> {penalized:.2f}")
                node = node.model_copy(update={"confidence": penalized})

        clean_nodes.append(node)

    # Edge validation: relation must be from closed set
    clean_edges = [e for e in delta.add_edges if e.relation in VALID_RELATIONS]
    dropped_edges = len(delta.add_edges) - len(clean_edges)
    if dropped_edges:
        warnings.append(f"INVALID_RELATIONS: {dropped_edges} edges dropped")

    clean_delta = delta.model_copy(update={"add_nodes": clean_nodes, "add_edges": clean_edges})
    return clean_delta, warnings



# ---------------------------------------------------------------------------
# SEMANTIC GRAPH QUERY — one LLM call, any question about the graph
# ---------------------------------------------------------------------------

_SEMANTIC_QUERY_PROMPT = """\
You are analyzing a psychological belief graph about a person.

BELIEFS IN GRAPH:
{beliefs}

QUESTION: {query}

Return ONLY a JSON array of matching belief labels (exact strings from the list above).
Return [] if nothing matches.
["label one", "label two"]
"""


async def query_graph_semantic(
    tbg: "UserTBG",
    query: str,
    llm_call_fn,
) -> List[str]:
    """Find graph nodes that semantically match a natural-language query.

    One LLM call. Use for evaluation, insight generation, or user-facing queries.
    Not called per turn — invoke explicitly when needed.
    """
    if not tbg or not tbg.nodes:
        return []

    beliefs = "\n".join(
        f'  "{n.label}" [{n.category}] {n.confidence:.0%}'
        for n in sorted(tbg.nodes.values(), key=lambda n: n.confidence, reverse=True)
    )
    prompt = _SEMANTIC_QUERY_PROMPT.format(beliefs=beliefs, query=query)

    try:
        raw = await llm_call_fn(prompt)
        raw = _clean_json(raw)
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(x) for x in result if x]
    except Exception as e:
        logger.debug(f"query_graph_semantic error: {e}")

    return []


# ---------------------------------------------------------------------------
# PROMPT — facts + causal edges only, zero decision logic
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
Extract psychological beliefs from this conversation.

EXISTING BELIEFS — if this conversation mentions the SAME concept, use the EXACT existing label:
{existing_labels}

STRICT RULES:
- category MUST be exactly one of: career, mood, goals, values, fears, relationships, finances, identity
- relation MUST be exactly one of: causes, blocks, motivates, supports, contradicts
- confidence for "inferred" source: CEILING 0.65 — use actual certainty, not the ceiling
- confidence for "explicit" source: CEILING 0.85 — use actual certainty, not the ceiling
- If unsure about category, use the closest match from the list above — do NOT invent new categories
- Do NOT invent new categories or relations under any circumstances

Return ONLY valid JSON. No markdown. No code fences.

{{
  "reasoning": "one sentence, or 'no update'",
  "extraction_confidence": 0.0-1.0,
  "facts": [
    {{"label": "2-4 words in English",
      "category": "career|mood|goals|values|fears|relationships|finances|identity",
      "confidence": calibrate to actual certainty (ceiling: explicit≤0.85, inferred≤0.65),
      "source": "explicit|inferred",
      "type": "fact|state|value|intention",
      "domain": "career|money|health|relationships|family|lifestyle|meaning|self|other",
      "stance": "approach|avoid|neutral"}}
  ],
  "edges": [
    {{"source": "label", "target": "label",
      "relation": "causes|blocks|motivates|supports|contradicts",
      "confidence": 0.0-1.0}}
  ]
}}

extraction_confidence: your overall confidence in this extraction.
  1.0 = message is clear and beliefs are unambiguous
  0.5 = message is vague, extraction is speculative
  0.0 = nothing meaningful to extract

Rules:
- All text must be in English
- Extract FACTS: durable beliefs, values, fears, goals, identity patterns
- LABEL REUSE: If the same concept already exists in the list above (even worded differently),
  use the EXACT existing label — do not invent a new one. Match by meaning, not by words.
  Example: existing "protects clients" covers "lawyer professional role" — reuse it.
- LABEL GRANULARITY: Use episodically CONCRETE labels, not premature abstractions.
    good: "wants to leave NYC", "fears losing team", "wife earns less than mortgage"
    bad:  "Relocation doubt", "Career uncertainty", "Life dissatisfaction"
  Concrete labels enable conflict detection. Abstraction is insight generation's job.
- TYPE field: classify each belief as one of:
    fact      = something that happened or objective situation ("I lost my job", "I moved to NYC")
    state     = current feeling or subjective experience ("I feel anxious", "I am exhausted")
    value     = principle or belief about what matters ("I value stability", "freedom is essential")
    intention = plan, desire, or goal ("I want to quit", "I plan to move")
- DOMAIN field: the life area this concept belongs to (wheel-of-life), exactly one of:
    career, money, health, relationships, family, lifestyle, meaning, self, other.
    This is the real-world life area — independent of `category`. Pick the single best fit.
- STANCE field: approach (moving toward), avoid (moving away from), or neutral.
- Do NOT decide reinforce / deduplicate — handled automatically
- Use "contradicts" edge when two beliefs are psychologically incompatible
- Skip small talk, weather, food, commute, one-off trivial events
- If nothing meaningful changed: reasoning="no update", facts=[], edges=[]

User: "{user_text}"
Assistant: "{assistant_text}"
"""

EXTRACTION_PROMPT_CLOSED_VOCAB = """\
Extract psychological beliefs from this conversation.

EXISTING BELIEFS — each line shows: "label" [category] confidence (id=concept_id).
{existing_labels}

CRITICAL RULES FOR LABEL REUSE:
- If the user message refers to ANY belief in the EXISTING BELIEFS list above (even worded differently), copy the EXACT label from that list character-by-character.
- Match by MEANING, not by surface words. "freedom matters" = "values freedom" if the latter exists.
- Use the SAME category as the existing belief — do NOT change category between turns.
- ONLY introduce a new label when the concept is genuinely absent from the EXISTING BELIEFS list.

STRICT RULES:
- category MUST be exactly one of: career, mood, goals, values, fears, relationships, finances, identity
- relation MUST be exactly one of: causes, blocks, motivates, supports, contradicts
- confidence for "inferred" source: CEILING 0.65 — use actual certainty, not the ceiling
- confidence for "explicit" source: CEILING 0.85 — use actual certainty, not the ceiling
- If unsure about category, use the closest match — do NOT invent new categories
- Do NOT invent new categories or relations under any circumstances

Return ONLY valid JSON. No markdown. No code fences.

{{
  "reasoning": "one sentence, or 'no update'",
  "extraction_confidence": 0.0-1.0,
  "facts": [
    {{"label": "EXACT copy from EXISTING BELIEFS or new 2-4 word label",
      "category": "career|mood|goals|values|fears|relationships|finances|identity",
      "confidence": calibrate to actual certainty (ceiling: explicit≤0.85, inferred≤0.65),
      "source": "explicit|inferred",
      "type": "fact|state|value|intention",
      "domain": "career|money|health|relationships|family|lifestyle|meaning|self|other",
      "stance": "approach|avoid|neutral"}}
  ],
  "edges": [
    {{"source": "label", "target": "label",
      "relation": "causes|blocks|motivates|supports|contradicts",
      "confidence": 0.0-1.0}}
  ]
}}

extraction_confidence: your overall confidence in this extraction.
  1.0 = message is clear and beliefs are unambiguous
  0.5 = message is vague, extraction is speculative
  0.0 = nothing meaningful to extract

Rules:
- All text must be in English
- Extract FACTS: durable beliefs, values, fears, goals, identity patterns
- LABEL GRANULARITY: Use episodically CONCRETE labels for new concepts.
    good: "wants to leave NYC", "fears losing team"
    bad:  "Relocation doubt", "Life dissatisfaction"
- TYPE field: classify each belief as one of:
    fact      = something objective ("I lost my job")
    state     = current feeling ("I feel anxious")
    value     = principle ("I value stability")
    intention = plan or goal ("I want to quit")
- DOMAIN field: life area (wheel-of-life), exactly one of:
    career, money, health, relationships, family, lifestyle, meaning, self, other.
    Real-world life area, independent of `category`. Pick the single best fit.
- STANCE field: approach (toward), avoid (away from), or neutral.
- Do NOT decide reinforce / deduplicate — handled automatically
- Skip small talk, weather, food, commute, trivial events
- If nothing meaningful changed: reasoning="no update", facts=[], edges=[]

User: "{user_text}"
Assistant: "{assistant_text}"
"""

# op/ref extraction prompt (TBG_OPREF=1). Same single call; offers a closed list
# of existing concepts and lets the model reference one verbatim via `ref`.
EXTRACTION_PROMPT_OPREF = """\
Extract psychological beliefs from this conversation.

EXISTING CONCEPTS — each line: - "label" [concept_id]
{candidates}

CONCEPT REFERENCE (op/ref):
- If a fact refers to a concept ALREADY in the EXISTING CONCEPTS list (even worded
  differently), add a field "ref" whose value is the EXACT label from that list,
  copied character-for-character.
- If the fact is a genuinely NEW concept, do NOT include "ref".
- "ref" is either copied verbatim from the list above, or absent. NEVER invent a
  ref that is not in the list. Still provide a "label" for the fact as usual.

STRICT RULES:
- category MUST be exactly one of: career, mood, goals, values, fears, relationships, finances, identity
- relation MUST be exactly one of: causes, blocks, motivates, supports, contradicts
- confidence for "inferred" source: CEILING 0.65 — use actual certainty, not the ceiling
- confidence for "explicit" source: CEILING 0.85 — use actual certainty, not the ceiling
- If unsure about category, use the closest match — do NOT invent new categories
- Do NOT invent new categories or relations under any circumstances

Return ONLY valid JSON. No markdown. No code fences.

{{
  "reasoning": "one sentence, or 'no update'",
  "extraction_confidence": 0.0-1.0,
  "facts": [
    {{"label": "2-4 words in English",
      "ref": "EXACT label from EXISTING CONCEPTS, or omit if new",
      "category": "career|mood|goals|values|fears|relationships|finances|identity",
      "confidence": calibrate to actual certainty (ceiling: explicit≤0.85, inferred≤0.65),
      "source": "explicit|inferred",
      "type": "fact|state|value|intention",
      "domain": "career|money|health|relationships|family|lifestyle|meaning|self|other",
      "stance": "approach|avoid|neutral"}}
  ],
  "edges": [
    {{"source": "label", "target": "label",
      "relation": "causes|blocks|motivates|supports|contradicts",
      "confidence": 0.0-1.0}}
  ]
}}

extraction_confidence: your overall confidence in this extraction.
  1.0 = message is clear and beliefs are unambiguous
  0.5 = message is vague, extraction is speculative
  0.0 = nothing meaningful to extract

Rules:
- All text must be in English
- Extract FACTS: durable beliefs, values, fears, goals, identity patterns
- LABEL GRANULARITY: Use episodically CONCRETE labels for new concepts.
    good: "wants to leave NYC", "fears losing team"
    bad:  "Relocation doubt", "Life dissatisfaction"
- TYPE field: classify each belief as one of:
    fact      = something objective ("I lost my job")
    state     = current feeling ("I feel anxious")
    value     = principle ("I value stability")
    intention = plan or goal ("I want to quit")
- DOMAIN field: life area (wheel-of-life), exactly one of:
    career, money, health, relationships, family, lifestyle, meaning, self, other.
    Real-world life area, independent of `category`. Pick the single best fit.
- STANCE field: approach (toward), avoid (away from), or neutral.
- Do NOT decide reinforce / deduplicate — handled automatically
- Use "contradicts" edge when two beliefs are psychologically incompatible
- Skip small talk, weather, food, commute, trivial events
- If nothing meaningful changed: reasoning="no update", facts=[], edges=[]

User: "{user_text}"
Assistant: "{assistant_text}"
"""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _build_existing_labels(tbg: UserTBG) -> str:
    if not tbg or not tbg.nodes:
        return "none"
    if _CLOSED_VOCAB:
        # Closed-vocab: include concept_id explicitly, instruct LLM to copy verbatim.
        lines = [
            f'  "{n.label}" [{n.category}] {n.confidence:.0%}  (id={n.concept_id})'
            for n in sorted(tbg.nodes.values(), key=lambda n: n.confidence, reverse=True)[:30]
        ]
    else:
        lines = [
            f'  "{n.label}" [{n.category}] {n.confidence:.0%}'
            for n in sorted(tbg.nodes.values(), key=lambda n: n.confidence, reverse=True)[:30]
        ]
    return "\n".join(lines)


def _smart_truncate(text: str, max_chars: int = 600) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + "\n...\n" + text[-tail:]


def _clean_json(raw_text: str) -> str:
    if not raw_text:
        return ""
    raw_text = raw_text.strip()
    raw_text = re.sub(r"```json|```", "", raw_text).strip()
    s, e = raw_text.find("{"), raw_text.rfind("}")
    if s != -1 and e != -1:
        raw_text = raw_text[s:e + 1]
    # Trailing commas before ] or } — common LLM mistake
    raw_text = re.sub(r",\s*([\]}])", r"\1", raw_text)
    return raw_text


# Benchmark/debug: when on, dump raw + cleaned model output on parse failure.
_EXTRACT_DEBUG = os.getenv("TBG_EXTRACT_DEBUG", "0") == "1"


def _repair_truncated_json(s: str) -> str:
    """Best-effort repair of truncated/unclosed LLM JSON, stdlib only.

    Covers the dominant failure mode the span-extract pass cannot: output cut
    off by max_tokens mid-string/mid-object. Tracks string/escape state so it
    never miscounts braces inside strings, closes any open string, drops a
    dangling trailing comma, and appends the missing closers in stack order.
    This is the 'Repair' tier of a Generate -> Validate -> Repair -> Parse
    pipeline; it is a fallback, not the primary path.
    """
    if not s:
        return s
    stack, in_str, escape = [], False, False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = in_str
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    repaired = s + ('"' if in_str else "")
    repaired = re.sub(r",\s*$", "", repaired)
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _ret(result, status, return_status, tele=None):
    """Backward-compatible return. Default: bare result (existing contract).
    Opt-in (benchmark/debug): (result, status) so callers can distinguish
    no_update / empty_facts / failed_json / failed_other from a real delta.

    Telemetry (Part B extension per Step 0): the single return chokepoint. When a
    `tele` accumulator is supplied it is emitted as one `extraction` event with the
    status this function already knows — covering ALL five return paths. `emit`
    is itself a no-op when TBG_TELEMETRY is off, so with tele=None (or OFF) the
    return value is byte-identical to before."""
    if tele is not None:
        _tele_emit({"event": "extraction", "status": status, **tele})
    return (result, status) if return_status else result


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

async def extract_tbg_delta(
    user_text: str,
    assistant_text: str,
    existing_tbg_summary: str,           # kept for API compat, unused
    existing_label_to_uuid: Dict[str, str],  # kept for API compat, unused
    llm_call_fn,
    tbg: Optional[UserTBG] = None,
    return_status: bool = False,
):
    """Returns Optional[TBGDelta] by default (None = no-op, unchanged contract).
    If return_status=True, returns (Optional[TBGDelta], status_str) where status
    is one of: ok / no_update / empty_facts / failed_json / failed_other."""

    if _opref_enabled() and tbg:
        # op/ref: offer the existing concepts as a closed vocabulary in the same
        # single extraction call. Falls through to the standard prompt when OFF.
        prompt = EXTRACTION_PROMPT_OPREF.format(
            candidates=_build_opref_block(tbg),
            user_text=_smart_truncate(user_text, 600),
            assistant_text=_smart_truncate(assistant_text, 400),
        )
    else:
        existing_labels = _build_existing_labels(tbg) if tbg else "none"
        prompt_template = (
            EXTRACTION_PROMPT_CLOSED_VOCAB if _CLOSED_VOCAB else EXTRACTION_PROMPT
        )
        prompt = prompt_template.format(
            existing_labels=existing_labels,
            user_text=_smart_truncate(user_text, 600),
            assistant_text=_smart_truncate(assistant_text, 400),
        )

    # evidence_type clause is injected on top of whichever base prompt was built,
    # only when the flag is ON (OFF -> prompt untouched -> byte-identical).
    if _evidence_type_enabled():
        prompt = _inject_evidence_type_clause(prompt)

    # Telemetry accumulator (Part B). Populated as locals become known; fields
    # unknown on an early failure are simply absent (not zeroed). Emitted once,
    # with the status, inside _ret — covering all five return paths.
    _tele = {
        "user_id": (tbg.user_id if tbg else "_extract_dummy"),
        "msg_count": (tbg.message_count if tbg else 0),
        "prompt_variant": ("opref" if (_opref_enabled() and tbg)
                           else ("closed_vocab" if _CLOSED_VOCAB else "default")),
        "evidence_type_on": _evidence_type_enabled(),
        "repair_used": False,
    }

    raw_text = None
    cleaned = None
    try:
        raw_text = await llm_call_fn(prompt)
        cleaned = _clean_json(raw_text)
        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            # Repair tier: handle truncation / unclosed structures, retry once.
            _tele["repair_used"] = True
            cleaned = _repair_truncated_json(cleaned)
            raw = json.loads(cleaned)

        if raw.get("reasoning") == "no update" and not raw.get("facts"):
            return _ret(None, "no_update", return_status, tele=_tele)

        raw_facts = raw.get("facts", []) or []
        raw_edges = raw.get("edges", []) or []
        extraction_confidence = min(1.0, max(0.0, float(raw.get("extraction_confidence", 0.7))))
        _tele["facts_raw"] = len(raw_facts)
        _tele["edges_raw"] = len(raw_edges)
        _tele["extraction_confidence"] = extraction_confidence

        if not raw_facts:
            # Distinct from no_update: the model produced JSON but no facts. May
            # signal a format slip (e.g. edges present, or signal misrouted), not
            # genuine silence. Surface it instead of swallowing it.
            if _EXTRACT_DEBUG:
                logger.warning(
                    f"TBG empty_facts (edges={len(raw_edges)}): "
                    f"{str(raw.get('reasoning',''))[:80]} | raw={str(raw_text)[:200]!r}"
                )
            return _ret(None, "empty_facts", return_status, tele=_tele)

        effective_tbg = tbg or UserTBG(user_id="_extract_dummy")

        # Phase 4: canonicalize labels before SDL (feature-flagged, safe no-op)
        if _LABEL_CANONICALIZATION_ENABLED:
            await _canonicalize_labels(raw_facts, effective_tbg, llm_call_fn)

        loop  = asyncio.get_running_loop()
        delta = await loop.run_in_executor(
            None,
            lambda: _sdl.resolve(raw_facts, user_text, effective_tbg, raw_edges),
        )

        delta = _ensure_identity_fallback(user_text, delta, effective_tbg)

        # Post-extraction validation: catch hallucinated categories,
        # cap uncalibrated confidence, penalize low-quality extractions.
        delta, warnings = validate_delta(delta, extraction_confidence)
        _tele["warnings"] = warnings
        _tele["warnings_count"] = len(warnings)
        if warnings:
            logger.warning(
                f"TBG validator [{(tbg.user_id if tbg else '?')[:8]}]: {warnings}"
            )

        logger.info(
            f"TBG delta: +{len(delta.add_nodes)} nodes "
            f"+{len(delta.add_edges)} edges "
            f"reinforce={len(delta.reinforce_ids)} "
            f"contradict={len(delta.contradict_ids)} "
            f"strong_contradict={len(delta.strong_contradict_ids)} "
            f"extraction_conf={extraction_confidence:.2f} "
            f"warnings={len(warnings)} "
            f"| {raw.get('reasoning', '')[:80]}"
        )
        return _ret(delta, "ok", return_status, tele=_tele)

    except json.JSONDecodeError as e:
        logger.warning(f"TBG: invalid JSON after repair attempt: {e}")
        if _EXTRACT_DEBUG:
            logger.warning(f"TBG raw output : {str(raw_text)[:400]!r}")
            logger.warning(f"TBG cleaned    : {str(cleaned)[:400]!r}")
        return _ret(None, "failed_json", return_status, tele=_tele)
    except Exception as e:
        logger.error(f"TBG extraction error: {e}", exc_info=True)
        if _EXTRACT_DEBUG and raw_text is not None:
            logger.error(f"TBG raw output : {str(raw_text)[:400]!r}")
        return _ret(None, "failed_other", return_status, tele=_tele)

