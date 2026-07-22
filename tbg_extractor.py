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
import numpy as np
from typing import Optional, Dict, List, FrozenSet

from tbg_schema import TBGDelta, BeliefNode, BeliefEdge, UserTBG
import her_resolver

logger = logging.getLogger(__name__)

# ── provenance instrumentation (logging-only) ───────────────────────────────
# Tags WHICH resolve() branch produced each contradict/reinforce/edge so an
# offline analysis can attribute belief decay to the LLM extractor vs SDL logic.
# STRICTLY no-op unless _PROV_ON is True (set by the attribution runner only).
# With _PROV_ON False, every _prov(...) call returns immediately and changes
# nothing -> golden E0 snapshot stays byte-identical (proof of no-op).
_PROV_ON = False
_PROV: List[dict] = []
_LAST_OPP_REASON = ""   # set by _is_opposition each call: 'polarity'|'epa'|'none'


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

DEDUP_MERGE_THRESHOLD = 0.82   # ≥ this → same concept, reinforce
DEDUP_FLAG_THRESHOLD  = 0.72   # ≥ this and < MERGE → related, oscillation pair
# Within same category the merge threshold drops to FLAG level.
# "corporate lawyer identity" vs "professional identity" (sim ~0.74, same category)
# should merge, not create a false conflicts_with edge.
DEDUP_SAME_CATEGORY_MERGE = DEDUP_FLAG_THRESHOLD

# Cross-turn concept unification. Used when a new label finds NO match in the
# current graph but a paraphrase (pruned in an earlier turn) survives in
# tbg.concept_registry. Stricter than MERGE because it operates on canonical
# labels without category-level guardrails.
#
# NOTE (empirical, 2026-04): with the current embedding model, real paraphrase
# pairs like ('M&A attorney', 'corporate lawyer') score ~0.70, well below 0.87.
# At this threshold, registry lookup almost always misses and mints new ids.
# The feature is present but practically dormant until one of:
#   - a stronger embedding model is plugged into fact_engine, OR
#   - the threshold is recalibrated on labeled paraphrase pairs (~0.72 was
#     the empirical lower bound for same-concept pairs, but carries merge-risk
#     for concepts like 'data engineer'/'software engineer' at ~0.74).
# Deliberately not lowered here — calibration is a separate task requiring
# a labeled dataset, not a guess.
CONCEPT_REGISTRY_THRESHOLD = 0.87

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


# ---------------------------------------------------------------------------
# EMBEDDING
# ---------------------------------------------------------------------------

from fact_engine import get_embed_model as _get_embed_model


def _embed_labels(labels: List[str]) -> np.ndarray:
    if not labels:
        return np.array([])
    return _get_embed_model().encode(labels, normalize_embeddings=True)


_SIM_NONE  = "none"
_SIM_FLAG  = "flag"
_SIM_MERGE = "merge"


def _check_semantic_similarity(new_label: str, new_category: str, existing_nodes: Dict) -> tuple:
    """
    Compare new_label against existing nodes. Returns (match_type, node_id, similarity).

    Two-pass strategy:
    1. Within same category: merge threshold = DEDUP_SAME_CATEGORY_MERGE (0.72).
       Prevents false conflicts_with edges between paraphrases of the same concept.
    2. Global: standard MERGE (0.82) / FLAG (0.72) thresholds.
    """
    if not existing_nodes:
        return (_SIM_NONE, None, None)

    existing_ids    = list(existing_nodes.keys())
    existing_labels = [existing_nodes[nid].label for nid in existing_ids]

    try:
        all_labels    = existing_labels + [new_label]
        embeddings    = _embed_labels(all_labels)
        new_emb       = embeddings[-1]
        existing_embs = embeddings[:-1]

        similarities = existing_embs @ new_emb

        # Pass 1 — within same category at lower threshold
        if new_category:
            same_cat = [
                (i, float(similarities[i]))
                for i, nid in enumerate(existing_ids)
                if existing_nodes[nid].category == new_category
            ]
            if same_cat:
                best_i, best_sim_cat = max(same_cat, key=lambda x: x[1])
                if best_sim_cat >= DEDUP_SAME_CATEGORY_MERGE:
                    return (_SIM_MERGE, existing_ids[best_i], best_sim_cat)

        # Pass 2 — global thresholds
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_id  = existing_ids[best_idx]

        if best_sim >= DEDUP_MERGE_THRESHOLD:
            return (_SIM_MERGE, best_id, best_sim)
        if best_sim >= DEDUP_FLAG_THRESHOLD:
            return (_SIM_FLAG, best_id, best_sim)

    except Exception as e:
        logger.debug(f"Semantic similarity error: {e}")

    return (_SIM_NONE, None, None)


def _lookup_or_register(label: str, category: str, tbg: UserTBG) -> str:
    """
    Cross-turn concept unification via tbg.concept_registry.

    The active graph loses context when nodes are pruned. The registry keeps
    every concept_id ever seen, mapped to its canonical label — so a paraphrase
    arriving several turns later can still be matched to the original id.

    Contract:
      - Returns a concept_id. The id is guaranteed to exist in
        tbg.concept_registry after this call.
      - If the new label is cosine-similar to any canonical label at
        ≥ CONCEPT_REGISTRY_THRESHOLD, returns the existing concept_id.
      - Otherwise generates a fresh concept_id (category:slug) and registers
        it with `label` as the canonical form.

    Called only from the _SIM_NONE branch of resolve() — MERGE/FLAG paths
    already resolve concept_id via the live graph.
    """
    # Step 4 (closed-vocab Stage 2): deterministic alias cache lookup BEFORE cosine.
    # If LLM returned exact same label as before, instant O(1) match — no embedding,
    # no threshold, no path-dependence. Per LINK-KG (2025) Prompt Cache pattern.
    if _CLOSED_VOCAB:
        cached_id = tbg.concept_aliases.get(label.lower().strip())
        if cached_id:
            logger.info(f"concept_aliases hit: '{label}' -> id={cached_id}")
            return cached_id

    registry = tbg.concept_registry

    if registry:
        try:
            concept_ids      = list(registry.keys())
            canonical_labels = [registry[cid] for cid in concept_ids]
            embeddings       = _embed_labels(canonical_labels + [label])
            new_emb          = embeddings[-1]
            existing_embs    = embeddings[:-1]

            similarities = existing_embs @ new_emb
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= CONCEPT_REGISTRY_THRESHOLD:
                matched_id = concept_ids[best_idx]
                logger.info(
                    f"concept_registry hit: '{label}' -> '{canonical_labels[best_idx]}' "
                    f"(id={matched_id}, sim={best_sim:.3f})"
                )
                # Step 4: cache this alias — next exact match becomes O(1).
                if _CLOSED_VOCAB:
                    tbg.concept_aliases[label.lower().strip()] = matched_id
                return matched_id
        except Exception as e:
            logger.debug(f"concept_registry lookup error: {e}")

    # No match — mint a fresh concept_id and register it. Preserve the first
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

# Words that carry negative psychological valence (doubt, absence, fear…)
_NEGATIVE_POLARITY: FrozenSet[str] = frozenset({
    "lack", "poor", "absence", "fear", "doubt", "uncertainty",
    "insecurity", "inability", "failure", "regret", "anxiety", "anxious",
    "dread", "trapped", "exhaustion", "burnout", "meaningless", "meaninglessness",
    "dissatisfaction", "dissatisfied", "unhappy", "unhappiness",
    "opposition", "pressure", "procrastination", "avoidance", "distrust",
    "instability", "dependence", "conflict", "worry", "worried",
    "worthless", "hopeless", "helpless", "powerless", "stuck", "stagnant",
    "hostility", "resentment", "frustration", "depression", "loneliness",
})

# Words that carry positive psychological valence (resolve, pride, freedom…)
_POSITIVE_POLARITY: FrozenSet[str] = frozenset({
    "resolve", "commitment", "confidence", "pride", "satisfaction",
    "happiness", "trust", "hope", "growth", "success",
    "appreciation", "stability", "security", "clarity", "freedom",
    "autonomy", "fulfillment", "discipline", "reliability", "dedication",
    "ambition", "motivation", "aspiration", "resilience", "courage",
    "optimism", "purpose", "meaning", "strength", "independence",
    "mastery", "competence", "creativity", "curiosity", "passion",
})

# Explicit reversal language in user text
_REVERSAL_PATTERNS: List[re.Pattern] = [
    re.compile(r'^No[,\.]?\s', re.I | re.MULTILINE),
    re.compile(r'\bactually\b', re.I),
    re.compile(r'\bI changed my mind\b', re.I),
    re.compile(r'\bI was wrong\b', re.I),
    re.compile(r'\bnot anymore\b', re.I),
    re.compile(r'\bI decided\b', re.I),
    re.compile(r"\bI'?ve decided\b", re.I),
    re.compile(r'\bI realize[d]?\b', re.I),
    re.compile(r'\bI need to try\b', re.I),
    re.compile(r'\bon second thoughts?\b', re.I),
    re.compile(r'\bhaving second thoughts\b', re.I),
    re.compile(r'\bI don\'t want to live with\b', re.I),
    re.compile(r'\bscrew (it|that|this)\b', re.I),
    re.compile(r'\bforgot? it\b', re.I),
]

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
    existing_identity_labels = [
        n.label for n in tbg.nodes.values()
        if n.category == "identity" and n.confidence >= 0.25
    ]
    if existing_identity_labels:
        try:
            existing_embs = _embed_labels(existing_identity_labels)
        except Exception:
            existing_embs = None
    else:
        existing_embs = None

    new_nodes = []
    for phrase in candidates[:2]:
        if existing_embs is not None:
            try:
                new_emb = _embed_labels([phrase])
                sims = existing_embs @ new_emb[0]
                if float(sims.max()) > 0.70:
                    continue  # already covered
            except Exception:
                pass
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

# Threshold above which a contradicted node gets "strong_contradict" treatment
_STRONG_CONTRADICT_THRESHOLD = 0.75

# EPA axis-based opposition detection.
# Activated only when polarity dict returns 0 for both labels (no polarity words).
# Checks 3 most discriminative axes from S2.5 trajectory analysis.
# Threshold calibrated to avoid false positives: |distance| > 0.40 on any axis.
_EPA_AXES          = ("evaluation", "potency", "autonomy")
_EPA_OPP_THRESHOLD = 0.42

# ---------------------------------------------------------------------------
# TYPE GATING — ontological conflict rules
# ---------------------------------------------------------------------------

# Pairs where a conflicts_with edge is ontologically invalid.
# FACT/STATE cannot conflict with VALUE — they operate on different levels of reality.
# A fact ("I lost my job") cannot contradict a value ("I value stability") —
# the fact is an event, the value is a principle. They coexist without contradiction.
_CONFLICT_FORBIDDEN: FrozenSet[tuple] = frozenset([
    ("fact",  "value"),
    ("value", "fact"),
    ("state", "value"),
    ("value", "state"),
])


def _types_can_conflict(type_a: str, type_b: str) -> bool:
    """
    Returns False if creating a conflicts_with edge between these types
    is ontologically invalid. Empty/unknown types always pass (no gating).

    Forbidden: FACT↔VALUE, STATE↔VALUE (different levels of reality).
    Allowed: FACT↔FACT, FACT↔STATE, FACT↔INTENTION, VALUE↔VALUE,
             VALUE↔INTENTION, STATE↔STATE, STATE↔INTENTION, INTENTION↔INTENTION.
    """
    if not type_a or not type_b:
        return True  # legacy nodes — no type info, don't block
    return (type_a, type_b) not in _CONFLICT_FORBIDDEN


class SemanticDecisionLayer:
    """
    Deterministic epistemic decision engine.

    Takes raw facts extracted by the LLM + the current UserTBG and returns
    a TBGDelta — no LLM calls, fully testable.

    Decision tree per fact:
        MERGE range (≥0.88):
            + reversal signal + polarity opposition → strong_contradict + create uncertain node
            + otherwise                             → reinforce existing
        FLAG range (0.72–0.88):
            + reversal signal + polarity opposition → contradict + create node
            + otherwise                             → create both + conflicts_with edge
        NONE (< 0.72):
            → create new node (concept_id resolved via concept_registry)

    After all facts: performative sweep + optional NLI contradiction scan.
    """

    def __init__(self, nli_detector=None):
        """
        Parameters
        ----------
        nli_detector : NLIContradictionDetector | None
            Optional NLI contradiction detector. When provided, runs a semantic
            scan for conflicts_with pairs that polarity-dict misses.
            Controlled via TBG_NLI_ENABLED env var (see module bottom).
        """
        self.nli_detector = nli_detector

    def resolve(
        self,
        raw_facts: List[dict],
        user_text: str,
        tbg: UserTBG,
        raw_edges: Optional[List[dict]] = None,
    ) -> TBGDelta:

        is_reversal   = self._detect_reversal(user_text)
        perf_domains  = self._detect_performative(user_text)

        add_nodes:             List[BeliefNode] = []
        reinforce_ids:         List[str]        = []
        contradict_ids:        List[str]        = []
        strong_contradict_ids: List[str]        = []
        oscillation_edges:     List[BeliefEdge] = []
        label_to_new_id:       Dict[str, str]   = {}

        existing = tbg.nodes

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

            # Wheel-of-life routing fields (TBG_HER_ROUTING). Validated; empty if off/unknown.
            _raw_domain = str(raw.get("domain", "")).strip().lower()
            node_domain = _raw_domain if _raw_domain in VALID_DOMAINS else ""
            _raw_stance = str(raw.get("stance", "")).strip().lower()
            node_stance = _raw_stance if _raw_stance in VALID_STANCES else ""

            match_type, match_id, sim = _check_semantic_similarity(label, category, existing)

            # her_resolver gates — only on the MERGE branch, before any reinforce.
            # BLOCK -> downgrade to NEW/adjacent (treat as a distinct concept).
            # Both flags default OFF: with both off this block is a no-op and resolve
            # is byte-identical to the golden E0 baseline.
            if match_type == _SIM_MERGE and (_HER_ROUTING or _OPPOSITION_GATE):
                cand = existing[match_id]
                if _HER_ROUTING:
                    _new_shim = type("N", (), {"domain": node_domain})()
                    if her_resolver.gate(_new_shim, cand) == her_resolver.BLOCK_MERGE:
                        logger.info(
                            f"HER VETO cross-domain: '{label}'[{node_domain}] vs "
                            f"'{cand.label}'[{cand.domain}] -> NEW (no merge)"
                        )
                        match_type = _SIM_NONE
                if match_type == _SIM_MERGE and _OPPOSITION_GATE:
                    if _eval_prod(label, cand.label) < 0:
                        logger.info(
                            f"OPPOSITION-GATE: '{label}' vs '{cand.label}' eval_prod<0 -> NEW (no merge)"
                        )
                        match_type = _SIM_NONE

            if match_type == _SIM_MERGE:
                existing_label = existing[match_id].label
                existing_conf  = existing[match_id].confidence

                # Ensure existing node has a concept_id (backfill for old data)
                if not existing[match_id].concept_id:
                    existing[match_id].concept_id = _generate_concept_id(
                        existing[match_id].category, existing[match_id].label
                    )
                matched_concept_id = existing[match_id].concept_id
                # Step 4: cache alias for fast lookup next turn.
                if _CLOSED_VOCAB:
                    tbg.concept_aliases[label.lower().strip()] = matched_concept_id

                if is_reversal and self._is_opposition(label, existing_label,
                                                       node_type, existing[match_id].node_type):
                    # Explicit reversal of near-identical concept
                    if existing_conf >= _STRONG_CONTRADICT_THRESHOLD:
                        if match_id not in strong_contradict_ids:
                            strong_contradict_ids.append(match_id)
                        _prov("SDL_REVERSAL", "strong_contradict", match_id)
                    else:
                        if match_id not in contradict_ids:
                            contradict_ids.append(match_id)
                        _prov("SDL_REVERSAL", "contradict", match_id)
                    # New node at reduced confidence — tension is real but unresolved.
                    # Same concept_id: engine will merge these if they converge.
                    node = self._make_node(label, category, min(confidence, 0.50), source,
                                           concept_id=matched_concept_id, node_type=node_type,
                                           domain=node_domain, stance=node_stance)
                    add_nodes.append(node)
                    label_to_new_id[label.lower()] = node.id
                else:
                    if match_id not in reinforce_ids:
                        reinforce_ids.append(match_id)
                    _prov("SDL_COSINE_MERGE", "reinforce", match_id)
                    label_to_new_id[label.lower()] = match_id

            elif match_type == _SIM_FLAG:
                existing_label = existing[match_id].label
                existing_conf  = existing[match_id].confidence

                # Ensure existing node has a concept_id (backfill for old data)
                if not existing[match_id].concept_id:
                    existing[match_id].concept_id = _generate_concept_id(
                        existing[match_id].category, existing[match_id].label
                    )
                matched_concept_id = existing[match_id].concept_id
                # Step 4: cache alias for fast lookup next turn.
                if _CLOSED_VOCAB:
                    tbg.concept_aliases[label.lower().strip()] = matched_concept_id

                is_opp         = self._is_opposition(label, existing_label,
                                                      node_type, existing[match_id].node_type)

                if is_reversal and is_opp:
                    # Explicit reversal of a semantically related but distinct concept
                    if existing_conf >= _STRONG_CONTRADICT_THRESHOLD:
                        if match_id not in strong_contradict_ids:
                            strong_contradict_ids.append(match_id)
                        _prov("SDL_REVERSAL", "strong_contradict", match_id)
                    else:
                        if match_id not in contradict_ids:
                            contradict_ids.append(match_id)
                        _prov("SDL_REVERSAL", "contradict", match_id)
                    node = self._make_node(
                        label, category, min(confidence, 0.55), source,
                        concept_id=(matched_concept_id if _FIX_FLAG_CONCEPT_ID else None),
                        node_type=node_type, domain=node_domain, stance=node_stance,
                    )
                    add_nodes.append(node)
                    label_to_new_id[label.lower()] = node.id
                else:
                    # Oscillation pair — create new node, link as conflict
                    node = self._make_node(
                        label, category, confidence, source,
                        concept_id=(matched_concept_id if _FIX_FLAG_CONCEPT_ID else None),
                        node_type=node_type, domain=node_domain, stance=node_stance,
                    )
                    add_nodes.append(node)
                    label_to_new_id[label.lower()] = node.id
                    oscillation_edges.append(BeliefEdge(
                        source_id=node.id,
                        target_id=match_id,
                        relation="conflicts_with",
                        confidence=round(sim, 3),
                    ))
                    _prov("SDL_COSINE_FLAG", "edge", (node.id, match_id, "conflicts_with"))

            else:
                # No live-graph match. Registry may still hold the canonical
                # concept_id from an earlier turn whose node was pruned —
                # a "M&A attorney" message after "corporate lawyer" was archived.
                concept_id = _lookup_or_register(label, category, tbg)
                node = self._make_node(label, category, confidence, source,
                                       concept_id=concept_id, node_type=node_type,
                                       domain=node_domain, stance=node_stance)
                add_nodes.append(node)
                label_to_new_id[label.lower()] = node.id

                # Same-category opposition: link as conflicts_with only for the
                # STRONGEST opposing node in the same category (max 1 edge per new node).
                # Prevents O(n²) edge explosion in long conversations.
                if category and self._get_polarity(label) != 0:
                    best_opp_id = None
                    best_opp_sim = 0.0
                    new_emb = _get_embed_model().encode(label, normalize_embeddings=True)
                    for ex_id, ex_node in existing.items():
                        if (ex_node.category == category
                                and self._is_opposition(label, ex_node.label,
                                                         node_type, ex_node.node_type)
                                and ex_node.confidence >= 0.50):
                            sim_val = float(np.dot(
                                new_emb,
                                _get_embed_model().encode(ex_node.label, normalize_embeddings=True),
                            ))
                            if 0.35 <= sim_val < DEDUP_FLAG_THRESHOLD and sim_val > best_opp_sim:
                                best_opp_sim = sim_val
                                best_opp_id = ex_id
                    if best_opp_id:
                        oscillation_edges.append(BeliefEdge(
                            source_id=node.id,
                            target_id=best_opp_id,
                            relation="conflicts_with",
                            confidence=round(best_opp_sim, 3),
                        ))
                        if _PROV_ON:
                            self._is_opposition(label, existing[best_opp_id].label,
                                                node_type, existing[best_opp_id].node_type)
                            _prov("SDL_EPA" if _LAST_OPP_REASON == "epa" else "SDL_POLARITY",
                                  "edge", (node.id, best_opp_id, "conflicts_with"))

        # Performative sweep: contradict old-state nodes in affected domains
        if perf_domains:
            new_ids = set(label_to_new_id.values())
            all_contra = set(contradict_ids) | set(strong_contradict_ids)

            # R5: compute action embedding once for polarity check
            perf_direction: Optional[np.ndarray] = None
            if _FIX_PERFORMATIVE_POLARITY:
                perf_direction = _embed_for_polarity(user_text)

            for node in existing.values():
                if (node.category in perf_domains
                        and node.confidence >= 0.45
                        and node.id not in all_contra
                        and node.id not in new_ids
                        and node.id not in reinforce_ids):

                    if _FIX_PERFORMATIVE_POLARITY and perf_direction is not None:
                        polarity = _node_polarity_vs_action(node, perf_direction)
                        if polarity == "aligned":
                            logger.info(
                                f"R5 skip-contradict: '{node.label}' aligns with "
                                f"performative action, reinforcing instead"
                            )
                            reinforce_ids.append(node.id)
                            _prov("SDL_PERFORMATIVE", "reinforce", node.id)
                            continue

                    contradict_ids.append(node.id)
                    _prov("SDL_PERFORMATIVE", "contradict", node.id)

        # NLI semantic contradiction scan (optional, feature-flagged).
        # Catches oppositions that polarity-dict misses — labels with no
        # words in _NEGATIVE_POLARITY / _POSITIVE_POLARITY but semantically
        # contradicted (e.g. "wants to be artist" vs "values job stability").
        # Runs AFTER polarity logic so we don't double-add edges.
        nli_edges: List[BeliefEdge] = []
        if self.nli_detector is not None and add_nodes:
            existing_nodes_list = list(tbg.nodes.values())
            already_flagged = (
                set(contradict_ids)
                | set(strong_contradict_ids)
                | set(reinforce_ids)
            )
            # Build quick index for duplicate-edge check
            existing_edge_pairs = {
                (e.source_id, e.target_id)
                for e in oscillation_edges
            }
            try:
                nli_hits = self.nli_detector.find_contradictions(
                    new_nodes=add_nodes,
                    existing_nodes=existing_nodes_list,
                )
                for new_id, existing_id, score in nli_hits:
                    if (new_id, existing_id) in existing_edge_pairs:
                        continue  # polarity dict already created this edge
                    if existing_id in already_flagged:
                        continue  # already contradicted by polarity or perf sweep

                    # Type gate for NLI edges — same ontological rules as polarity
                    new_node_obj = next((n for n in add_nodes if n.id == new_id), None)
                    ex_node_obj  = tbg.nodes.get(existing_id)
                    if new_node_obj and ex_node_obj:
                        if not _types_can_conflict(new_node_obj.node_type,
                                                   ex_node_obj.node_type):
                            logger.debug(
                                f"TypeGate blocked NLI edge: "
                                f"'{new_node_obj.label}'[{new_node_obj.node_type}] ↔ "
                                f"'{ex_node_obj.label}'[{ex_node_obj.node_type}]"
                            )
                            continue

                    nli_edges.append(BeliefEdge(
                        source_id=new_id,
                        target_id=existing_id,
                        relation="conflicts_with",
                        confidence=round(min(0.6, score), 3),
                    ))
                    _prov("SDL_NLI", "edge", (new_id, existing_id, "conflicts_with"))
                    # No contradict_ids.append here — edge alone registers the
                    # conflict via _apply_graph_influence. Direct _apply_evidence
                    # on top would be double punishment (conf collapses to 0.13).
            except Exception as e:
                logger.warning(f"NLI scan failed gracefully: {e}")

        add_edges = (
            self._resolve_edges(raw_edges or [], tbg, label_to_new_id)
            + oscillation_edges
            + nli_edges
        )

        return TBGDelta(
            add_nodes=add_nodes,
            add_edges=add_edges,
            reinforce_ids=list(dict.fromkeys(reinforce_ids)),
            contradict_ids=list(dict.fromkeys(contradict_ids)),
            strong_contradict_ids=list(dict.fromkeys(strong_contradict_ids)),
        )

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _detect_reversal(text: str) -> bool:
        return any(p.search(text) for p in _REVERSAL_PATTERNS)

    @staticmethod
    def _detect_performative(text: str) -> Optional[frozenset]:
        for pattern, domains in _PERFORMATIVE:
            if pattern.search(text):
                return domains
        return None

    @staticmethod
    def _get_polarity(label: str) -> int:
        words = set(label.lower().split())
        if words & _NEGATIVE_POLARITY:
            return -1
        if words & _POSITIVE_POLARITY:
            return 1
        return 0

    @classmethod
    def _is_opposition(cls, label_a: str, label_b: str,
                       type_a: str = "", type_b: str = "") -> bool:
        # Type gate — ontologically invalid pairs never conflict
        if not _types_can_conflict(type_a, type_b):
            logger.debug(
                f"TypeGate blocked: '{label_a}'[{type_a}] ↔ '{label_b}'[{type_b}]"
            )
            return False

        global _LAST_OPP_REASON
        _LAST_OPP_REASON = "none"

        pa = cls._get_polarity(label_a)
        pb = cls._get_polarity(label_b)

        # Primary: polarity dict — fast, no model needed
        if pa != 0 and pb != 0:
            if pa * pb < 0:
                _LAST_OPP_REASON = "polarity"
            return pa * pb < 0  # opposite signs → opposition

        # Secondary: EPA — only when BOTH labels are neutral in polarity
        if pa != 0 or pb != 0:
            return False

        try:
            from tbg_axes import get_belief_axes
            # Single project_batch call — avoids triple encode that separate
            # project(a) + project(b) + axis_distance(a, b) would cause.
            proj_a, proj_b = get_belief_axes().project_batch([label_a, label_b])
            # Noise filter: skip near-zero projections (‖z‖ < 0.08 → no signal)
            norm_a = np.linalg.norm(list(proj_a.values()))
            norm_b = np.linalg.norm(list(proj_b.values()))
            if norm_a < 0.08 or norm_b < 0.08:
                return False
            dist = {ax: proj_a[ax] - proj_b[ax] for ax in proj_a}
            for axis in _EPA_AXES:
                if abs(dist.get(axis, 0.0)) > _EPA_OPP_THRESHOLD:
                    logger.debug(
                        f"EPA opposition: '{label_a}' ↔ '{label_b}' "
                        f"axis={axis} dist={dist[axis]:.3f}"
                    )
                    _LAST_OPP_REASON = "epa"
                    return True
        except Exception as _e:
            logger.debug(f"EPA opposition check failed: {_e}")

        return False

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
        )

    @staticmethod
    def _resolve_edges(
        raw_edges: List[dict],
        tbg: UserTBG,
        label_to_new_id: Dict[str, str],
    ) -> List[BeliefEdge]:
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
# NLI detector is lazy-loaded on first use (model not loaded at import).
# Feature flag: TBG_NLI_ENABLED=1 enables semantic contradiction detection.
_nli_detector = None
if os.getenv("TBG_NLI_ENABLED", "0") == "1":
    try:
        from tbg_nli import NLIContradictionDetector
        _nli_detector = NLIContradictionDetector(
            model_name  = os.getenv("TBG_NLI_MODEL",      "cross-encoder/nli-deberta-v3-small"),
            threshold   = float(os.getenv("TBG_NLI_THRESHOLD", "0.75")),
            top_k       = int(os.getenv("TBG_NLI_TOP_K",       "5")),
            batch_size  = int(os.getenv("TBG_NLI_BATCH_SIZE",  "16")),
        )
        logger.info("NLI contradiction detector registered (model load deferred to first use)")
    except Exception as e:
        logger.warning(f"TBG_NLI_ENABLED=1 but failed to init detector: {e}. NLI disabled.")

_sdl = SemanticDecisionLayer(nli_detector=_nli_detector)


# ---------------------------------------------------------------------------
# DETERMINISTIC EXTRACTOR — feature flag TBG_DETERMINISTIC_EXTRACTOR=1
# ---------------------------------------------------------------------------

class DeterministicExtractor:
    """
    Zero-LLM extraction path (TBG_DETERMINISTIC_EXTRACTOR=1).

    Algorithm per turn:
      1. Split user_text into spans on punctuation + conjunctions
      2. Project each span → EPA axes → z (dict)
      3. Select span with max ‖z‖; skip if ‖z‖ < 0.08 (noise)
      4. Cosine(best_span, existing_labels)
         sim ≥ 0.72 → dot(z_new, z_existing) > 0 → reinforce; < 0 → contradict
         sim < 0.72 → new node; label = span[:50]; conf = clip(‖z‖, 0.30, 0.65)
      5. Optional LLM naming: if new node and len(span) > 60, llm_call_fn shortens label.

    SemanticDecisionLayer is unchanged and still runs under flag=0.
    """

    _SPLIT_RE = re.compile(r'[.!?;]|\b(?:but|however|and|because|so)\b', re.I)
    _NORM_FLOOR = 0.08
    _CONF_MIN   = 0.30
    _CONF_MAX   = 0.65

    _NAMING_PROMPT = 'User said: \"{span}\". Give a short name (2-5 words) for this psychological belief. Return JSON: {{\"label\": \"your label here\"}}'

    @classmethod
    def split_to_spans(cls, text: str) -> List[str]:
        parts = cls._SPLIT_RE.split(text)
        return [p.strip() for p in parts if len(p.strip()) > 10]

    @staticmethod
    def _axis_norm(z: dict) -> float:
        return float(np.linalg.norm(list(z.values()))) if z else 0.0

    @staticmethod
    def _axis_dot(z_a: dict, z_b: dict) -> float:
        return sum(z_a.get(ax, 0.0) * z_b.get(ax, 0.0) for ax in z_a)

    @staticmethod
    def _infer_category(z: dict) -> str:
        """
        Route to TBG category via EPA axis pattern.
        Axes (sorted keys): activity, autonomy, evaluation, potency, self_other.

        Rules (priority order):
          evaluation↓ (< -0.08)               → mood (negative affect) or fears (+ potency↓)
          evaluation↑ AND potency↑ (> +0.08)  → career (if potency dominates) or goals
          autonomy↑ (> +0.08)                 → identity (if au > po) or values
          activity↑ (> +0.08)                 → goals
          fallback                             → values
        """
        if not z:
            return "values"
        ev = z.get("evaluation", 0.0)
        po = z.get("potency",    0.0)
        au = z.get("autonomy",   0.0)
        ac = z.get("activity",   0.0)

        if ev < -0.08:
            return "fears" if po < -0.08 else "mood"
        if ev > 0.08 and po > 0.08:
            return "career" if po > au else "goals"
        if au > 0.08:
            return "identity" if au > po else "values"
        if ac > 0.08:
            return "goals"
        return "values"

    def extract_sync(self, user_text: str, tbg: UserTBG) -> Optional[TBGDelta]:
        """Synchronous extraction — called via run_in_executor."""
        spans = self.split_to_spans(user_text)
        if not spans:
            return None

        # Project all spans through EPA axes
        try:
            from tbg_axes import get_belief_axes
            axes = get_belief_axes()
            projections = axes.project_batch(spans)
        except Exception as e:
            logger.debug(f"DeterministicExtractor: project_batch failed: {e}")
            return None

        # Select span with max ‖z‖
        scored = [(self._axis_norm(z), i, z) for i, z in enumerate(projections)]
        best_norm, best_idx, best_z = max(scored, key=lambda x: x[0])
        best_span = spans[best_idx]

        if best_norm < self._NORM_FLOOR:
            logger.debug(f"DeterministicExtractor: norm={best_norm:.3f} < floor, skip")
            return None

        # Cosine against existing nodes
        existing = tbg.nodes
        if not existing:
            return self._new_node_delta(best_span, best_z, best_norm)

        existing_ids = list(existing.keys())
        # Span-vs-span matching: use stored signal_span when available.
        # "my career is stable" vs "Actually my job is great" → sim ~0.65 (spans)
        # vs sim ~0.45 (span vs LLM-named label "Occupational Optimism").
        existing_texts = [
            (existing[nid].signal_span or existing[nid].label)
            for nid in existing_ids
        ]

        try:
            all_texts  = existing_texts + [best_span]
            embeddings = _embed_labels(all_texts)
            new_emb    = embeddings[-1]
            sims       = embeddings[:-1] @ new_emb

            best_sim_idx = int(np.argmax(sims))
            best_sim     = float(sims[best_sim_idx])
            match_id     = existing_ids[best_sim_idx]
        except Exception as e:
            logger.debug(f"DeterministicExtractor: embed failed: {e}")
            return self._new_node_delta(best_span, best_z, best_norm)

        if best_sim >= DEDUP_FLAG_THRESHOLD:
            ex_z = existing[match_id].axis_projection
            dot  = self._axis_dot(best_z, ex_z) if ex_z else 1.0
            if dot > 0:
                logger.debug(f"DeterministicExtractor: reinforce '{existing[match_id].label}' sim={best_sim:.2f} dot={dot:.3f}")
                return TBGDelta(reinforce_ids=[match_id])
            else:
                logger.debug(f"DeterministicExtractor: contradict '{existing[match_id].label}' sim={best_sim:.2f} dot={dot:.3f}")
                return TBGDelta(contradict_ids=[match_id])
        else:
            return self._new_node_delta(best_span, best_z, best_norm)

    def _new_node_delta(self, span: str, z: dict, norm: float) -> TBGDelta:
        # Keep full span as label — async path in extract_tbg_delta will:
        #   len > 40 → LLM naming (truncated to 50)
        #   len ≤ 40 → span[:30]
        label      = span[:150].strip()
        confidence = round(max(self._CONF_MIN, min(self._CONF_MAX, norm)), 3)
        category   = self._infer_category(z)
        node = BeliefNode(
            label=label,
            category=category,
            confidence=confidence,
            source="inferred",
            axis_projection=z,
            signal_span=span,
        )
        logger.debug(f"DeterministicExtractor: new node span='{label[:40]}' [{category}] conf={confidence}")
        return TBGDelta(add_nodes=[node])


_USE_DETERMINISTIC = os.getenv("TBG_DETERMINISTIC_EXTRACTOR", "0") == "1"
_det_extractor: Optional[DeterministicExtractor] = (
    DeterministicExtractor() if _USE_DETERMINISTIC else None
)

# Phase 4: enable via TBG_LABEL_CANONICALIZATION=1.
_LABEL_CANONICALIZATION_ENABLED = os.getenv("TBG_LABEL_CANONICALIZATION", "0") == "1"

# Step 2: disable the extraction_confidence-based penalty — the LLM's self-assessment
# is not calibrated against ground truth and injects variance into BDA.
_DISABLE_EXTRACTION_CONF_PENALTY = os.getenv("TBG_DISABLE_EXTRACTION_CONF_PENALTY", "0") == "1"

# Step 3: inherit concept_id from the matched node in the _SIM_FLAG branches.
# Without this, an oscillation pair of the same concept gets different concept_ids and
# is not deduplicated during prune.
_FIX_FLAG_CONCEPT_ID = os.getenv("TBG_FIX_FLAG_CONCEPT_ID", "0") == "1"

# R5: polarity check for performative action sweep.
# Without this, "quit job" contradicts ALL nodes in {career, identity, values,
# finances, goals} with conf >= 0.45 — including nodes whose polarity ALIGNS
# with the action (e.g., "wants freedom" gets contradicted, but quitting job
# REINFORCES wants_freedom).
_FIX_PERFORMATIVE_POLARITY = os.getenv("TBG_FIX_PERFORMATIVE_POLARITY", "0") == "1"

# Step 4: closed-vocabulary extraction + deterministic alias cache.
# Stage 1: the prompt passes concept_id alongside the label and requires exact copying.
# Stage 2: deterministic lookup in tbg.concept_aliases before cosine matching.
# Basis: GenIE (EPFL 2022), LLM StructCore (CL4Health 2026), LINK-KG (2025).
_CLOSED_VOCAB = os.getenv("TBG_CLOSED_VOCAB", "0") == "1"

# her_resolver integration — two independent flags, default OFF (to measure single-variable).
# TBG_HER_ROUTING: cross-domain VETO on the _SIM_MERGE branch (her_resolver.gate).
# TBG_OPPOSITION_GATE (spike-fix B): block the merge on opposition along the EPA evaluation axis
#   (eval_prod<0) — catches same-domain stability↔change, which routing doesn't see.
_HER_ROUTING = os.getenv("TBG_HER_ROUTING", "0") == "1"
_OPPOSITION_GATE = os.getenv("TBG_OPPOSITION_GATE", "0") == "1"

VALID_DOMAINS: FrozenSet[str] = frozenset({
    "career", "money", "health", "relationships",
    "family", "lifestyle", "meaning", "self", "other",
})
VALID_STANCES: FrozenSet[str] = frozenset({"approach", "avoid", "neutral"})


def _eval_prod(label_a: str, label_b: str) -> float:
    """Product of projections onto the EPA-evaluation axis. <0 => opposite
    valence (opposition). Any error -> 0.0 (don't block). Deterministic."""
    try:
        from tbg_axes import get_belief_axes
        pa, pb = get_belief_axes().project_batch([label_a, label_b])
        return float(pa.get("evaluation", 0.0) * pb.get("evaluation", 0.0))
    except Exception as e:
        logger.debug(f"_eval_prod failed: {e}")
        return 0.0


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


def _embed_for_polarity(text: str) -> Optional[np.ndarray]:
    """R5: embed text for polarity comparison. Returns None on failure."""
    try:
        return _get_embed_model().encode(text, normalize_embeddings=True)
    except Exception:
        return None


def _node_polarity_vs_action(node: "BeliefNode", action_embedding: np.ndarray) -> str:
    """
    R5: Returns 'aligned', 'opposed', or 'neutral'.
    Uses cosine similarity between node label and performative action text.
    Conservative thresholds: only mark as aligned if cos >= 0.5.
    """
    try:
        node_emb = _get_embed_model().encode(node.label, normalize_embeddings=True)
        cos = float(node_emb @ action_embedding)
        if cos >= 0.5:
            return "aligned"
        if cos <= -0.3:
            return "opposed"
    except Exception:
        pass
    return "neutral"


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


def _ret(result, status, return_status):
    """Backward-compatible return. Default: bare result (existing contract).
    Opt-in (benchmark/debug): (result, status) so callers can distinguish
    no_update / empty_facts / failed_json / failed_other from a real delta."""
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

    # Deterministic path — no LLM extraction.
    # LLM naming called only when: new node AND len(span) > 40 chars.
    if _USE_DETERMINISTIC and _det_extractor is not None:
        effective_tbg = tbg or UserTBG(user_id="_extract_dummy")
        loop  = asyncio.get_running_loop()
        delta = await loop.run_in_executor(
            None,
            lambda: _det_extractor.extract_sync(user_text, effective_tbg),
        )
        if delta is None:
            return _ret(None, "no_update", return_status)

        # LLM naming: span > 20 chars → LLM (JSON {"label": "..."}); else → [:30]
        for node in delta.add_nodes:
            if len(node.label) > 20 and llm_call_fn is not None:
                try:
                    prompt = DeterministicExtractor._NAMING_PROMPT.format(span=node.label[:200])
                    raw_label = await asyncio.wait_for(llm_call_fn(prompt), timeout=15.0)
                    # llm_call_fn returns JSON string; try to parse {"label": "..."}
                    short = ""
                    try:
                        parsed = json.loads(_clean_json(raw_label))
                        short = str(parsed.get("label") or parsed.get("name") or "").strip()[:50]
                    except Exception:
                        short = raw_label.strip().strip('"').strip("'")[:50]
                    node.label = short if short else node.label[:30]
                except Exception as e:
                    logger.debug(f"DeterministicExtractor: naming LLM failed: {e}")
                    node.label = node.label[:30]
            else:
                node.label = node.label[:30]

        logger.info(
            f"TBG delta [DET]: +{len(delta.add_nodes)} nodes "
            f"reinforce={len(delta.reinforce_ids)} "
            f"contradict={len(delta.contradict_ids)}"
        )
        return _ret(delta, "ok", return_status)

    existing_labels = _build_existing_labels(tbg) if tbg else "none"

    if _CLOSED_VOCAB:
        prompt_template = EXTRACTION_PROMPT_CLOSED_VOCAB
    else:
        prompt_template = EXTRACTION_PROMPT

    prompt = prompt_template.format(
        existing_labels=existing_labels,
        user_text=_smart_truncate(user_text, 600),
        assistant_text=_smart_truncate(assistant_text, 400),
    )

    raw_text = None
    cleaned = None
    try:
        raw_text = await llm_call_fn(prompt)
        cleaned = _clean_json(raw_text)
        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            # Repair tier: handle truncation / unclosed structures, retry once.
            cleaned = _repair_truncated_json(cleaned)
            raw = json.loads(cleaned)

        if raw.get("reasoning") == "no update" and not raw.get("facts"):
            return _ret(None, "no_update", return_status)

        raw_facts = raw.get("facts", []) or []
        raw_edges = raw.get("edges", []) or []
        extraction_confidence = min(1.0, max(0.0, float(raw.get("extraction_confidence", 0.7))))

        if not raw_facts:
            # Distinct from no_update: the model produced JSON but no facts. May
            # signal a format slip (e.g. edges present, or signal misrouted), not
            # genuine silence. Surface it instead of swallowing it.
            if _EXTRACT_DEBUG:
                logger.warning(
                    f"TBG empty_facts (edges={len(raw_edges)}): "
                    f"{str(raw.get('reasoning',''))[:80]} | raw={str(raw_text)[:200]!r}"
                )
            return _ret(None, "empty_facts", return_status)

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
        return _ret(delta, "ok", return_status)

    except json.JSONDecodeError as e:
        logger.warning(f"TBG: invalid JSON after repair attempt: {e}")
        if _EXTRACT_DEBUG:
            logger.warning(f"TBG raw output : {str(raw_text)[:400]!r}")
            logger.warning(f"TBG cleaned    : {str(cleaned)[:400]!r}")
        return _ret(None, "failed_json", return_status)
    except Exception as e:
        logger.error(f"TBG extraction error: {e}", exc_info=True)
        if _EXTRACT_DEBUG and raw_text is not None:
            logger.error(f"TBG raw output : {str(raw_text)[:400]!r}")
        return _ret(None, "failed_other", return_status)

