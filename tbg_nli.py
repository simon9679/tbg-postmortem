"""
TBG NLI Contradiction Detector

Adds semantic contradiction detection beyond polarity-word matching.
Uses a cross-encoder NLI model to find conflicts_with pairs that
_is_opposition() misses — specifically cases where labels have no words
from _NEGATIVE_POLARITY / _POSITIVE_POLARITY but are semantically opposed:

    "wants to be an artist" vs "values job stability"   ← not in polarity dict
    "I want to quit"        vs "stability matters to me" ← not in polarity dict

Architecture (cascade to keep latency bounded):
    1. Embedding pre-filter (MiniLM, already loaded in fact_engine):
       cosine similarity ≥ 0.3 → top-k candidates per new node.
       Semantic opposites are rarely close in embedding space, so the threshold
       is intentionally wide. top_k=5 default (CPU-safe; 10 × bidirectional = 20
       NLI pairs ≈ 400-800ms on CPU; 5 × bidir = 10 pairs ≈ 200-400ms).
    2. NLI cross-encoder on top-k pairs in BOTH directions.
       Bidirectional because NLI is not symmetric:
       contradiction(A,B) ≠ contradiction(B,A).
    3. Returns (new_id, existing_id, max_bidir_score) for pairs above threshold.

Feature flag: TBG_NLI_ENABLED=1 (env var, default OFF until validated on DriftBench).

CPU latency note: all estimates assume CPU-only (torch+cpu).
deberta-v3-small: ~200-400ms for 10 batched pairs.
deberta-v3-xsmall: ~100-200ms — use as fallback if P95 > 500ms.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tbg_schema import BeliefNode

logger = logging.getLogger(__name__)

# label order in cross-encoder/nli-deberta-v3-small output
# Verified from model card: ['contradiction', 'entailment', 'neutral']
_CONTRADICTION_IDX = 0
_ENTAILMENT_IDX = 1
_NEUTRAL_IDX = 2

# ---------------------------------------------------------------------------
# AUDIT STATS — accumulate across all find_contradictions calls in one session.
# Call get_audit_stats() after eval run to see distribution and detect issues:
#   scans_total == 0        → NLI not firing at all (env/import bug)
#   detections == 0         → NLI fires but threshold too high
#   detections / pairs > 0.5 → threshold too low (too many false positives)
# ---------------------------------------------------------------------------

_audit_stats: dict = {
    "scans_total":                0,
    "pairs_scored":               0,   # total bidirectional pairs sent to model
    "detections_above_threshold": 0,
    "all_scores":                 [],  # all per-pair max-bidir scores for distribution
}


def get_audit_stats() -> dict:
    """Return shallow copy of current audit counters."""
    s = dict(_audit_stats)
    s["all_scores"] = list(_audit_stats["all_scores"])
    return s


def reset_audit_stats() -> None:
    """Reset all counters. Call between eval runs."""
    global _audit_stats
    _audit_stats = {
        "scans_total":                0,
        "pairs_scored":               0,
        "detections_above_threshold": 0,
        "all_scores":                 [],
    }


# Module-level singleton — lazy-loaded on first use
_nli_model = None


def _get_nli_model(model_name: str):
    """Load NLI CrossEncoder once, share across requests."""
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"NLI: loading '{model_name}' (CPU-only, first load ~5-10s)...")
            _nli_model = CrossEncoder(model_name, num_labels=3)
            logger.info("NLI: model ready")
        except Exception as e:
            logger.error(f"NLI: failed to load model '{model_name}': {e}")
            raise
    return _nli_model


class NLIContradictionDetector:
    """
    Semantic contradiction detector using an NLI cross-encoder.

    Designed to be cheap to instantiate (lazy model load) and reused
    as a SemanticDecisionLayer attribute across all coroutines.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Default: cross-encoder/nli-deberta-v3-small.
        Fallback: cross-encoder/nli-deberta-v3-xsmall (faster, lower quality).
    threshold : float
        Min contradiction score to register a conflicts_with edge.
        0.75 is conservative. Lower to 0.65 if OCS doesn't rise;
        raise to 0.85 if too many false positives.
    top_k : int
        Candidates per new node after embedding pre-filter.
        5 = ~10 NLI pairs (bidir) ≈ 200-400ms on CPU.
        10 = ~20 NLI pairs ≈ 400-800ms on CPU.
    batch_size : int
        NLI batch size per .predict() call. 16 optimal for CPU.
    min_existing_for_nli : int
        Skip NLI if graph has fewer existing nodes. No point scanning
        an almost-empty graph; avoids cold-start overhead.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        threshold: float = 0.75,
        top_k: int = 5,
        batch_size: int = 16,
        min_existing_for_nli: int = 3,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.top_k = top_k
        self.batch_size = batch_size
        self.min_existing_for_nli = min_existing_for_nli
        self._model = None  # loaded lazily on first call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_pair(self, premise: str, hypothesis: str) -> dict:
        """
        NLI score for a single pair.
        Returns {'contradiction': float, 'entailment': float, 'neutral': float}.
        Primarily used for unit tests and audit.
        """
        model = self._load()
        scores = model.predict([[premise, hypothesis]], apply_softmax=True)[0]
        return {
            "contradiction": float(scores[_CONTRADICTION_IDX]),
            "entailment":    float(scores[_ENTAILMENT_IDX]),
            "neutral":       float(scores[_NEUTRAL_IDX]),
        }

    def find_contradictions(
        self,
        new_nodes: List["BeliefNode"],
        existing_nodes: List["BeliefNode"],
        top_k_candidates: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Find contradicting pairs between new_nodes and existing_nodes.

        Returns
        -------
        List of (new_node_id, existing_node_id, contradiction_score)
        for pairs where max bidirectional NLI contradiction ≥ threshold.
        Each pair appears at most once (deduped by (new_id, existing_id)).
        """
        if not new_nodes or len(existing_nodes) < self.min_existing_for_nli:
            return []

        top_k = top_k_candidates or self.top_k

        # Step 1: embedding pre-filter -----------------------------------------
        from fact_engine import get_embed_model
        embed_model = get_embed_model()

        new_labels      = [n.label for n in new_nodes]
        existing_labels = [n.label for n in existing_nodes]

        try:
            all_labels = new_labels + existing_labels
            all_embs   = embed_model.encode(all_labels, normalize_embeddings=True, show_progress_bar=False)
            new_embs      = all_embs[:len(new_labels)]
            existing_embs = all_embs[len(new_labels):]
        except Exception as e:
            logger.warning(f"NLI: embedding pre-filter failed: {e}")
            return []

        # Cosine similarity matrix: shape (n_new, n_existing)
        sim_matrix = new_embs @ existing_embs.T

        # Step 2: collect top-k candidate pairs --------------------------------
        pairs_to_score: List[Tuple[int, int]] = []  # (new_idx, existing_idx)

        for ni in range(len(new_nodes)):
            sims     = sim_matrix[ni]
            # Wide filter: sim ≥ 0.3. Semantic opposites can be distant in
            # embedding space but still pass NLI — keep the net wide.
            above    = np.where(sims >= 0.3)[0]
            if len(above) == 0:
                continue
            sorted_i = above[np.argsort(sims[above])[::-1]]
            for ei in sorted_i[:top_k]:
                pairs_to_score.append((ni, int(ei)))

        if not pairs_to_score:
            return []

        # Step 3: bidirectional NLI batch --------------------------------------
        model = self._load()
        batch_inputs = []
        for ni, ei in pairs_to_score:
            batch_inputs.append([new_labels[ni],      existing_labels[ei]])  # A→B
            batch_inputs.append([existing_labels[ei], new_labels[ni]])       # B→A

        _audit_stats["scans_total"] += 1
        _audit_stats["pairs_scored"] += len(batch_inputs)

        try:
            all_scores = model.predict(
                batch_inputs,
                apply_softmax=True,
                batch_size=self.batch_size,
            )
        except Exception as e:
            logger.warning(f"NLI: predict failed: {e}")
            return []

        # Step 4: extract contradictions above threshold -----------------------
        results: List[Tuple[str, str, float]] = []
        seen: set = set()

        for i, (ni, ei) in enumerate(pairs_to_score):
            scores_ab = all_scores[2 * i]
            scores_ba = all_scores[2 * i + 1]

            # Take max of both directions — if EITHER direction is a contradiction,
            # the pair is contradictory.
            contradiction_score = max(
                float(scores_ab[_CONTRADICTION_IDX]),
                float(scores_ba[_CONTRADICTION_IDX]),
            )

            _audit_stats["all_scores"].append(contradiction_score)
            if contradiction_score >= self.threshold:
                _audit_stats["detections_above_threshold"] += 1

            if contradiction_score < self.threshold:
                continue

            new_id      = new_nodes[ni].id
            existing_id = existing_nodes[ei].id

            pair_key = (new_id, existing_id)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            logger.info(
                "NLI contradiction: '%s' <-> '%s' score=%.3f",
                new_labels[ni], existing_labels[ei], contradiction_score,
            )
            results.append((new_id, existing_id, contradiction_score))

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self):
        if self._model is None:
            self._model = _get_nli_model(self.model_name)
        return self._model
