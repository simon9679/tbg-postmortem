"""
TBG Cognitive Axes — S2.1

Semantic projection of belief labels onto 5 orthogonal axes.
Each axis is a unit vector in MiniLM embedding space, computed as:
    mean(embed(positive_pole_words)) - mean(embed(negative_pole_words))
    normalized to unit length.

Projection score ∈ [-1.0, 1.0]:
    > 0  → label leans toward positive pole
    < 0  → label leans toward negative pole
    ≈ 0  → neutral / unrelated to this axis

Axes are based on Osgood (1957) EPA model + McAdams self/other dimension.
Read-only layer — does NOT modify tbg_engine, tbg_extractor, or tbg_schema.
"""

import logging
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Axes configuration
# ---------------------------------------------------------------------------

AXES_CONFIG = {
    "evaluation": {
        "description": "good <-> bad (Osgood 1957)",
        "positive_pole": [
            "good", "positive", "beneficial", "valuable", "meaningful",
            "wonderful", "excellent", "right", "moral", "healthy",
        ],
        "negative_pole": [
            "bad", "negative", "harmful", "worthless", "meaningless",
            "terrible", "awful", "wrong", "immoral", "toxic",
        ],
    },
    "potency": {
        "description": "strong <-> weak (Osgood 1957)",
        "positive_pole": [
            "strong", "powerful", "capable", "dominant", "confident",
            "decisive", "effective", "in control", "resilient", "assertive",
        ],
        "negative_pole": [
            "weak", "powerless", "incapable", "submissive", "insecure",
            "indecisive", "ineffective", "helpless", "fragile", "passive",
        ],
    },
    "activity": {
        "description": "active <-> passive (Osgood 1957)",
        "positive_pole": [
            "active", "energetic", "moving", "dynamic", "engaged",
            "doing", "working", "striving", "pursuing", "acting",
        ],
        "negative_pole": [
            "passive", "still", "stagnant", "inactive", "withdrawn",
            "waiting", "avoiding", "stuck", "frozen", "idle",
        ],
    },
    "autonomy": {
        "description": "freedom <-> obligation",
        "positive_pole": [
            "free", "independent", "autonomous", "self-directed", "my choice",
            "want to", "decide for myself", "on my own terms", "liberated", "unconstrained",
        ],
        "negative_pole": [
            "obligated", "must", "should", "required", "duty",
            "have to", "expected of me", "constrained", "trapped", "no choice",
        ],
    },
    "self_other": {
        "description": "self-focused <-> other-focused (McAdams)",
        "positive_pole": [
            "I want", "my needs", "for myself", "self-care", "personal",
            "my own", "I feel", "my goals", "self-development", "about me",
        ],
        "negative_pole": [
            "for others", "their needs", "family", "responsibility to",
            "I owe", "they expect", "for the team", "others first", "duty to others", "selfless",
        ],
    },
}


# ---------------------------------------------------------------------------
# BeliefAxes
# ---------------------------------------------------------------------------

class BeliefAxes:
    """
    Semantic projection axes for belief labels.

    Each axis is a unit vector computed as:
        mean(embed(positive_pole_words)) - mean(embed(negative_pole_words))
    normalized to unit length.

    Projection of a label onto an axis:
        score = dot(embed(label), axis_vector)  in [-1.0, 1.0]
        > 0  -> label leans toward positive pole
        < 0  -> label leans toward negative pole
        ~0   -> neutral / unrelated to this axis

    Lazy-loaded: axes computed on first call to project() or on explicit init().
    """

    def __init__(self):
        self._axes: Dict[str, np.ndarray] = {}   # axis_name -> unit vector
        self._model = None                         # sentence-transformers model
        self._ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Compute axis vectors. Call once at startup after embed model is loaded."""
        from fact_engine import get_embed_model
        self._model = get_embed_model()

        for axis_name, cfg in AXES_CONFIG.items():
            pos_words = cfg["positive_pole"]
            neg_words = cfg["negative_pole"]

            pos_embs = self._model.encode(pos_words, normalize_embeddings=True, show_progress_bar=False)
            neg_embs = self._model.encode(neg_words, normalize_embeddings=True, show_progress_bar=False)

            pos_mean = pos_embs.mean(axis=0)
            neg_mean = neg_embs.mean(axis=0)

            axis_vec = pos_mean - neg_mean
            norm = np.linalg.norm(axis_vec)
            if norm > 1e-8:
                axis_vec = axis_vec / norm

            self._axes[axis_name] = axis_vec.astype(np.float32)

        self._ready = True
        logger.info(f"BeliefAxes: initialized {len(self._axes)} axes")

    def project(self, label: str) -> Dict[str, float]:
        """
        Project a belief label onto all 5 axes.
        Returns dict: axis_name -> float in [-1.0, 1.0], rounded to 3 decimal places.
        Returns {} if axes not initialized and lazy-init fails.
        """
        if not self._ready:
            self.init()
        if not self._axes:
            return {}

        emb = self._model.encode(label, normalize_embeddings=True, show_progress_bar=False)
        return {
            ax: round(float(np.dot(emb, vec)), 3)
            for ax, vec in self._axes.items()
        }

    def project_batch(self, labels: List[str]) -> List[Dict[str, float]]:
        """
        Project multiple labels at once (single encode call — efficient).
        Returns list of dicts in same order as labels.
        """
        if not labels:
            return []
        if not self._ready:
            self.init()
        if not self._axes:
            return [{} for _ in labels]

        # shape: (n_labels, embed_dim)
        embs = self._model.encode(labels, normalize_embeddings=True, show_progress_bar=False)

        # Build axes matrix: (embed_dim, n_axes)
        axis_names = list(self._axes.keys())
        axes_matrix = np.stack([self._axes[ax] for ax in axis_names], axis=1)

        # scores_matrix: (n_labels, n_axes)
        scores_matrix = embs @ axes_matrix

        result = []
        for row in scores_matrix:
            result.append({
                ax: round(float(row[i]), 3)
                for i, ax in enumerate(axis_names)
            })
        return result

    def axis_distance(self, label_a: str, label_b: str) -> Dict[str, float]:
        """
        Signed distance between two labels on each axis.
        distance[axis] = projection(a) - projection(b)

        Large absolute value -> strong divergence on this axis.
        Useful for conflict detection: |distance| > 0.6 -> potential opposition.
        """
        projections = self.project_batch([label_a, label_b])
        proj_a, proj_b = projections[0], projections[1]
        return {
            ax: round(proj_a[ax] - proj_b[ax], 3)
            for ax in proj_a
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_belief_axes = BeliefAxes()


def get_belief_axes() -> BeliefAxes:
    """Return initialized BeliefAxes singleton."""
    return _belief_axes


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    axes = BeliefAxes()
    axes.init()

    # Test 1: Evaluation axis -- "I love my job" vs "I hate my job"
    p1 = axes.project("I love my job")
    p2 = axes.project("I hate my job")
    assert p1["evaluation"] > p2["evaluation"], (
        f"evaluation: love={p1['evaluation']:.3f} should > hate={p2['evaluation']:.3f}"
    )
    print(f"OK evaluation: love={p1['evaluation']:.3f} > hate={p2['evaluation']:.3f}")

    # Test 2: Potency axis -- "I feel powerful" vs "I feel helpless"
    p3 = axes.project("I feel powerful and in control")
    p4 = axes.project("I feel helpless and powerless")
    assert p3["potency"] > p4["potency"], (
        f"potency: powerful={p3['potency']:.3f} should > helpless={p4['potency']:.3f}"
    )
    print(f"OK potency: powerful={p3['potency']:.3f} > helpless={p4['potency']:.3f}")

    # Test 3: Autonomy axis -- "I want to" vs "I have to"
    p5 = axes.project("I want to change careers on my own terms")
    p6 = axes.project("I have to stay because of obligations")
    assert p5["autonomy"] > p6["autonomy"], (
        f"autonomy: want={p5['autonomy']:.3f} should > have_to={p6['autonomy']:.3f}"
    )
    print(f"OK autonomy: want={p5['autonomy']:.3f} > have_to={p6['autonomy']:.3f}")

    # Test 4: axis_distance -- "freedom" vs "obligation" should diverge on autonomy
    dist = axes.axis_distance("personal freedom", "family obligation")
    assert abs(dist["autonomy"]) > 0.3, (
        f"autonomy distance too small: {dist['autonomy']:.3f}"
    )
    print(f"OK axis_distance autonomy: {dist['autonomy']:.3f}")

    # Test 5: project_batch returns same as individual project
    batch = axes.project_batch(["I love my job", "I hate my job"])
    assert abs(batch[0]["evaluation"] - p1["evaluation"]) < 1e-4
    assert abs(batch[1]["evaluation"] - p2["evaluation"]) < 1e-4
    print(f"OK project_batch consistent with project()")

    print("\nAll tests passed")
