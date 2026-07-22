"""
her_resolver — cross-domain false-merge VETO. A clean, detachable layer (OSS candidate).

Sole job: prevent merging two concepts from DIFFERENT wheel-of-life domains.
This closes the "career security ↔ financial security" catastrophe (career vs money),
which cosine cannot suppress (see the E3 series: cosine measures lexicon, not identity).

Deliberately NOT included:
  - cosine penalty / threshold parameters (cosine-based PENALTY is dead — E3a proved it);
  - self-exempt (we do NOT exempt self — decided by the data: cheaper to lose
    a rare cross-domain synonym via self than to open a hole in the VETO);
  - reading anything other than .domain.

The gate is symmetric and deterministic. domain="" (legacy / routing off) => ALLOW
(we never block when information is absent — a safe no-op).
"""
from typing import Any

BLOCKING = {
    "career", "money", "health", "relationships",
    "family", "lifestyle", "meaning", "self", "other",
}

BLOCK_MERGE = "BLOCK_MERGE"
ALLOW = "ALLOW"


def gate(new: Any, cand: Any) -> str:
    """new, cand — any objects with a .domain attribute (BeliefNode or shim).

    BLOCK_MERGE  — both domains are known and DIFFERENT (cross-domain catastrophe).
    ALLOW        — domains match, or at least one is unknown ("").
    """
    nd = (getattr(new, "domain", "") or "").strip().lower()
    cd = (getattr(cand, "domain", "") or "").strip().lower()
    if nd in BLOCKING and cd in BLOCKING and nd != cd:
        return BLOCK_MERGE
    return ALLOW
