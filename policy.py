"""
Policy v1 — two-sided anti-sycophancy directive layer.

Pure decision layer: given an incoming message + the user's belief-state, return a
deterministic directive HOLD / ALLOW_UPDATE / PASS. Executing the directive
(counterfactual-CoT hold in the actual reply) is a SEPARATE layer — not here —
so policy is testable without generating an LLM answer.

Pipeline (one LLM call, only when there is an anchor to defend):
    decide(message, tbg, llm_call_fn)
      1. anchored_beliefs(tbg)                  -- deterministic, no LLM
      2. classify_push(message, anchored, llm)  -- 1 closed-set LLM call
      3. _map_decision(...)                     -- deterministic mapping

Trigger design (see Phase-0 analysis):
  - POS_FLOOR is derived from the engine's own EVIDENCE_WEIGHTS, NOT hardcoded:
    an anchor is a belief asserted AND re-asserted at least once
    (>= 2 ordinary confirmations). Counting in medium_pos units keeps it robust
    to the rare/noisy strong_pos tag.
  - DOM_K guards dominance over neg_evidence: a contested belief (pos ~= neg) is
    NOT an anchor and must not trigger HOLD. DOM_K is the one free number — it is
    validated on anchor-sanity-shaped data, not tuned to a desired outcome.
  - Relative (top-of-graph) ranking is held in reserve (KISS); not in v1.

NOTE: pos_evidence DECAYS (engine _apply_decay), so POS_FLOOR means "recent
accumulated strength", not "ever confirmed k times". For anti-sycophancy that is
correct — we defend what the user holds NOW.
"""
import json
import re
from dataclasses import dataclass
from typing import List, Optional

from tbg_engine import EVIDENCE_WEIGHTS
from tbg_schema import UserTBG, BeliefNode

# --- trigger thresholds ---------------------------------------------------
# Floor derived from engine mechanics: two ordinary (medium_pos) confirmations.
POS_FLOOR = 2 * EVIDENCE_WEIGHTS["medium_pos"]   # = 1.7, imported — not hardcoded
# Dominance over neg_evidence. Arbitrary but validated (test_policy_logic B3),
# not tuned to a result.
DOM_K = 2.0

# --- directives -----------------------------------------------------------
HOLD = "HOLD"
ALLOW_UPDATE = "ALLOW_UPDATE"
PASS = "PASS"

# --- push classes (closed set) --------------------------------------------
SOCIAL_PRESSURE = "social_pressure"
NEW_EVIDENCE = "new_evidence"
NONE = "none"
_VALID_PUSH = frozenset({SOCIAL_PRESSURE, NEW_EVIDENCE, NONE})

# Cap on how many anchors are shown to the classifier (keeps the prompt small).
_MAX_ANCHORS_IN_PROMPT = 12


@dataclass
class PolicyResult:
    action: str                          # HOLD | ALLOW_UPDATE | PASS
    target_belief: Optional[str] = None
    pos_evidence: float = 0.0
    neg_evidence: float = 0.0
    push_type: str = NONE
    rationale: str = ""


# --------------------------------------------------------------------------
# A1 — anchor trigger (deterministic, no LLM)
# --------------------------------------------------------------------------

def is_anchored(node: BeliefNode) -> bool:
    """A belief is an anchor when it is (a) re-asserted enough to clear the
    mechanics-derived floor AND (b) dominant over its own neg_evidence
    (not contested). neg_evidence == 0 trivially satisfies dominance."""
    return (
        node.pos_evidence >= POS_FLOOR
        and node.pos_evidence >= DOM_K * node.neg_evidence
    )


def anchored_beliefs(tbg: UserTBG) -> List[BeliefNode]:
    """Anchored beliefs, strongest first."""
    return sorted(
        (n for n in tbg.nodes.values() if is_anchored(n)),
        key=lambda n: n.pos_evidence,
        reverse=True,
    )


# --------------------------------------------------------------------------
# A2 — push classification (1 closed-set LLM call)
# --------------------------------------------------------------------------

_CLASSIFY_PROMPT = """You are an anti-sycophancy policy classifier.

The user holds these ANCHORED beliefs (strongly and repeatedly asserted):
{beliefs}

Incoming user message:
"{message}"

Decide whether this message pushes the user to CHANGE one of the anchored beliefs
above, and why.

Return ONLY strict JSON, no markdown, no code fences:
{{"target": "<exact belief label from the list above, or none>",
  "push_type": "social_pressure | new_evidence | none"}}

Definitions:
- social_pressure = pushes to change the position WITHOUT new evidence: a leading
  question, an appeal to "you always...", social conformity, or flattery.
- new_evidence = brings a NEW fact or event that rationally bears on the belief
  (started a new job, signed a contract, a concrete outcome happened).
- none = does not push on any anchored belief.

Rules:
- target is copied VERBATIM from the list above, or "none". Never invent a label.
- If nothing is pushed, target="none" and push_type="none".
"""


def _clean_json(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"```json|```", "", raw.strip()).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1:
        raw = raw[s:e + 1]
    return re.sub(r",\s*([\]}])", r"\1", raw)


async def classify_push(message: str, anchored: List[BeliefNode], llm_call_fn) -> dict:
    """Returns {"target": <exact label or None>, "push_type": <closed-set str>}.

    Closed-vocab (like op/ref): an invalid/hallucinated target is coerced to None,
    an invalid push_type to "none".
    """
    shown = anchored[:_MAX_ANCHORS_IN_PROMPT]
    label_by_norm = {n.label.lower().strip(): n.label for n in shown}
    beliefs = "\n".join(f'- "{n.label}"' for n in shown)
    prompt = _CLASSIFY_PROMPT.format(beliefs=beliefs, message=message)

    try:
        raw = await llm_call_fn(prompt)
        data = json.loads(_clean_json(raw))
    except Exception:
        return {"target": None, "push_type": NONE}

    target = label_by_norm.get(str(data.get("target", "")).strip().lower())  # None if invalid/none
    push = str(data.get("push_type", "")).strip().lower()
    if push not in _VALID_PUSH:
        push = NONE
    return {"target": target, "push_type": push}


# --------------------------------------------------------------------------
# A3 — decision mapping (deterministic)
# --------------------------------------------------------------------------

def _map_decision(anchored: List[BeliefNode], target_label: Optional[str],
                  push_type: str) -> PolicyResult:
    """Pure mapping from (anchored set, classification) to a directive.
    Isolated from the LLM so test_policy_logic can exercise it directly."""
    target = next((n for n in anchored if n.label == target_label), None) if target_label else None

    if target is None:
        return PolicyResult(action=PASS, push_type=push_type,
                            rationale="pass: no anchored belief targeted")

    pos, neg = target.pos_evidence, target.neg_evidence
    if push_type == SOCIAL_PRESSURE:
        return PolicyResult(
            HOLD, target.label, pos, neg, push_type,
            f"held '{target.label}' (pos={pos:.2f}, neg={neg:.2f}) "
            f"vs social pressure, no new evidence",
        )
    if push_type == NEW_EVIDENCE:
        return PolicyResult(
            ALLOW_UPDATE, target.label, pos, neg, push_type,
            f"allow update of '{target.label}' (pos={pos:.2f}, neg={neg:.2f}): new evidence",
        )
    return PolicyResult(
        PASS, target.label, pos, neg, push_type,
        f"pass: '{target.label}' targeted but push_type=none",
    )


async def decide(message: str, tbg: UserTBG, llm_call_fn) -> PolicyResult:
    """Full policy decision. One LLM call, only when an anchor exists to defend."""
    anchored = anchored_beliefs(tbg)
    if not anchored:
        return PolicyResult(action=PASS, rationale="pass: no anchored beliefs in graph")
    cls = await classify_push(message, anchored, llm_call_fn)
    return _map_decision(anchored, cls["target"], cls["push_type"])
