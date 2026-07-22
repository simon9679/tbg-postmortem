#!/usr/bin/env python3
"""
test_policy_logic — deterministic, no LLM.

Exercises the trigger boundaries (POS_FLOOR floor + DOM_K dominance guard) and the
HOLD / ALLOW_UPDATE / PASS mapping by injecting belief nodes and classification
results directly. No tuning: thresholds are asserted as structural boundaries.
"""
import sys

sys.path.insert(0, ".")
from tbg_schema import UserTBG, BeliefNode
import policy
from policy import (
    POS_FLOOR, DOM_K, HOLD, ALLOW_UPDATE, PASS,
    SOCIAL_PRESSURE, NEW_EVIDENCE, NONE,
    anchored_beliefs, is_anchored, _map_decision,
)

GREEN = "\033[92m"; RED = "\033[91m"; BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"

_results = []


def check(name, cond):
    _results.append(cond)
    print(f"  {GREEN+'PASS' if cond else RED+'FAIL'}{RESET}  {name}")


def mk(label, pos, neg, conf=0.85):
    return BeliefNode(
        label=label, category="values", confidence=conf, source="explicit",
        pos_evidence=pos, neg_evidence=neg,
        concept_id="values:" + label.replace(" ", "_"),
    )


def tbg_with(*nodes):
    t = UserTBG(user_id="logic")
    for n in nodes:
        t.set_node(n)
    return t


def main():
    print(f"\n{BOLD}test_policy_logic — deterministic policy logic{RESET}")
    print(f"{DIM}POS_FLOOR={POS_FLOOR} (= 2 * EVIDENCE_WEIGHTS['medium_pos']), DOM_K={DOM_K}{RESET}\n")

    # POS_FLOOR must come from the engine weights, not a literal.
    from tbg_engine import EVIDENCE_WEIGHTS
    check("POS_FLOOR derived from EVIDENCE_WEIGHTS (= 2*medium_pos)",
          POS_FLOOR == 2 * EVIDENCE_WEIGHTS["medium_pos"])

    # --- Anchor trigger boundaries ----------------------------------------
    print(f"\n{BOLD}Trigger boundaries{RESET}")
    strong = mk("values stability", pos=3.05, neg=0.0)        # clean anchor
    once = mk("likes jazz", pos=0.85, neg=0.0)                # said once -> below floor
    contested = mk("wants to relocate", pos=2.2, neg=2.2)     # pos ~= neg -> not dominant
    borderline_ok = mk("values family", pos=3.0, neg=1.4)     # 3.0 >= 2*1.4=2.8 -> anchor
    borderline_no = mk("values travel", pos=3.0, neg=1.6)     # 3.0 <  2*1.6=3.2 -> not

    check("clean anchor (pos=3.05,neg=0) IS anchored", is_anchored(strong))
    check("said-once (pos=0.85 < floor) is NOT anchored (floor guard)", not is_anchored(once))
    check("contested (pos=neg=2.2) is NOT anchored (dominance guard)", not is_anchored(contested))
    check("dominant borderline (3.0 >= 2*1.4) IS anchored", is_anchored(borderline_ok))
    check("non-dominant borderline (3.0 < 2*1.6) is NOT anchored", not is_anchored(borderline_no))

    t = tbg_with(strong, once, contested)
    anchored = anchored_beliefs(t)
    check("anchored_beliefs returns only the clean anchor",
          [n.label for n in anchored] == ["values stability"])

    # --- Mapping: anchored + push_type ------------------------------------
    print(f"\n{BOLD}Decision mapping{RESET}")
    r = _map_decision(anchored, "values stability", SOCIAL_PRESSURE)
    check("anchored + social_pressure -> HOLD", r.action == HOLD)
    print(f"    {DIM}{r.rationale}{RESET}")

    r = _map_decision(anchored, "values stability", NEW_EVIDENCE)
    check("anchored + new_evidence -> ALLOW_UPDATE", r.action == ALLOW_UPDATE)
    print(f"    {DIM}{r.rationale}{RESET}")

    r = _map_decision(anchored, None, NONE)
    check("target=none -> PASS", r.action == PASS)

    # Contested node targeted by pressure: it never reaches `anchored`, so even a
    # social_pressure classification maps to PASS, not HOLD (dominance guard works
    # end to end).
    r = _map_decision(anchored, "wants to relocate", SOCIAL_PRESSURE)
    check("contested target + social_pressure -> PASS, not HOLD", r.action == PASS)

    # Below-floor node targeted by pressure: also not anchored -> PASS.
    r = _map_decision(anchored, "likes jazz", SOCIAL_PRESSURE)
    check("below-floor target + social_pressure -> PASS (floor works)", r.action == PASS)

    # --- B3: DOM_K validation on anchor-sanity-shaped data -----------------
    print(f"\n{BOLD}B3 — DOM_K={DOM_K} separates clean anchor from contested{RESET}")
    # From anchor-sanity: a real anchor reached pos~3.05/neg~0; a contested belief
    # would carry comparable pos and neg. DOM_K must let the first through and
    # block the second.
    clean = mk("anchor", pos=3.05, neg=0.0)
    contested2 = mk("contested", pos=2.2, neg=2.2)
    check("DOM_K passes clean anchor (pos=3.05,neg=0)", is_anchored(clean))
    check("DOM_K blocks contested (pos=2.2,neg=2.2)", not is_anchored(contested2))
    sep = is_anchored(clean) and not is_anchored(contested2)
    print(f"    {DIM}DOM_K={DOM_K} separates cleanly -> keep (no retune needed){RESET}"
          if sep else f"    {RED}DOM_K={DOM_K} does NOT separate -> revisit once, then fix{RESET}")

    # --- summary ----------------------------------------------------------
    ok = all(_results)
    print(f"\n{BOLD}{(GREEN+'ALL PASS') if ok else (RED+'FAILURES')}{RESET}  "
          f"({sum(_results)}/{len(_results)})\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
