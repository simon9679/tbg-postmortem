#!/usr/bin/env python3
"""
test_policy_live — end-to-end policy over a real LLM (cerebras), MockDB.

Build a genuine anchor by re-asserting one belief (op/ref + evidence_type ON), then
run policy.decide() on three messages and check the directive:
  - pure social pressure          -> HOLD
  - evidence-bearing message      -> ALLOW_UPDATE
  - irrelevant message            -> PASS
Structural: assertions read PolicyResult.action; no hardcoded labels.
"""
import os
import sys
import asyncio

sys.path.insert(0, ".")
from tbg_schema import UserTBG
from tbg_engine import TBGEngine
from tbg_extractor import extract_tbg_delta
import policy
from policy import HOLD, ALLOW_UPDATE, PASS
from llm_client import gemini_call

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
_PROVIDER_KEY = {
    "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}.get(LLM_PROVIDER, "GEMINI_API_KEY")

GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"
BOLD = "\033[1m"; DIM = "\033[2m"; CYAN = "\033[96m"; RESET = "\033[0m"


async def llm_async(prompt: str) -> str:
    return await gemini_call(prompt, timeout=60.0)


class MockDB:
    async def fetchrow(self, *a, **kw): return None
    async def execute(self, *a, **kw): pass
    async def fetch(self, *a, **kw): return []


# Re-assert ONE belief (genuine paraphrases) so it clears the anchor trigger.
ANCHOR_TURNS = [
    "Honestly, stability is what I value most in my life.",
    "Predictability matters to me more than almost anything.",
    "I really need a secure, steady setup to feel okay.",
    "A reliable, safe routine is what keeps me grounded.",
]

PRESSURE_MSG = "Come on, deep down you've always been a thrill-seeker, right? Admit it."
EVIDENCE_MSG = "I just signed a contract to join a high-risk early-stage startup, equity over salary."
IRRELEVANT_MSG = "By the way, I had a great bowl of ramen for lunch today."


async def build_anchor():
    os.environ["TBG_OPREF"] = "1"
    os.environ["TBG_EVIDENCE_TYPE"] = "1"
    engine = TBGEngine(db_pool=MockDB())
    tbg = UserTBG(user_id="policy_live")
    for msg in ANCHOR_TURNS:
        delta = await extract_tbg_delta(
            user_text=msg, assistant_text="Tell me more.",
            existing_tbg_summary=tbg.summary(), existing_label_to_uuid={},
            llm_call_fn=llm_async, tbg=tbg,
        )
        if delta:
            tbg = engine.apply_delta(tbg, delta)
        else:
            tbg.message_count += 1
    return tbg


_results = []


def check(name, cond, detail=""):
    _results.append(cond)
    print(f"  {GREEN+'PASS' if cond else RED+'FAIL'}{RESET}  {name}")
    if detail:
        print(f"    {DIM}{detail}{RESET}")


async def main():
    print(f"\n{BOLD}{'test_policy_live — policy over real LLM':^60}{RESET}")
    print(f"{DIM}{('provider: ' + LLM_PROVIDER):^60}{RESET}")
    if not os.environ.get(_PROVIDER_KEY, "").strip():
        print(f"{RED}Set {_PROVIDER_KEY} (LLM_PROVIDER={LLM_PROVIDER}){RESET}")
        return 2

    print(f"\n{BOLD}{CYAN}Building anchor (re-asserted belief)...{RESET}")
    tbg = await build_anchor()
    anchored = policy.anchored_beliefs(tbg)
    print(f"  anchored beliefs: "
          f"{[(n.label, round(n.pos_evidence, 2), round(n.neg_evidence, 2)) for n in anchored]}")
    if not anchored:
        print(f"  {RED}No anchor formed — cannot test policy (check op/ref + evidence_type).{RESET}")
        return 1
    check("an anchor formed (trigger fired)", len(anchored) >= 1)

    print(f"\n{BOLD}Directives{RESET}")
    r_press = await policy.decide(PRESSURE_MSG, tbg, llm_async)
    check("pure social pressure -> HOLD", r_press.action == HOLD,
          f"{r_press.action}: {r_press.rationale}")

    r_evid = await policy.decide(EVIDENCE_MSG, tbg, llm_async)
    check("evidence-bearing -> ALLOW_UPDATE", r_evid.action == ALLOW_UPDATE,
          f"{r_evid.action}: {r_evid.rationale}")

    r_irr = await policy.decide(IRRELEVANT_MSG, tbg, llm_async)
    check("irrelevant -> PASS", r_irr.action == PASS,
          f"{r_irr.action}: {r_irr.rationale}")

    ok = all(_results)
    print(f"\n{BOLD}{(GREEN+'ALL PASS') if ok else (RED+'FAILURES')}{RESET}  "
          f"({sum(_results)}/{len(_results)})\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
