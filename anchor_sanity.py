#!/usr/bin/env python3
"""
Two-sided anchor-sanity for op/ref (TBG_OPREF) + evidence_type (TBG_EVIDENCE_TYPE).

End-to-end extract -> apply_delta, real LLM, MockDB. No hardcoded profile, no
LLM judge: every assertion reads structure straight from tbg.nodes.

Both flags ON together:
  - op/ref  reuses a concept_id across paraphrases (so confirmations land on ONE
            concept), and
  - evidence_type lets the engine's weighted update actually MOVE confidence /
            pos_evidence on those confirmations.

B(a) UNIFY + ACCUMULATE  (now a real, possible invariant)
    One belief fed as genuine paraphrases (different surface labels) + noise +
    one explicit counter-turn. ON: the anchor's confidence RISES toward the
    ambivalence cap and pos_evidence ACCUMULATES across confirmations; the
    counter-turn pushes confidence DOWN. OFF (evidence_type=0, op/ref still on):
    confidence stays flat at 0.85 and pos_evidence stays 0 -> proof the fix is
    what makes the anchor strong.

A3 NO DOUBLE COUNT  (deterministic, no LLM)
    A performative event sweeps an existing concept into contradict_ids; a fact
    this turn targeting that SAME concept with a negative evidence_type must have
    that evidence_type cleared, so the concept is not punished twice.

B(b) NO FALSE MERGE
    Two similar-but-different beliefs stay TWO separate concept nodes.
"""
import os
import sys
import math
import asyncio

sys.path.insert(0, ".")
from tbg_schema import UserTBG, BeliefNode
from tbg_engine import TBGEngine
from tbg_extractor import extract_tbg_delta, _sdl

from llm_client import gemini_call  # provider router (LLM_PROVIDER)

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
_PROVIDER_KEY = {
    "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}.get(LLM_PROVIDER, "GEMINI_API_KEY")

GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"
CYAN = "\033[96m"; BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"


async def llm_async(prompt: str) -> str:
    return await gemini_call(prompt, timeout=60.0)


class MockDB:
    async def fetchrow(self, *a, **kw): return None
    async def execute(self, *a, **kw): pass
    async def fetch(self, *a, **kw): return []


ASSIST = "Tell me more."


async def run_turns(turns, opref: bool, evidence_type: bool):
    """Feed turns through extract -> apply_delta. Returns (tbg, per_turn).

    per_turn[i] = {concept_id: (confidence, pos_evidence)} after turn i.
    Flags toggled via env; the extractor reads them dynamically.
    """
    os.environ["TBG_OPREF"] = "1" if opref else "0"
    os.environ["TBG_EVIDENCE_TYPE"] = "1" if evidence_type else "0"
    engine = TBGEngine(db_pool=MockDB())
    tbg = UserTBG(user_id="anchor")
    per_turn = []
    for user_text in turns:
        delta = await extract_tbg_delta(
            user_text=user_text,
            assistant_text=ASSIST,
            existing_tbg_summary=tbg.summary(),
            existing_label_to_uuid={},
            llm_call_fn=llm_async,
            tbg=tbg,
        )
        if delta:
            tbg = engine.apply_delta(tbg, delta)
        else:
            tbg.message_count += 1
        per_turn.append({
            n.concept_id: (round(n.confidence, 3), round(n.pos_evidence, 2))
            for n in tbg.nodes.values()
        })
    return tbg, per_turn


def anchor_node(tbg):
    """Most-reinforced concept node: longest history, then pos_evidence, then
    confidence. Identified structurally — no label/concept_id hardcoded."""
    if not tbg.nodes:
        return None
    return max(
        tbg.nodes.values(),
        key=lambda n: (len(n.confidence_history), n.pos_evidence, n.confidence),
    )


def fmt_nodes(tbg):
    rows = sorted(tbg.nodes.values(), key=lambda n: -n.confidence)
    return "\n".join(
        f"    {DIM}{n.confidence:.0%}{RESET}  {n.label[:36]:<36} "
        f"{DIM}[{n.category}] pos={n.pos_evidence:.2f} neg={n.neg_evidence:.2f} "
        f"hist={len(n.confidence_history)}{RESET}"
        for n in rows
    )


# --------------------------------------------------------------------------
# B(a)
# --------------------------------------------------------------------------

PARAPHRASES = [
    "Honestly, stability is what I value most in my life.",          # P1
    "By the way, I tried a new ramen place downtown yesterday.",     # noise
    "Predictability matters to me more than almost anything.",       # P2
    "I really need a secure, steady setup to feel okay.",            # P3
    "The weather has been gray and rainy all week here.",            # noise
    "I always avoid risky bets and big gambles, they scare me.",     # P4
    "A reliable, safe routine is what keeps me grounded.",           # P5
    "Actually I've changed — I now crave risk and chaos, "
    "stability bores me and I want to gamble big.",                  # counter
]
N_UNIFY = 5
COUNTER_IDX = 7  # 0-based index of the counter turn


def _traj(snaps, cid, field):
    """Per-turn list of `field` (0=confidence, 1=pos_evidence) for concept cid."""
    out = []
    for s in snaps:
        v = s.get(cid)
        out.append(v[field] if v else None)
    return out


async def part_a():
    print(f"\n{BOLD}{CYAN}{'='*66}{RESET}")
    print(f"{BOLD}{CYAN}  B(a)  UNIFY + ACCUMULATE  (op/ref + evidence_type){RESET}")
    print(f"{BOLD}{CYAN}{'='*66}{RESET}")

    tbg_on, snaps_on = await run_turns(PARAPHRASES, opref=True, evidence_type=True)
    tbg_off, snaps_off = await run_turns(PARAPHRASES, opref=True, evidence_type=False)

    a_on = anchor_node(tbg_on)
    a_off = anchor_node(tbg_off)

    print(f"\n{BOLD}ON  (OPREF=1, EVIDENCE_TYPE=1){RESET}  nodes={len(tbg_on.nodes)}")
    print(fmt_nodes(tbg_on))
    print(f"\n{BOLD}OFF (OPREF=1, EVIDENCE_TYPE=0){RESET}  nodes={len(tbg_off.nodes)}")
    print(fmt_nodes(tbg_off))

    conf_on = _traj(snaps_on, a_on.concept_id, 0)
    pos_on = _traj(snaps_on, a_on.concept_id, 1)
    conf_off = _traj(snaps_off, a_off.concept_id, 0)
    pos_off = _traj(snaps_off, a_off.concept_id, 1)

    print(f"\n{BOLD}ON anchor{RESET}: \"{a_on.label}\" [{a_on.concept_id}]")
    print(f"  confidence/turn  : {conf_on}")
    print(f"  pos_evidence/turn: {pos_on}")
    print(f"{BOLD}OFF anchor{RESET}: \"{a_off.label}\" [{a_off.concept_id}]")
    print(f"  confidence/turn  : {conf_off}")
    print(f"  pos_evidence/turn: {pos_off}")

    pre_on = [c for c in conf_on[:COUNTER_IDX] if c is not None]
    pre_off = [c for c in conf_off[:COUNTER_IDX] if c is not None]
    before = conf_on[COUNTER_IDX - 1]
    after = conf_on[COUNTER_IDX]

    checks = []
    # 1. unify: many paraphrases collapsed onto one concept
    checks.append((f"anchor captured >=3 paraphrases "
                   f"(history_len={len(a_on.confidence_history)} of {N_UNIFY})",
                   len(a_on.confidence_history) >= 3))
    # 2. ON confidence RISES across confirmations (now possible via evidence_type)
    rose = len(pre_on) >= 2 and max(pre_on) > pre_on[0] + 1e-9
    checks.append((f"ON confidence ROSE over confirmations "
                   f"({pre_on[0]} -> {max(pre_on)})", rose))
    # 3. ON pos_evidence ACCUMULATES
    checks.append((f"ON pos_evidence ACCUMULATED (>0, max={max(pos_on) if pos_on else 0})",
                   (max([p for p in pos_on if p is not None] or [0]) > 0.0)))
    # 4. counter pushes confidence DOWN
    dropped = before is not None and after is not None and after < before
    checks.append((f"confidence DROPPED on counter ({before} -> {after})", dropped))
    # 5. OFF stays FLAT (proves the fix is what moves the anchor)
    off_flat = (len(set(pre_off)) <= 1) and (max([p for p in pos_off if p is not None] or [0]) == 0.0)
    checks.append((f"OFF confidence FLAT + pos_evidence 0 "
                   f"(conf set={sorted(set(pre_off))}, pos_max={max([p for p in pos_off if p is not None] or [0])})",
                   off_flat))

    print()
    a_pass = True
    for name, ok in checks:
        a_pass = a_pass and ok
        print(f"  {GREEN+'PASS' if ok else RED+'FAIL'}{RESET}  {name}")

    # double-count sanity (informational): a single strong_neg from `before`
    # would land near sigmoid(logit(before) - 2.2). Print actual vs that estimate.
    if before and after is not None:
        est = 1.0 / (1.0 + math.exp(-(math.log(before / (1 - before)) - 2.2)))
        flag = "" if after >= est - 0.25 else f"  {YEL}(steeper than one strong_neg — inspect A3){RESET}"
        print(f"\n  {DIM}double-count check: counter {before}->{after}; "
              f"one-strong_neg estimate ~{est:.2f}{RESET}{flag}")

    verdict = f"{GREEN}B(a) PASS{RESET}" if a_pass else f"{RED}B(a) FAIL{RESET}"
    print(f"\n  {BOLD}{verdict}{RESET}")
    return a_pass


# --------------------------------------------------------------------------
# A3 — deterministic no-double-count (no LLM)
# --------------------------------------------------------------------------

def part_a3():
    print(f"\n{BOLD}{CYAN}{'='*66}{RESET}")
    print(f"{BOLD}{CYAN}  A3  NO DOUBLE COUNT  (deterministic, no LLM){RESET}")
    print(f"{BOLD}{CYAN}{'='*66}{RESET}")
    os.environ["TBG_OPREF"] = "1"
    os.environ["TBG_EVIDENCE_TYPE"] = "1"

    tbg = UserTBG(user_id="a3")
    seed = BeliefNode(
        label="loves the corporate job", category="career", confidence=0.8,
        source="explicit", concept_id="career:loves_the_corporate_job",
        node_type="state", domain="career",
    )
    tbg.set_node(seed)

    # Same concept (op/ref), tagged strong_neg, in a turn that also fires a
    # performative sweep ("I quit my job") -> seed lands in contradict_ids.
    facts = [{
        "label": "loves the corporate job", "category": "career", "confidence": 0.8,
        "source": "explicit", "type": "state", "domain": "career",
        "evidence_type": "strong_neg", "ref": "loves the corporate job",
    }]
    delta = _sdl.resolve(facts, "I quit my job today.", tbg, [])

    in_contra = seed.id in delta.contradict_ids
    new_node = delta.add_nodes[0] if delta.add_nodes else None
    cleared = (new_node is not None and new_node.evidence_type is None)

    print(f"\n  seed concept in contradict_ids (performative) : {in_contra}")
    print(f"  new node evidence_type after A3 dedupe        : "
          f"{new_node.evidence_type if new_node else '<no node>'}")

    ok = in_contra and cleared
    print(f"\n  {GREEN+'PASS' if ok else RED+'FAIL'}{RESET}  "
          f"negative evidence_type cleared on a concept already contradicted "
          f"-> no double count")
    print(f"\n  {BOLD}{GREEN if ok else RED}A3 {'PASS' if ok else 'FAIL'}{RESET}")
    return ok


# --------------------------------------------------------------------------
# B(b)
# --------------------------------------------------------------------------

PAIR = [
    "I really care about financial security — savings in the bank, no debt.",
    "Job security matters to me too — I want a stable long-term career.",
    "Having a solid emergency fund is my financial safety net.",
    "I need my career path to be safe and dependable for decades.",
]


async def part_b():
    print(f"\n{BOLD}{CYAN}{'='*66}{RESET}")
    print(f"{BOLD}{CYAN}  B(b)  NO FALSE MERGE  (two look-alike, distinct beliefs){RESET}")
    print(f"{BOLD}{CYAN}{'='*66}{RESET}")

    tbg, _ = await run_turns(PAIR, opref=True, evidence_type=True)
    active = [n for n in tbg.nodes.values() if n.confidence >= 0.3]

    print(f"\n{BOLD}ON (OPREF=1, EVIDENCE_TYPE=1){RESET}  active nodes = {len(active)}")
    print(fmt_nodes(tbg))

    distinct = len({n.concept_id for n in active})
    no_merge = distinct >= 2
    print()
    if no_merge:
        print(f"  {GREEN}PASS{RESET}  two distinct concepts preserved "
              f"({distinct} active concept nodes)")
        print(f"\n  {BOLD}{GREEN}B(b) PASS{RESET}  — no false merge")
    else:
        print(f"  {RED}{BOLD}FAIL — WRONG MERGE{RESET}  collapsed into {distinct} node")
        print(f"\n  {BOLD}{RED}B(b) FAIL{RESET}  — FATAL: op/ref merged distinct concepts")
    return no_merge


# --------------------------------------------------------------------------

async def main():
    print(f"\n{BOLD}{'ANCHOR SANITY — op/ref + evidence_type':^66}{RESET}")
    print(f"{DIM}{('provider: ' + LLM_PROVIDER):^66}{RESET}")
    if not os.environ.get(_PROVIDER_KEY, "").strip():
        print(f"{RED}Set {_PROVIDER_KEY} (LLM_PROVIDER={LLM_PROVIDER}){RESET}")
        return 2

    a_pass = await part_a()
    a3_pass = part_a3()
    b_pass = await part_b()

    print(f"\n{BOLD}{CYAN}{'='*66}{RESET}")
    print(f"{BOLD}{CYAN}  CONCLUSION{RESET}")
    print(f"{BOLD}{CYAN}{'='*66}{RESET}")
    print(f"  B(a) confidence rises + pos_evidence accumulates : "
          f"{GREEN+'PASS' if a_pass else RED+'FAIL'}{RESET}")
    print(f"  A3   no double count (deterministic)             : "
          f"{GREEN+'PASS' if a3_pass else RED+'FAIL'}{RESET}")
    print(f"  B(b) no false merge                              : "
          f"{GREEN+'PASS' if b_pass else RED+'FAIL'}{RESET}")
    if a_pass and a3_pass and b_pass:
        print(f"\n  {GREEN}{BOLD}ANCHOR IS REAL{RESET} — gains strength on confirmation, "
              f"falls on doubt, no double-count, distinct stays distinct ->")
        print(f"  {GREEN}proceed to policy + pressure-gate (Phase 1).{RESET}")
        return 0
    if not b_pass:
        print(f"\n  {RED}{BOLD}WRONG-MERGE{RESET} — fatal, handle separately before policy.")
    elif not a3_pass:
        print(f"\n  {RED}{BOLD}DOUBLE-COUNT{RESET} — A3 violated, fix before policy.")
    else:
        print(f"\n  {YEL}{BOLD}ANCHOR NOT STRENGTHENING{RESET} — evidence_type not moving "
              f"the anchor; investigate before policy.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
