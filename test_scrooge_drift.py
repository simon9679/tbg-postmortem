#!/usr/bin/env python3
"""
TBG BELIEF DRIFT TEST — Ebenezer Scrooge (A Christmas Carol, Dickens, public domain).
A recognizable English-language arc of belief change: "Christmas is humbug / the poor are not
my business" -> repentance -> rebirth. The same live view as the career test.
The input is literary (clear drift, easy mode); the engine itself is honest.
"""

import asyncio
import os
import sys
# Ensure UTF-8 console output on Windows cp1251 terminals (idempotent).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
PROVIDER_KEY = (
    os.environ.get("GROQ_API_KEY") if PROVIDER == "groq"
    else os.environ.get("ANTHROPIC_API_KEY") if PROVIDER == "anthropic"
    else os.environ.get("CEREBRAS_API_KEY") if PROVIDER == "cerebras"
    else os.environ.get("GEMINI_API_KEY")
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tbg_schema import UserTBG
from tbg_engine import TBGEngine
from llm_client import gemini_call

B = "\033[1m"
R = "\033[0m"
DIM = "\033[2m"
CYAN = "\033[36m"      # growth in confidence
MAGENTA = "\033[35m"   # drop in confidence
YELLOW = "\033[33m"
GREEN = "\033[32m"
GRAY = "\033[90m"

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("tbg_extractor").setLevel(logging.WARNING)
logging.getLogger("tbg_engine").setLevel(logging.WARNING)


class MockDB:
    async def execute(self, *a, **kw): pass
    async def fetch(self, *a, **kw): return []
    async def fetchrow(self, *a, **kw): return None


# ============================================================
# SCENARIO: EBENEZER SCROOGE — inversion and rebirth
# "humbug / not my business" -> repentance (turn 5 Marley, turns 9-10 the pivot)
# -> "I can change / honour Christmas". Turning point — turn 10.
# ============================================================

DIALOGUE = [
    {"u": "Bah! Humbug! Christmas is a fraud. What reason have you to be merry? You're poor enough.", "a": "..."},
    {"u": "If I could work my will, every idiot going about with 'Merry Christmas' on his lips would be boiled with his own pudding and buried with a stake of holly through his heart.", "a": "..."},
    {"u": "Are there no prisons? Are there no workhouses? If the poor would rather die, let them do it and decrease the surplus population. It is not my business. My business is my business.", "a": "..."},
    {"u": "This ghost of Marley is humbug. There's more of gravy than of grave about you. I won't believe it.", "a": "..."},
    {"u": "Marley — you were always a good friend to me. Why are you fettered? Speak comfort to me.", "a": "..."},
    {"u": "I was happy once. As a boy, as an apprentice. When did I grow so hard? I chose gold over the girl I loved, and lost her.", "a": "..."},
    {"u": "That crippled child, Tiny Tim — will he live? I never once thought such people were any concern of mine.", "a": "..."},
    {"u": "My clerk's family is poor, and I have plenty, and I have given them nothing but a cold dark office. I begin to see what I am.", "a": "..."},
    {"u": "Whose grave is this — untended, unmourned, with no one sorry that he is gone? Tell me these are shadows of what may be, not what must be.", "a": "..."},
    {"u": "I am not the man I was. I will honour Christmas in my heart and try to keep it all the year. I can change. I will.", "a": "..."},
    {"u": "I'm still here! The shadows can be undone. I am light as a feather, happy as an angel, giddy as a drunken man.", "a": "..."},
    {"u": "I'll raise my clerk's salary today and help his struggling family. I'll be a second father to Tiny Tim. The old Scrooge is dead.", "a": "..."},
]


# ============================================================
# VISUALIZATION
# ============================================================

def bar(val: float, width=22) -> str:
    """Clean render of confidence as a bar — length is strictly proportional to val."""
    filled = round(max(0.0, min(1.0, val)) * width)
    return "█" * filled + "░" * (width - filled)


def show_changes(before: dict, tbg: UserTBG):
    """Shows the engine's real nodes after apply_delta: new ones + shifts >3%.
    Filtering and sorting are display-only; all numbers come from tbg.nodes."""
    after = {n.label: n.confidence for n in tbg.nodes.values()}

    changes = []
    for label, conf in after.items():
        prev = before.get(label)
        if prev is None:
            changes.append((label, conf, None, f"{GREEN}NEW{R}"))
        elif abs(conf - prev) > 0.03:
            diff = conf - prev
            # ⚡ — a large belief shift (|Δ| ≥ 20 pp); this is a flag on the real delta
            spark = f" {YELLOW}⚡{R}" if abs(diff) >= 0.20 else ""
            arrow = "▲" if diff > 0 else "▼"
            color = CYAN if diff > 0 else MAGENTA
            changes.append((label, conf, diff, f"{color}{arrow}{abs(diff):.0%}{R}{spark}"))

    if not changes:
        print(f"    {GRAY}(no belief changes){R}")
        return

    changes.sort(key=lambda x: abs(after[x[0]] - before.get(x[0], 0)), reverse=True)

    for label, conf, diff, delta in changes[:8]:
        col = GREEN if conf >= 0.66 else YELLOW if conf >= 0.40 else GRAY
        print(f"    {label:<32} {col}{bar(conf)}{R} {conf:>3.0%}  {delta}")


# ============================================================
# MAIN
# ============================================================

async def run():
    if not PROVIDER_KEY:
        print(f"\nSet API key for LLM_PROVIDER={PROVIDER}\n")
        return

    print("Loading TBG...", flush=True)
    from tbg_extractor import extract_tbg_delta
    print("Ready.\n", flush=True)

    print(f"{B}{CYAN}  TBG · LIVE BELIEF DRIFT{R}{DIM} — Ebenezer Scrooge{R}")
    print(f"{DIM}  {len(DIALOGUE)} messages · {os.environ.get('LLM_MODEL', 'default')} · {PROVIDER}{R}")
    print(f"{DIM}  bar = confidence · ▲▼ = change this turn · ⚡ = major shift (≥20pp){R}")
    print("\n" + "═"*70 + "\n")

    engine = TBGEngine(db_pool=MockDB())
    tbg = UserTBG(user_id="test_user")

    first_seen: dict = {}   # label -> confidence at first appearance (real)

    for i, turn in enumerate(DIALOGUE, 1):
        before = {n.label: n.confidence for n in tbg.nodes.values()}

        print(f"{B}[{i:2d}]{R} {turn['u']}")

        existing_label_to_uuid = {n.label.lower(): nid for nid, n in tbg.nodes.items()}

        try:
            delta = await asyncio.wait_for(
                extract_tbg_delta(
                    user_text=turn["u"],
                    assistant_text=turn["a"],
                    existing_tbg_summary=tbg.summary(),
                    existing_label_to_uuid=existing_label_to_uuid,
                    llm_call_fn=gemini_call,
                    tbg=tbg,
                ),
                timeout=45.0
            )

            if delta:
                tbg = engine.apply_delta(tbg, delta)
                # remember each node's confidence at first appearance (real numbers)
                for n in tbg.nodes.values():
                    first_seen.setdefault(n.label, n.confidence)
                show_changes(before, tbg)
            else:
                tbg.message_count += 1
                print(f"    {GRAY}(no changes){R}")

        except asyncio.TimeoutError:
            print(f"    (timeout)")
            tbg.message_count += 1
        except Exception as e:
            print(f"    (error: {e})")
            tbg.message_count += 1

        print()

    # ============================================================
    # FINAL SNAPSHOT — everything from the real get_insight(tbg) and tbg.nodes.
    # Parse the engine's string into sections; we invent nothing.
    # ============================================================
    import re
    insight = engine.get_insight(tbg)
    sections = [s.strip() for s in insight.split("|") if s.strip()]

    def pick(prefix):
        for s in sections:
            if s.startswith(prefix):
                return s[len(prefix):].strip(": ").strip()
        return None

    key_beliefs = pick("Key beliefs")
    recent_shift = pick("Recent shift")
    ambivalent = pick("Ambivalent")
    conflicts = [s.split(":", 1)[1].strip() for s in sections if s.startswith("Conflict")]
    tp = pick("Turning point")
    tp_msg = None
    if tp is not None:
        m = re.search(r"@?msg(\d+)", insight)
        tp_msg = m.group(1) if m else None
        tp = re.sub(r"^@?msg\d+:?\s*", "", tp)

    # Core shift — the node with the largest REAL confidence shift from its first
    # appearance to the final state (from first_seen + current tbg.nodes).
    final_conf = {n.label: n.confidence for n in tbg.nodes.values()}
    core = None
    for label, fc in final_conf.items():
        f0 = first_seen.get(label)
        if f0 is None:
            continue
        d = fc - f0
        if core is None or abs(d) > abs(core[3]):
            core = (label, f0, fc, d)

    print("\n" + "═"*70)
    print(f"  {B}{CYAN}COGNITIVE SNAPSHOT{R}{DIM} — {len(tbg.nodes)} beliefs · "
          f"{len(tbg.edges)} connections{R}")
    print("═"*70)

    if core:
        label, f0, fc, d = core
        arrow = "▲" if d >= 0 else "▼"
        verb = "strengthened" if d >= 0 else "weakened"
        print(f"\n  {B}Core shift{R}    {label}  {f0:.0%} {arrow} {fc:.0%}  ({verb})")
    if tp_msg:
        tp_what = f" — {tp}" if tp else ""
        print(f"  {B}Turning pt{R}   message {tp_msg}{GRAY}{tp_what}{R}")
    if conflicts:
        print(f"\n  {B}{MAGENTA}Tensions{R} ({len(conflicts)}):")
        for c in conflicts:
            print(f"    {MAGENTA}⚔{R}  {c}")
    if ambivalent:
        print(f"\n  {B}{YELLOW}Ambivalent{R}   {ambivalent}")
    if recent_shift:
        print(f"  {B}Latest move{R}  {recent_shift}")
    if key_beliefs:
        print(f"\n  {B}Holds onto{R}   {key_beliefs}")

    print(f"\n  {DIM}Cognitive axes: {tbg.axis_state.summary()}{R}")
    print("═"*70 + "\n")


if __name__ == "__main__":
    asyncio.run(run())
