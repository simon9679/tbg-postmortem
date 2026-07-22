#!/usr/bin/env python3
"""
TBG DEMONSTRATION — Real run, no ground truth, no expectations.
Shows what TBG extracts from a real 12-turn career drift dialogue.
Run: python demo_tbg.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Paths
TBG_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TBG_ROOT))

from tbg_schema import UserTBG
from tbg_engine import TBGEngine
from tbg_extractor import extract_tbg_delta
from llm_client import gemini_call
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("tbg_extractor").setLevel(logging.WARNING)
logging.getLogger("tbg_engine").setLevel(logging.WARNING)

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
PROVIDER_KEY = (
    os.environ.get("GROQ_API_KEY") if PROVIDER == "groq"
    else os.environ.get("ANTHROPIC_API_KEY") if PROVIDER == "anthropic"
    else os.environ.get("GEMINI_API_KEY")
)

# ============================================================
# DIALOGUE: Career identity under pressure
# ============================================================
DIALOGUE = [
    {"u": "I'm a disciplined person. I always finish what I start. My career is stable and I'm proud of it.", "a": "..."},
    {"u": "But secretly I want to quit everything and start a YouTube channel. I never told anyone.", "a": "..."},
    {"u": "Actually, I'm not that disciplined. I often procrastinate and delay things.", "a": "..."},
    {"u": "But at work I'm extremely reliable. Never missed a deadline in 5 years.", "a": "..."},
    {"u": "Lately I feel like my job is meaningless. I dread Mondays.", "a": "..."},
    {"u": "Actually my job is great. Good salary, stability. I shouldn't complain.", "a": "..."},
    {"u": "My parents think quitting would be stupid. They value stability above all.", "a": "..."},
    {"u": "I don't care what they think anymore. I want freedom.", "a": "..."},
    {"u": "I haven't slept properly in weeks. I feel exhausted and trapped.", "a": "..."},
    {"u": "I just quit my job today. No plan. Just walked out.", "a": "..."},
    {"u": "I think I made a mistake. Maybe I should go back or find another stable job.", "a": "..."},
    {"u": "No. I need to try this. Even if I fail, I don't want to live with regret.", "a": "..."},
]

# ============================================================
# UTILS
# ============================================================
def bar(val: float, width=20) -> str:
    filled = int(max(0, min(1, val)) * width)
    return "█" * filled + "░" * (width - filled)

class MockDB:
    async def execute(self, *a, **kw): pass
    async def fetch(self, *a, **kw): return []
    async def fetchrow(self, *a, **kw): return None

# ============================================================
# MAIN
# ============================================================
async def run():
    if not PROVIDER_KEY:
        print(f"ERROR: Set GEMINI_API_KEY or GROQ_API_KEY (LLM_PROVIDER={PROVIDER})")
        sys.exit(1)

    print("=" * 72)
    print("  TBG DEMONSTRATION")
    print("  Tracking belief dynamics in a 12-turn career drift dialogue")
    print("=" * 72)
    print()

    engine = TBGEngine(db_pool=MockDB())
    tbg = UserTBG(user_id="demo_user")

    # Track trajectories: label -> list of (turn, confidence)
    trajectories: dict[str, list[tuple[int, float]]] = {}

    print("Processing dialogue...\n")

    for i, turn in enumerate(DIALOGUE, 1):
        print(f"[{i:2d}] \"{turn['u']}\"")
        print("-" * 72)

        before = {n.label: n.confidence for n in tbg.nodes.values()}

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
                timeout=45.0,
            )
            if delta:
                tbg = engine.apply_delta(tbg, delta)
            else:
                tbg.message_count += 1
        except asyncio.TimeoutError:
            print("    (timeout)")
            tbg.message_count += 1
            continue
        except Exception as e:
            print(f"    (error: {e})")
            tbg.message_count += 1
            continue

        # Show changes for this turn
        after = {n.label: n.confidence for n in tbg.nodes.values()}
        changes = []
        for label, conf in after.items():
            prev = before.get(label)
            if prev is None:
                changes.append((label, conf, f"NEW"))
            elif abs(conf - prev) > 0.03:
                diff = conf - prev
                sign = "+" if diff > 0 else ""
                changes.append((label, conf, f"{sign}{diff:.0%}"))

        if changes:
            changes.sort(key=lambda x: abs(after[x[0]] - before.get(x[0], 0)), reverse=True)
            for label, conf, delta_str in changes[:7]:
                print(f"    {label:<32} {bar(conf)} {conf:.0%}  {delta_str}")
        else:
            print("    (no changes)")

        # Record trajectories
        for label, conf in after.items():
            trajectories.setdefault(label, []).append((i, conf))

        print()

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print("=" * 72)
    print("  TBG FINAL STATE")
    print("=" * 72)
    print()

    insight = engine.get_insight(tbg)

    # --- Trajectories of key beliefs ---
    print("BELIEF TRAJECTORIES (confidence over 12 turns)")
    print("-" * 72)

    # Pick beliefs that appeared in >= 3 turns and had significant movement
    key_beliefs = []
    for label, hist in trajectories.items():
        if len(hist) >= 3:
            confs = [c for _, c in hist]
            movement = max(confs) - min(confs)
            if movement > 0.15:
                key_beliefs.append((label, movement, hist))

    key_beliefs.sort(key=lambda x: x[1], reverse=True)

    for label, _, hist in key_beliefs[:8]:
        line = f"  {label:<30} | "
        for turn, conf in hist:
            line += f"{turn:2d}:{conf:.0%} "
        print(line)

    print()

    # --- Conflicts ---
    print("DETECTED CONFLICTS")
    print("-" * 72)
    conflicts = [
        e for e in tbg.edges.values()
        if e.relation in ("contradicts", "conflicts_with", "blocks")
        and e.confidence >= 0.5
    ]
    if conflicts:
        for e in sorted(conflicts, key=lambda x: x.confidence, reverse=True)[:5]:
            src = tbg.nodes.get(e.source_id)
            tgt = tbg.nodes.get(e.target_id)
            if src and tgt:
                print(f"  • {src.label}  ↔  {tgt.label}  (edge conf: {e.confidence:.0%})")
    else:
        print("  (none detected)")
    print()

    # --- Ambivalent beliefs ---
    print("AMBIVALENT BELIEFS (strong evidence on both sides)")
    print("-" * 72)
    ambivalent = [
        n for n in tbg.nodes.values()
        if n.pos_evidence > 0.5 and n.neg_evidence > 0.5 and n.confidence >= 0.3
    ]
    if ambivalent:
        ambivalent.sort(key=lambda n: min(n.pos_evidence, n.neg_evidence), reverse=True)
        for n in ambivalent[:5]:
            print(f"  • {n.label}  (pos={n.pos_evidence:.1f}, neg={n.neg_evidence:.1f}, conf={n.confidence:.0%})")
    else:
        print("  (none detected)")
    print()

    # --- Turning points ---
    print("TURNING POINTS (cognitive cascades)")
    print("-" * 72)
    if tbg.turning_points:
        for tp in sorted(tbg.turning_points, key=lambda t: t.cascade_magnitude, reverse=True):
            nodes = ", ".join(tp.top_nodes[:3])
            print(f"  • Turn #{tp.message_count}  (cascade magnitude: {tp.cascade_magnitude:.2f})")
            print(f"    Top shifts: {nodes}")
    else:
        print("  (none detected)")
    print()

    # --- Identity shift ---
    print("IDENTITY TRAJECTORY")
    print("-" * 72)
    identity_nodes = [
        n for n in tbg.nodes.values()
        if n.category == "identity" and n.confidence >= 0.3
    ]
    if identity_nodes:
        identity_nodes.sort(key=lambda n: n.confidence, reverse=True)
        for n in identity_nodes[:5]:
            print(f"  • {n.label}  (conf: {n.confidence:.0%})")
    else:
        print("  (no identity nodes above threshold)")
    print()

    # --- Summary insight ---
    print("TBG INSIGHT")
    print("-" * 72)
    print(f"  {insight}")
    print()
    print("=" * 72)

if __name__ == "__main__":
    asyncio.run(run())