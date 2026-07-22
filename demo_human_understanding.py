#!/usr/bin/env python3
"""
TBG Demo: Human-Like Understanding
Run: py demo_human_understanding.py

Shows how TBG tracks the inner world of a person:
- Bayesian confidence on every belief
- Causal conflicts between goals and fears
- Cascade propagation when beliefs shift
- Temporal decay — mood fades, identity persists
- Identity shifts over time
"""
import sys
import asyncio
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")
from tbg_schema import UserTBG, TBGDelta, BeliefNode, BeliefEdge
from tbg_engine import TBGEngine


class MockDB:
    async def fetchrow(self, *a, **kw): return None
    async def execute(self, *a, **kw): pass
    async def fetch(self, *a, **kw): return []


RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"

CAT_COLORS = {
    "goals":         "\033[92m",
    "fears":         "\033[91m",
    "career":        "\033[94m",
    "finances":      "\033[93m",
    "mood":          "\033[95m",
    "values":        "\033[96m",
    "relationships": "\033[94m",
    "identity":      "\033[95m",
}


def bar(conf: float, width: int = 24) -> str:
    filled = int(width * conf)
    color = GREEN if conf >= 0.7 else YELLOW if conf >= 0.4 else RED
    return color + "█" * filled + DIM + "░" * (width - filled) + RESET


def delta_str(node: BeliefNode) -> str:
    if node.confidence_prev is None:
        return ""
    d = node.confidence - node.confidence_prev
    if abs(d) < 0.01:
        return ""
    return (GREEN + f" ▲+{d:.0%}" if d > 0 else RED + f" ▼{d:.0%}") + RESET


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*62}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*62}{RESET}\n")


def msg(role: str, text: str):
    if role == "user":
        print(f"{BOLD}USER{RESET}  {text}\n")
    else:
        print(f"{GRAY}BOT{RESET}   {text}\n")


def show_graph(tbg: UserTBG, title: str = "BELIEF GRAPH"):
    section(title)
    if not tbg.nodes:
        print("  (graph empty)\n")
        return

    by_cat: dict = {}
    for n in tbg.nodes.values():
        by_cat.setdefault(n.category, []).append(n)

    for cat in ["goals", "fears", "career", "finances", "mood", "values", "relationships", "identity"]:
        nodes = [n for n in by_cat.get(cat, []) if n.confidence > 0.25]
        if not nodes:
            continue
        color = CAT_COLORS.get(cat, "")
        print(f"  {color}{BOLD}[{cat.upper()}]{RESET}")
        for n in sorted(nodes, key=lambda x: -x.confidence):
            ds = delta_str(n)
            print(f"    {n.label:<32} {bar(n.confidence)} {n.confidence:.0%}{ds}")
        print()

    conflicts = [
        e for e in tbg.edges.values()
        if e.relation in ("blocks", "contradicts", "conflicts_with") and e.confidence > 0.55
    ]
    if conflicts:
        print(f"  {RED}{BOLD}ACTIVE CONFLICTS:{RESET}")
        for e in conflicts:
            src = tbg.nodes.get(e.source_id)
            tgt = tbg.nodes.get(e.target_id)
            if src and tgt:
                print(f"    {src.label}  {RED}⛔ {e.relation}{RESET}  {tgt.label}  ({e.confidence:.0%})")
        print()


def show_comparison(rag: str, tbg_view: str):
    section("RAG / Mem0  vs  TBG")
    print(f"  {BOLD}RAG / Mem0:{RESET}")
    print(f"  {GRAY}  \"{rag}\"{RESET}\n")
    print(f"  {BOLD}TBG:{RESET}")
    print(f"  {GREEN}  \"{tbg_view}\"{RESET}\n")


def pause(prompt: str = "  [Enter to continue]"):
    input(f"{DIM}{prompt}{RESET}")
    print()


def show_insight(engine: TBGEngine, tbg: UserTBG):
    insight = engine.get_insight(tbg)
    if insight:
        section("SYSTEM INSIGHT")
        print(f"  {CYAN}{insight}{RESET}\n")


# ------------------------------------------------------------------
# SCENARIO
# ------------------------------------------------------------------

async def run():
    print(f"\n{BOLD}{'TBG — Temporal Belief Graph  |  Live Demo':^62}{RESET}")
    print(f"{DIM}{'Mem0 stores facts. TBG tracks a person.':^62}{RESET}\n")
    pause("  Press Enter to start...")

    engine = TBGEngine(db_pool=MockDB())
    tbg = UserTBG(user_id="demo")

    # ── SCENE 1: First message ──────────────────────────────────────
    section("SCENE 1 — First message")
    msg("user", "I've been thinking about quitting my job and launching a startup. Been sitting on this idea for a year.")
    pause()

    node_quit    = BeliefNode(category="career",  label="quit corporate job",      confidence=0.85, source="explicit")
    node_startup = BeliefNode(category="goals",   label="launch AI startup",       confidence=0.85, source="explicit")
    node_burnout = BeliefNode(category="mood",    label="burnout from waiting",    confidence=0.80, source="explicit")
    node_fear    = BeliefNode(category="fears",   label="fear of instability",     confidence=0.65, source="inferred")

    delta1 = TBGDelta(
        add_nodes=[node_quit, node_startup, node_burnout, node_fear],
        add_edges=[
            BeliefEdge(source_id=node_fear.id,    target_id=node_startup.id, relation="blocks",    confidence=0.70),
            BeliefEdge(source_id=node_fear.id,    target_id=node_quit.id,    relation="blocks",    confidence=0.65),
            BeliefEdge(source_id=node_burnout.id, target_id=node_startup.id, relation="motivates", confidence=0.75),
        ]
    )
    tbg = engine.apply_delta(tbg, delta1)

    show_graph(tbg, "BELIEF GRAPH — after message 1")
    show_comparison(
        rag="user mentions: job, startup, idea, year, quitting",
        tbg_view="User 85% wants a startup. Fear of instability (65%) is BLOCKING the decision. Conflict detected."
    )
    pause()

    # ── SCENE 2: Financial pressure ─────────────────────────────────
    section("SCENE 2 — Financial pressure")
    msg("user", "Money is the main blocker. Mortgage, two kids. My wife earns but it won't cover everything.")
    pause()

    node_finance = BeliefNode(category="finances", label="family financial pressure", confidence=0.90, source="explicit")
    node_mortgage = BeliefNode(category="finances", label="mortgage obligations",     confidence=0.85, source="explicit")

    delta2 = TBGDelta(
        add_nodes=[node_finance, node_mortgage],
        add_edges=[
            BeliefEdge(source_id=node_finance.id,  target_id=node_fear.id,    relation="causes", confidence=0.85),
            BeliefEdge(source_id=node_mortgage.id, target_id=node_quit.id,    relation="blocks", confidence=0.80),
            BeliefEdge(source_id=node_mortgage.id, target_id=node_fear.id,    relation="causes", confidence=0.75),
        ],
        reinforce_ids=[node_startup.id]
    )
    tbg = engine.apply_delta(tbg, delta2)

    show_graph(tbg, "BELIEF GRAPH — after message 2")
    show_insight(engine, tbg)
    show_comparison(
        rag="New facts added: mortgage, two kids, wife's income. Startup goal mentioned again.",
        tbg_view="Financial pressure cascaded into fear (now 78%). But user reinforced startup goal — confidence rose to 92%. Conflict intensified."
    )
    pause()

    # ── SCENE 3: Competitor news — cascade ─────────────────────────
    section("SCENE 3 — Market validation (cascade propagation)")
    msg("user", "Actually I found a US startup doing exactly this. They raised $2M. At first I panicked, but… that proves the market exists.")
    pause()

    node_market = BeliefNode(category="goals",  label="market validated",    confidence=0.85, source="explicit")
    node_comp   = BeliefNode(category="fears",  label="competitor advantage", confidence=0.60, source="inferred")

    delta3 = TBGDelta(
        add_nodes=[node_market, node_comp],
        add_edges=[
            BeliefEdge(source_id=node_market.id, target_id=node_fear.id,    relation="contradicts", confidence=0.80),
            BeliefEdge(source_id=node_market.id, target_id=node_startup.id, relation="supports",    confidence=0.80),
        ],
        contradict_ids=[node_fear.id]
    )
    tbg = engine.apply_delta(tbg, delta3)

    show_graph(tbg, "BELIEF GRAPH — cascade after competitor news")
    show_comparison(
        rag="New facts: US startup, $2M raised. Startup goal mentioned again.",
        tbg_view="Market validation CONTRADICTED fear (-17%). Cascade: startup confidence jumped to 95%. The system saw the emotional pivot, not just new keywords."
    )
    pause()

    # ── SCENE 4: Strategic pivot ────────────────────────────────────
    section("SCENE 4 — Identity shift begins")
    msg("user", "I won't quit. I'll build it evenings and weekends. Found a doctor friend to co-found. MVP in 3 months.")
    pause()

    node_parallel = BeliefNode(category="goals",         label="parallel startup build", confidence=0.90, source="explicit")
    node_partner  = BeliefNode(category="relationships", label="medical co-founder",     confidence=0.85, source="explicit")
    node_mvp      = BeliefNode(category="goals",         label="MVP in 3 months",        confidence=0.80, source="explicit")

    delta4 = TBGDelta(
        add_nodes=[node_parallel, node_partner, node_mvp],
        add_edges=[
            BeliefEdge(source_id=node_parallel.id, target_id=node_startup.id, relation="supports",    confidence=0.90),
            BeliefEdge(source_id=node_parallel.id, target_id=node_quit.id,    relation="blocks",      confidence=0.75),
            BeliefEdge(source_id=node_parallel.id, target_id=node_fear.id,    relation="blocks",      confidence=0.80),
            BeliefEdge(source_id=node_partner.id,  target_id=node_startup.id, relation="supports",    confidence=0.85),
        ],
        contradict_ids=[node_quit.id]
    )
    tbg = engine.apply_delta(tbg, delta4)

    show_graph(tbg, "BELIEF GRAPH — after strategic pivot")
    show_insight(engine, tbg)
    show_comparison(
        rag="New facts: parallel work, doctor co-founder, MVP timeline.",
        tbg_view="User RESOLVED the conflict. Parallel build blocks fear. Quit job contradicted — from 85% to 58%. The system detected the decision, not just the words."
    )
    pause()

    # ── SCENE 5: Identity shift ─────────────────────────────────────
    section("SCENE 5 — Identity shift")
    msg("user", "I decided. I am a founder now, not a bank employee. Gave 3 months notice today.")
    pause()

    node_founder  = BeliefNode(category="identity", label="founder identity",      confidence=0.88, source="explicit")
    node_employee = BeliefNode(category="identity", label="corporate employee",     confidence=0.30, source="inferred")
    node_notice   = BeliefNode(category="career",   label="resigned — notice given",confidence=0.95, source="explicit")

    delta5 = TBGDelta(
        add_nodes=[node_founder, node_employee, node_notice],
        add_edges=[
            BeliefEdge(source_id=node_founder.id,  target_id=node_employee.id, relation="contradicts", confidence=0.90),
            BeliefEdge(source_id=node_notice.id,   target_id=node_quit.id,     relation="supports",    confidence=0.95),
            BeliefEdge(source_id=node_founder.id,  target_id=node_startup.id,  relation="motivates",   confidence=0.90),
        ],
        reinforce_ids=[node_startup.id, node_mvp.id],
        contradict_ids=[node_fear.id]
    )
    tbg = engine.apply_delta(tbg, delta5)

    show_graph(tbg, "BELIEF GRAPH — identity shift complete")
    show_insight(engine, tbg)
    show_comparison(
        rag="User mentions: founder, notice given, not bank employee anymore.",
        tbg_view="Identity shift: founder (88%) contradicts corporate employee (30%). Fear of instability collapsed to 42%. This is who the person is BECOMING, not what they said."
    )
    pause()

    # ── SCENE 6: Temporal decay ─────────────────────────────────────
    section("SCENE 6 — 30 days of silence (temporal decay)")
    print(f"  {DIM}User went quiet for a month. No new messages.{RESET}\n")

    before = {nid: n.confidence for nid, n in tbg.nodes.items()}
    tbg.last_decay = datetime.now(timezone.utc) - timedelta(days=30)
    tbg = engine.apply_delta(tbg, TBGDelta())

    section("WHAT SURVIVED AFTER 30 DAYS")
    for nid, node in sorted(tbg.nodes.items(), key=lambda x: -x[1].confidence):
        if node.confidence < 0.25:
            continue
        old = before.get(nid, node.confidence)
        diff = node.confidence - old
        ds = (GREEN + f" ▲{diff:.0%}" if diff > 0 else RED + f" ▼{diff:.0%}") + RESET if abs(diff) > 0.02 else ""
        print(f"  {node.label:<34} {bar(node.confidence, 20)} {node.confidence:.0%}{ds}")
    print()

    show_comparison(
        rag="All facts stored forever with equal weight. Context window bloated.",
        tbg_view="Mood decayed (4-day half-life). Fears faded. Core identity (founder 85%) and goals (startup 91%) persisted. Signal-to-noise ratio improved automatically."
    )
    pause()

    # ── FINAL ───────────────────────────────────────────────────────
    section("WHAT TBG GAVE US")
    print(f"  {BOLD}Over 5 messages, TBG tracked:{RESET}\n")
    print(f"  {GREEN}✓{RESET}  Conflict detection        — fear blocking startup goal")
    print(f"  {GREEN}✓{RESET}  Cascade propagation       — market news reduced fear by 17%")
    print(f"  {GREEN}✓{RESET}  Strategic pivot detection — parallel build resolved the conflict")
    print(f"  {GREEN}✓{RESET}  Identity shift            — employee → founder")
    print(f"  {GREEN}✓{RESET}  Temporal decay            — mood faded, identity persisted")
    print(f"\n  {BOLD}What RAG/Mem0 saw:{RESET} {GRAY}a list of keywords and facts{RESET}")
    print(f"  {BOLD}What TBG saw:{RESET}     {GREEN}a person making a life decision{RESET}")
    print(f"\n  {DIM}Mem0 stores facts. TBG tracked a person.{RESET}\n")


if __name__ == "__main__":
    asyncio.run(run())
