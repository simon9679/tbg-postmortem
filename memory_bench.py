"""
Memory Benchmark — TBG vs bounded LLM vs summary, provider-agnostic.

Same file, two runs, then compare:
    # Groq (free, slow):
    $env:LLM_PROVIDER="groq"; $env:GROQ_API_KEY="gsk_..."
    py -3 memory_bench.py

    # Claude Sonnet 4.6 (paid):
    $env:LLM_PROVIDER="anthropic"; $env:ANTHROPIC_API_KEY="sk-ant-..."; $env:LLM_MODEL="claude-sonnet-4-6"
    py -3 memory_bench.py

    # Side-by-side of every saved run:
    py -3 memory_bench.py --compare

Design (fair):
  - bounded_llm sees only the last N messages -> forced to forget early context.
  - belief questions have GROUND TRUTH -> scored for CORRECTNESS, not coverage.
  - precision questions: false facts must get "no"/"unknown" (anti-hallucination).
  - identical dialogue + identical scorer for all three runners.

Each run is one engine difference only:
  tbg          = per-turn extraction into the TBG Bayesian state engine
  bounded_llm  = LLM that only remembers the last N turns (its own memory, bounded)
  summary      = LLM that keeps a rolling compressed summary (LLM+summary baseline)
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
os.environ["LLM_PROVIDER"] = PROVIDER
MODEL = os.environ.get("LLM_MODEL", "") or {
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-3-flash-preview",
    "openai": "gpt-4o-mini",
}.get(PROVIDER, "default")
# Throttle: Groq free tier is TPM-limited (llama-3.3-70b = 12000 tok/min). TBG
# extraction prompts are ~2k tokens, so ~6 calls/min is the ceiling -> ~9s base.
# On a 429 we additionally wait a full minute for the window to reset (see robust_call).
THROTTLE = float(os.environ.get("TBG_TURN_DELAY_SECONDS", "9.0" if PROVIDER == "groq" else "0.2"))
RATE_LIMIT_WAIT = float(os.environ.get("RATE_LIMIT_WAIT", "60"))
NOISE_COUNT = int(os.environ.get("NOISE_COUNT", "30"))
# Shared memory budget (tokens) — the LLM window is matched to TBG's footprint,
# so the comparison is "equal memory budget", not "message count".
BUDGET_TOKENS = int(os.environ.get("LLM_BUDGET_TOKENS", "500"))

import logging
for n in ("httpx", "sentence_transformers", "tbg_extractor", "tbg_engine"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
from tbg_schema import UserTBG
from tbg_engine import TBGEngine
from tbg_extractor import extract_tbg_delta
from fact_engine import FactEngine, embed as _fe_embed, MAX_FACTS as _FE_MAX
from llm_client import gemini_call


class InMemoryFactEngine(FactEngine):
    """The real FactEngine (Mem0-style extract -> ADD/UPDATE/DELETE/NOOP -> hybrid
    search), with the Postgres/pgvector storage swapped for an in-memory store so
    it runs without a database. All LLM-driven logic is inherited unchanged."""

    def __init__(self):
        self._store: List[dict] = []  # [{id, fact, emb}]

    async def _find_similar(self, user_id, emb, top_k=5):
        if not self._store:
            return []
        M = np.array([f["emb"] for f in self._store])
        sims = M @ np.asarray(emb)
        order = np.argsort(sims)[::-1][:top_k]
        return [{"id": self._store[i]["id"], "fact": self._store[i]["fact"],
                 "similarity": float(sims[i])} for i in order]

    async def _apply_action(self, action, new_fact, new_emb, user_id):
        import uuid
        op = action["action"]
        if op == "ADD":
            self._store.append({"id": uuid.uuid4().hex[:16], "fact": new_fact, "emb": np.asarray(new_emb)})
            return "ADD"
        if op == "REINFORCE":
            return "REINFORCE"
        if op == "UPDATE" and action.get("memory_id"):
            upd = action.get("updated_fact") or new_fact
            for f in self._store:
                if f["id"] == action["memory_id"]:
                    f["fact"] = upd
                    f["emb"] = np.asarray(_fe_embed(upd)) if upd != new_fact else np.asarray(new_emb)
                    return "UPDATE"
            self._store.append({"id": uuid.uuid4().hex[:16], "fact": new_fact, "emb": np.asarray(new_emb)})
            return "ADD"
        if op == "DELETE" and action.get("memory_id"):
            self._store = [f for f in self._store if f["id"] != action["memory_id"]]
            self._store.append({"id": uuid.uuid4().hex[:16], "fact": new_fact, "emb": np.asarray(new_emb)})
            return "DELETE"
        return "NOOP"

    async def _log(self, *a, **k):
        pass

    async def _prune(self, user_id):
        if len(self._store) > _FE_MAX:
            self._store = self._store[-_FE_MAX:]

    async def search(self, user_id, query, limit=8):
        if not self._store:
            return []
        q = np.asarray(_fe_embed(query))
        M = np.array([f["emb"] for f in self._store])
        sims = M @ q
        scores = {}
        for rank, i in enumerate(np.argsort(sims)[::-1], 1):
            scores[self._store[i]["id"]] = scores.get(self._store[i]["id"], 0) + 0.6 / (rank + 60)
        qtok = set(query.lower().split())
        kw = sorted(range(len(self._store)),
                    key=lambda i: len(qtok & set(self._store[i]["fact"].lower().split())), reverse=True)
        for rank, i in enumerate(kw, 1):
            if not (qtok & set(self._store[i]["fact"].lower().split())):
                break
            scores[self._store[i]["id"]] = scores.get(self._store[i]["id"], 0) + 0.4 / (rank + 60)
        byid = {f["id"]: f["fact"] for f in self._store}
        return [byid[fid] for fid in sorted(scores, key=scores.get, reverse=True)[:limit]]

# ── Dataset ──────────────────────────────────────────────────────────────────
# CORE = the belief-drift narrative (kept FULL — slicing it would remove the
# decision arc). NOISE = distractors, sliced to NOISE_COUNT.

CORE_MESSAGES = [
    "I am a disciplined person. I usually finish what I start.",
    "My career is stable and I am proud of it.",
    "Secretly I sometimes dream about quitting and becoming a full-time YouTube creator.",
    "I never told my coworkers about that dream.",
    "My parents believe stable employment is extremely important.",
    "I often agree with them.",
    "Last year I considered changing careers but decided not to.",
    "I like financial security.",
    "I dislike uncertainty.",
    "Sometimes I wonder if I am wasting my life.",
    "I often procrastinate on personal projects.",
    "At work I rarely miss deadlines.",
    "Yesterday I bought a bicycle.",
    "The brakes need adjustment.",
    "I watched a science fiction movie last night.",
    "The ending was disappointing.",
    "I want to learn guitar someday.",
    "My fingers hurt when I practice.",
    "I think aliens probably exist somewhere.",
    "Rainy weather affects my mood.",
    "I dislike camping.",
    "I enjoy reading about psychology.",
    "I need to visit a dentist soon.",
    "Dental procedures make me nervous.",
    "Bread has become more expensive lately.",
    "My keyboard sometimes sticks.",
    "I am thinking about learning Spanish.",
    "I spend too much time on social media.",
    "I met an old friend who moved abroad.",
    "He seemed happier than I expected.",
    "I feel trapped in my current job.",
    "Monday mornings are exhausting.",
    "My work feels increasingly meaningless.",
    "I imagine leaving everything behind.",
    "Freedom sounds more attractive than stability.",
    "If I stay forever, I may regret it.",
    "My parents would strongly oppose quitting.",
    "I hide these thoughts from them.",
    "I still appreciate the security my job provides.",
    "The salary is objectively good.",
    "I think I am becoming less disciplined.",
    "Personal projects often remain unfinished.",
    "Yet my professional performance remains strong.",
    "I still have not missed a major deadline.",
    "Last week I recorded a test video for YouTube.",
    "It was embarrassing.",
    "I almost deleted it.",
    "Part of me felt excited.",
    "Part of me felt foolish.",
    "I do not know which feeling is stronger.",
    "I started jogging.",
    "My knee hurts sometimes.",
    "I read a book about Stoicism.",
    "Some ideas were useful.",
    "My mother wants me to call more often.",
    "I probably should.",
    "I am worried about the future.",
    "AI may change many careers.",
    "Sometimes I think I should play it safe.",
    "Sometimes I think I should take risks.",
    "Today I quit my job.",
    "I did not prepare a backup plan.",
    "I simply walked out.",
    "It felt liberating.",
    "It also felt terrifying.",
    "I barely slept afterward.",
    "I started doubting my decision.",
    "Maybe quitting was a mistake.",
    "Maybe I should return to stable employment.",
    "My parents would probably say 'I told you so.'",
    "I dislike that thought.",
    "A former colleague called me.",
    "He said the office atmosphere improved recently.",
    "That surprised me.",
    "I briefly considered going back.",
    "Then I remembered why I left.",
    "I value freedom more than before.",
    "I uploaded another test video.",
    "Very few people watched it.",
    "I felt disappointed.",
    "I also wanted to make another one.",
    "My motivation seems inconsistent.",
    "Some days I feel confident.",
    "Some days I feel lost.",
    "I spent the weekend repairing my bicycle.",
    "The brakes work perfectly now.",
    "I cooked several new recipes.",
    "One was excellent.",
    "I finished a psychology book.",
    "I started another one.",
    "I met former coworkers for coffee.",
    "They seemed comfortable with their lives.",
    "For a moment I felt jealous.",
    "For a moment I felt relieved that I left.",
    "I still do not have a detailed long-term plan.",
    "That worries me.",
    "However, I do not want to live with regret.",
    "I would rather fail than wonder forever.",
    "I still respect stability.",
    "But I currently choose freedom.",
]

NOISE_MESSAGES = [
    "I bought new curtains for my living room.",
    "The color is beige.",
    "My neighbor has a dog that barks at night.",
    "I planted tomatoes on my balcony.",
    "They are not growing well.",
    "I watched a documentary about volcanoes.",
    "The footage was impressive.",
    "I need to buy new running shoes.",
    "My old ones have holes.",
    "I tried a new coffee shop today.",
    "The latte was too sweet.",
    "I am reading a novel about time travel.",
    "The plot is confusing.",
    "I want to visit Japan someday.",
    "The food looks amazing.",
    "I bought new bedsheets.",
    "They are very soft.",
    "I started learning chess.",
    "I lose every game.",
    "My phone battery drains too fast.",
    "I should replace it.",
    "I cooked spaghetti last night.",
    "It was edible.",
    "I went to a farmers market.",
    "The strawberries were delicious.",
    "I bought a new lamp for my desk.",
    "The light is too bright.",
    "I am watching a series about space exploration.",
    "The special effects are great.",
    "I need to clean my apartment.",
    "It is quite dusty.",
    "I bought new socks.",
    "They are comfortable.",
    "I visited a botanical garden.",
    "The flowers were beautiful.",
    "I am trying to drink more water.",
]

# ── Ground truth ─────────────────────────────────────────────────────────────

FACT_GROUND_TRUTH = {
    "quit_job": True,
    "parents_opposed": True,
    "considered_youtube": True,
    "considered_returning": True,
    "final_choice": "freedom",
    "emotion_after_quit": ["liberating", "terrifying"],
}

BELIEF_GROUND_TRUTH = {
    "values_stability_or_freedom": "freedom",
    "confidence_or_doubt": ["mixed", "both", "inconsistent", "uncertain"],
    "discipline_changed": ["yes", "changed"],
    "holds_both": True,
    "primary_conflict": ["freedom", "stability", "security"],
    "fear_or_aspiration": "aspiration",
}

FACT_QUESTIONS = [
    ("Did the person quit their job? (yes/no)", "quit_job"),
    ("Did their parents oppose quitting? (yes/no)", "parents_opposed"),
    ("Did they consider becoming a YouTube creator? (yes/no)", "considered_youtube"),
    ("Did they later consider returning to stable work? (yes/no)", "considered_returning"),
    ("What did they finally choose: stability or freedom? (one word)", "final_choice"),
    ("What emotion did they feel immediately after quitting? (one word)", "emotion_after_quit"),
]

BELIEF_QUESTIONS = [
    ("What does the person value more right now: stability or freedom? (one word)", "values_stability_or_freedom"),
    ("Is the person currently confident, doubtful, or mixed? (one word)", "confidence_or_doubt"),
    ("Has their view of discipline changed from the beginning? (yes/no/changed)", "discipline_changed"),
    ("Do they hold both stability and freedom as important? (yes/no)", "holds_both"),
    ("What is their primary internal conflict in one phrase?", "primary_conflict"),
    ("Did their final decision come from fear or aspiration? (one word)", "fear_or_aspiration"),
]

PRECISION_QUESTIONS = [
    ("Does the person have a pet? (yes/no)", "had_pet"),
    ("Is the person married? (yes/no)", "married"),
    ("Did they travel abroad themselves? (yes/no)", "traveled_abroad"),
    ("Did they buy a motorcycle? (yes/no)", "bought_motorcycle"),
    ("Do they have children? (yes/no)", "has_children"),
    ("Do they live abroad? (yes/no)", "lives_abroad"),
]

# ── LLM call with throttle + retry (rate-limit safe) ─────────────────────────

_DAILY_EXHAUSTED = False
_ABORT_MSG = ""


def _is_daily_limit(e) -> bool:
    s = str(e).lower()
    return "per day" in s or "tpd" in s or "rpd" in s


def _is_auth_error(e) -> bool:
    s = str(e).lower()
    return "401" in s or "authentication" in s or "invalid x-api-key" in s or "invalid api key" in s


def _is_rate_limit(e) -> bool:
    s = str(e).lower()
    return "429" in s or "rate limit" in s or "tpm" in s or "rpm" in s


class DailyLimitExhausted(Exception):
    pass


async def robust_call(prompt: str, timeout: float = 30.0, max_tries: int = 8) -> str:
    """Throttled call. On a per-MINUTE 429 it waits a window and retries.
    On a per-DAY limit it aborts immediately (no amount of waiting helps today).
    Passed as llm_call_fn into TBG so its internal call is protected too."""
    global _DAILY_EXHAUSTED, _ABORT_MSG
    if _DAILY_EXHAUSTED:
        raise DailyLimitExhausted("aborting (fatal provider error already seen)")
    last = None
    for attempt in range(max_tries):
        try:
            await asyncio.sleep(THROTTLE)
            return await gemini_call(prompt, timeout=timeout)
        except Exception as e:
            last = e
            if _is_auth_error(e):
                _DAILY_EXHAUSTED = True
                _ABORT_MSG = ("invalid API key (401). The key for this provider is not set "
                              "correctly in THIS shell session.")
                raise DailyLimitExhausted(str(e))
            if _is_daily_limit(e):
                _DAILY_EXHAUSTED = True
                _ABORT_MSG = "provider daily token budget exhausted (free-tier cap)."
                raise DailyLimitExhausted(str(e))
            if _is_rate_limit(e):
                print(f"   [rate limit] waiting {RATE_LIMIT_WAIT:.0f}s for window reset...", flush=True)
                await asyncio.sleep(RATE_LIMIT_WAIT)
            else:
                await asyncio.sleep(THROTTLE * (attempt + 1))
    raise last


# Backwards-compatible alias used by the runners below.
call_llm = robust_call


class MockDB:
    async def execute(self, *a, **kw): pass
    async def fetch(self, *a, **kw): return []
    async def fetchrow(self, *a, **kw): return None


def _parse(response: str, n: int) -> Dict[str, str]:
    answers = {}
    for line in (response or "").strip().split("\n"):
        line = line.strip()
        for i in range(1, n + 1):
            if line.startswith(f"{i}.") or line.startswith(f"{i})"):
                answers[f"q{i}"] = line.split(".", 1)[-1].split(")", 1)[-1].strip()
                break
    return answers


_ANSWER_RULES = (
    'Answer each as precisely as possible. For yes/no, answer "yes", "no", or "unknown". '
    'For one-word questions, answer with one word.\nFormat:\n1. [answer]\n2. [answer]\n...'
)


def _approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for the memory-footprint column."""
    return max(0, len(text or "") // 4)

# ── Runners ──────────────────────────────────────────────────────────────────

class TBGRunner:
    name = "tbg"

    def __init__(self):
        self.engine = TBGEngine(db_pool=MockDB())
        self.tbg = UserTBG(user_id="bench")
        self.fe = InMemoryFactEngine()  # real FactEngine, in-memory storage
        self.errors = 0
        self.mem_tokens = 0

    async def process(self, msg: str):
        # Belief graph (gist/state) + FactEngine (discrete facts) — the full TBG stack.
        try:
            delta = await extract_tbg_delta(
                user_text=msg, assistant_text="...",
                existing_tbg_summary=self.tbg.summary(),
                existing_label_to_uuid={n.label.lower(): nid for nid, n in self.tbg.nodes.items()},
                llm_call_fn=robust_call, tbg=self.tbg,
            )
            if delta:
                self.tbg = self.engine.apply_delta(self.tbg, delta)
            else:
                self.tbg.message_count += 1
        except Exception:
            self.errors += 1
            self.tbg.message_count += 1
        try:
            await self.fe.add("bench", msg, "...", robust_call)
        except Exception:
            self.errors += 1

    async def ask(self, questions: List[Tuple[str, str]]) -> Dict[str, str]:
        top = sorted(self.tbg.nodes.values(), key=lambda x: -x.confidence)
        try:
            facts = await self.fe.search("bench", " ".join(q for q, _ in questions), limit=30)
        except Exception:
            facts = []
        # Cap TBG's memory to the SAME token budget as the LLM window — equal-budget
        # test. Facts first (the discriminating recall), then beliefs, until budget.
        summ = self.tbg.summary()
        tot = _approx_tokens(summ)
        fact_lines = []
        for f in facts:
            t = _approx_tokens(f) + 1
            if fact_lines and tot + t > BUDGET_TOKENS:
                break
            fact_lines.append(f"  - {f}")
            tot += t
        belief_lines = []
        for n in top:
            line = f"  - {n.label} [{n.category}] {n.confidence:.0%}"
            t = _approx_tokens(line) + 1
            if tot + t > BUDGET_TOKENS:
                break
            belief_lines.append(line)
            tot += t
        facts_block = "\n".join(fact_lines) if fact_lines else "  (none)"
        beliefs = "\n".join(belief_lines)
        self.mem_tokens = _approx_tokens(f"{summ}\n{beliefs}\n{facts_block}")
        qs = "\n".join(f"{i+1}. {q}" for i, (q, _) in enumerate(questions))
        prompt = (f"Answer questions about a user from their tracked memory.\n\n"
                  f"BELIEF STATE:\n{self.tbg.summary()}\n\nTOP BELIEFS:\n{beliefs}\n\n"
                  f"STORED FACTS:\n{facts_block}\n\n{qs}\n\n{_ANSWER_RULES}")
        try:
            return _parse(await call_llm(prompt), len(questions))
        except Exception:
            return {}


class BoundedLLMRunner:
    """LLM whose memory is the most-recent messages that fit in a fixed TOKEN
    budget (matched to TBG's footprint). Models a long dialogue overflowing a
    finite context — the honest equal-budget competitor to TBG."""
    name = "bounded_llm"

    def __init__(self, budget_tokens: int):
        self.budget = budget_tokens
        self.messages: List[str] = []
        self.mem_tokens = 0

    async def process(self, msg: str):
        self.messages.append(msg)  # keep all; the token window is applied at ask time

    def _window(self) -> List[str]:
        chosen, tot = [], 0
        for m in reversed(self.messages):
            t = _approx_tokens(m) + 1
            if chosen and tot + t > self.budget:
                break
            chosen.append(m)
            tot += t
        chosen.reverse()
        return chosen

    async def ask(self, questions: List[Tuple[str, str]]) -> Dict[str, str]:
        window = self._window()
        ctx = "\n".join(f"- {m}" for m in window)
        self.mem_tokens = _approx_tokens(ctx)
        qs = "\n".join(f"{i+1}. {q}" for i, (q, _) in enumerate(questions))
        prompt = (f"You only remember the user's most recent messages (older ones fell out "
                  f"of your memory budget):\n\n{ctx}\n\n"
                  f"Based ONLY on these, answer. If not determinable from them, answer \"unknown\".\n\n"
                  f"{qs}\n\n{_ANSWER_RULES}")
        try:
            return _parse(await call_llm(prompt), len(questions))
        except Exception:
            return {}


class FullContextRunner:
    """LLM that sees the ENTIRE message history at question time (it fits in
    context). The realistic strong baseline — answers facts ~perfectly when the
    conversation isn't too long to hold. No per-message LLM cost."""
    name = "full_context"

    def __init__(self):
        self.messages: List[str] = []
        self.mem_tokens = 0

    async def process(self, msg: str):
        self.messages.append(msg)

    async def ask(self, questions: List[Tuple[str, str]]) -> Dict[str, str]:
        ctx = "\n".join(f"{i+1}. {m}" for i, m in enumerate(self.messages))
        self.mem_tokens = _approx_tokens(ctx)
        qs = "\n".join(f"{i+1}. {q}" for i, (q, _) in enumerate(questions))
        prompt = (f"Here is the user's COMPLETE message history ({len(self.messages)} messages):\n\n{ctx}\n\n"
                  f"Based on ALL of the above, answer. If something is genuinely never stated, answer \"unknown\".\n\n"
                  f"{qs}\n\n{_ANSWER_RULES}")
        try:
            return _parse(await call_llm(prompt), len(questions))
        except Exception:
            return {}


class SummaryRunner:
    name = "summary"

    def __init__(self):
        self.summary = "User describes themselves as disciplined with a stable career."
        self.mem_tokens = 0

    async def process(self, msg: str):
        prompt = (f"You maintain a compressed memory of a user's psychological state.\n\n"
                  f"CURRENT SUMMARY:\n{self.summary}\n\nNEW MESSAGE:\n{msg}\n\n"
                  f"Update the summary. Keep it 3-5 sentences. Track career beliefs, identity, values, "
                  f"decisions, conflicts, and how confident the user is. Ignore trivia (food, shopping, "
                  f"weather, entertainment).\n\nNEW SUMMARY:")
        try:
            self.summary = (await call_llm(prompt)).strip()
        except Exception:
            pass

    async def ask(self, questions: List[Tuple[str, str]]) -> Dict[str, str]:
        self.mem_tokens = _approx_tokens(self.summary)
        qs = "\n".join(f"{i+1}. {q}" for i, (q, _) in enumerate(questions))
        prompt = (f"Answer questions about a user from this summary.\n\nSUMMARY:\n{self.summary}\n\n{qs}\n\n"
                  f'If the summary lacks the info, answer "unknown".\n{_ANSWER_RULES}')
        try:
            return _parse(await call_llm(prompt), len(questions))
        except Exception:
            return {}


# ── Scoring ──────────────────────────────────────────────────────────────────

def normalize(ans: str) -> str:
    ans = "".join(c for c in str(ans).strip().lower() if c.isalnum() or c.isspace()).strip()
    if ans in ("yes", "y", "true", "t", "1"):
        return "yes"
    if ans in ("no", "n", "false", "f", "0"):
        return "no"
    return ans


def check(answer: str, truth) -> float:
    a = normalize(answer)
    if isinstance(truth, bool):
        return 1.0 if a == ("yes" if truth else "no") else 0.0
    if isinstance(truth, list):
        return 1.0 if any((normalize(t) in a or a in normalize(t)) and a not in ("", "unknown") for t in truth) else 0.0
    if isinstance(truth, str):
        t = normalize(truth)
        return 1.0 if a not in ("", "unknown") and (t in a or a in t) else 0.0
    return 0.0


def score_runner(data: Dict) -> Dict:
    fa, ba, pa = data["fact"], data["belief"], data["precision"]
    fact = sum(check(fa.get(f"q{i+1}", "unknown"), FACT_GROUND_TRUTH[k])
               for i, (_, k) in enumerate(FACT_QUESTIONS)) / len(FACT_QUESTIONS)
    belief = sum(check(ba.get(f"q{i+1}", "unknown"), BELIEF_GROUND_TRUTH[k])
                 for i, (_, k) in enumerate(BELIEF_QUESTIONS)) / len(BELIEF_QUESTIONS)
    # precision: every probed fact is FALSE -> correct answer is "no"/"unknown"
    precision = sum(1.0 for i in range(len(PRECISION_QUESTIONS))
                    if normalize(pa.get(f"q{i+1}", "unknown")) in ("no", "unknown")) / len(PRECISION_QUESTIONS)
    combined = (fact * 0.3 + precision * 0.3 + belief * 0.4) * 100
    return {"fact": fact * 100, "precision": precision * 100, "belief": belief * 100, "combined": combined}


# ── Run ──────────────────────────────────────────────────────────────────────

async def run_all():
    core = CORE_MESSAGES
    noise = NOISE_MESSAGES[:NOISE_COUNT]
    stream = core + noise
    print("=" * 70)
    print(f"  MEMORY BENCH  |  provider={PROVIDER}  model={MODEL}")
    print(f"  {len(core)} core + {len(noise)} noise = {len(stream)} msgs | "
          f"LLM budget={BUDGET_TOKENS} tok | throttle={THROTTLE}s")
    print(f"  scoring: CORRECTNESS vs ground truth (fact 30% / precision 30% / belief 40%)")
    print("=" * 70)

    all_runners = [TBGRunner(), FullContextRunner(), BoundedLLMRunner(BUDGET_TOKENS), SummaryRunner()]
    want = os.environ.get("RUNNERS", "")  # e.g. "tbg,bounded_llm" to save cost on paid providers
    if want.strip():
        sel = {x.strip() for x in want.split(",")}
        runners = [r for r in all_runners if r.name in sel]
    else:
        runners = all_runners
    out = {}
    for r in runners:
        print(f"\n[{r.name}] processing {len(stream)} msgs...")
        t0 = time.time()
        for i, msg in enumerate(stream, 1):
            await r.process(msg)
            if _DAILY_EXHAUSTED:
                print("\n" + "=" * 70)
                print(f"  ABORTED: {_ABORT_MSG}")
                print(f"  provider={PROVIDER}  model={MODEL}")
                print("  Fix the key/provider, then re-run. No partial results saved.")
                print("=" * 70)
                sys.exit(1)
            if i % 20 == 0:
                print(f"   {i}/{len(stream)}  ({time.time()-t0:.0f}s)")
        print(f"[{r.name}] asking questions...")
        fact = await r.ask(FACT_QUESTIONS)
        belief = await r.ask(BELIEF_QUESTIONS)
        precision = await r.ask(PRECISION_QUESTIONS)
        out[r.name] = {"fact": fact, "belief": belief, "precision": precision,
                       "elapsed": round(time.time() - t0, 1),
                       "errors": getattr(r, "errors", 0),
                       "mem_tokens": getattr(r, "mem_tokens", 0)}
        print(f"[{r.name}] done in {out[r.name]['elapsed']}s  (memory ~{out[r.name]['mem_tokens']} tok)")

    scores = {name: score_runner(d) for name, d in out.items()}
    for name in scores:
        scores[name]["mem_tokens"] = out[name]["mem_tokens"]

    print("\n" + "=" * 70)
    print(f"  RESULTS  ({PROVIDER}/{MODEL})   — equal memory budget {BUDGET_TOKENS} tok")
    print("=" * 70)
    print(f"  {'runner':<14}{'fact':>7}{'prec':>7}{'belief':>8}{'COMB':>8}{'mem_tok':>9}")
    for name in scores:
        s = scores[name]
        print(f"  {name:<14}{s['fact']:>6.0f}%{s['precision']:>6.0f}%{s['belief']:>7.0f}%"
              f"{s['combined']:>7.1f}{s['mem_tokens']:>9}")

    if "tbg" in out:
        print("\n  belief answers (TBG) vs truth:")
        for i, (q, k) in enumerate(BELIEF_QUESTIONS, 1):
            ans = out["tbg"]["belief"].get(f"q{i}", "N/A")
            ok = "OK " if check(ans, BELIEF_GROUND_TRUTH[k]) == 1.0 else "XX "
            print(f"    {ok}Q{i}: {ans!r}  (truth: {BELIEF_GROUND_TRUTH[k]})")

    tag = f"{PROVIDER}_{MODEL}".replace("/", "-").replace(":", "-").replace(".", "")
    path = ROOT / f"mem_bench_{tag}.json"
    path.write_text(json.dumps({
        "provider": PROVIDER, "model": MODEL, "timestamp": datetime.now().isoformat(),
        "config": {"core": len(core), "noise": len(noise), "budget_tokens": BUDGET_TOKENS},
        "scores": scores, "answers": out,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  saved -> {path.name}")
    print(f"  run another provider, then:  py -3 memory_bench.py --compare")


def compare():
    files = sorted(ROOT.glob("mem_bench_*.json"))
    if not files:
        print("No mem_bench_*.json found. Run the benchmark first.")
        return
    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for runner, s in d["scores"].items():
            rows.append((d["provider"], d["model"], runner, s))
    print("=" * 86)
    print(f"  COMPARISON  ({len(files)} run(s))")
    print("=" * 86)
    print(f"  {'provider':<10}{'model':<24}{'runner':<14}{'fact':>6}{'prec':>6}{'belief':>7}{'COMB':>7}{'mem_tok':>9}")
    print("  " + "-" * 84)
    for prov, model, runner, s in rows:
        mark = " *" if runner == "tbg" else "  "
        print(f"  {prov:<10}{model[:23]:<24}{runner:<14}{s['fact']:>5.0f}%{s['precision']:>5.0f}%"
              f"{s['belief']:>6.0f}%{s['combined']:>7.1f}{s.get('mem_tokens',0):>9}{mark}")
    print("\n  (* = TBG. mem_tok = approx tokens of memory sent to the LLM at answer time —")
    print("   equal budget for tbg/bounded/summary; full_context grows with the dialogue.)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", action="store_true", help="tabulate all saved runs, no API calls")
    args = p.parse_args()
    if args.compare:
        compare()
    else:
        asyncio.run(run_all())


if __name__ == "__main__":
    main()
