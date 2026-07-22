#!/usr/bin/env python3
"""
TBG FactEngine на LoCoMo (F1, 1 разговор). Сравнение с ОПУБЛИКОВАННЫМ Mem0 (F1-полоса ~0.41-0.46).
Mem0 сами НЕ гоняем. Скоринг — детерминированный token-F1, без judge.

Pipeline (ТЗ):
  ingest: turns -> FactEngine.add(...) на free-модели  -> кэш фактов (tools/locomo_facts.json)
  answer: per QA -> FactEngine.search top-K -> 1 LLM-вызов (фикс. answer-промпт) -> ответ
  score : token-F1 vs gold, разбивка по категориям 1-4 (+ cat5 adversarial отдельно)

Хранилище: in-memory shim (locomo_shim.InMemoryPool); методы FactEngine как есть.
Бюджет-гард: кап на LLM-вызовы, abort при превышении.

ЗАПУСК (Cerebras free):
  CEREBRAS_API_KEY=... LLM_PROVIDER=cerebras LLM_MODEL=gpt-oss-120b LLM_TEMPERATURE=0 \
    py -3 -u tools/locomo_harness.py --ingest      # строит память -> кэш (дорогая фаза)
  ... --answer    # из кэша, скоринг (повторяемо)
  ... (без флагов = ingest + answer)
"""
import os
import sys
import re
import json
import asyncio
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from locomo_shim import InMemoryPool

LOCOMO = HERE / "locomo10.json"
FACTS_CACHE = HERE / "locomo_facts.json"
PROGRESS = HERE / "locomo_progress.txt"   # heartbeat — обновляется каждый турн
OUT = HERE / "locomo_result.json"
USER_ID = "locomo0"
SEARCH_K = 8
CALL_CAP = 1500           # abort if exceeded
MEM0_F1_BAND = (0.41, 0.46)

ANSWER_PROMPT = (
    "Answer the question using ONLY the memory facts below. "
    "Give the shortest possible answer — a name, date, number, place, or short phrase. "
    "Do not explain. If the facts do not contain the answer, reply exactly: no information.\n\n"
    "Memory facts:\n{facts}\n\nQuestion: {q}\nShort answer:"
)

_calls = {"ingest": 0, "answer": 0}


def _conv0():
    data = json.loads(LOCOMO.read_text(encoding="utf-8"))
    return data[0]


def _ordered_turns(conv):
    sess = sorted([k for k in conv if re.fullmatch(r"session_\d+", k)],
                  key=lambda k: int(k.split("_")[1]))
    turns = []
    for sk in sess:
        date = conv.get(f"{sk}_date_time", "")
        for t in conv[sk]:
            turns.append({"speaker": t.get("speaker", ""), "text": t.get("text", ""), "date": date})
    return turns


# ── LLM (free provider) with budget guard ──
async def _llm(prompt, phase):
    from llm_client import gemini_call
    if _calls["ingest"] + _calls["answer"] >= CALL_CAP:
        raise RuntimeError(f"CALL_CAP {CALL_CAP} exceeded")
    _calls[phase] += 1
    return await gemini_call(prompt, timeout=90.0)


# ── INGEST ──
async def ingest():
    from fact_engine import FactEngine
    turns = _ordered_turns(_conv0()["conversation"])
    print(f"conv0: {len(turns)} turns; ingesting via FactEngine.add (free model)...", flush=True)

    pool = InMemoryPool()
    fe = FactEngine(pool)

    async def llm(p):
        return await _llm(p, "ingest")

    def _hb(i, note=""):
        PROGRESS.write_text(
            f"{datetime.now(timezone.utc).isoformat()} turn={i}/{len(turns)} "
            f"facts={len(pool.facts)} calls={_calls['ingest']} {note}\n", encoding="utf-8")

    def _dump():
        dump = [{"id": r["id"], "fact": r["fact"],
                 "embedding": np.asarray(r["embedding"], dtype=float).tolist(),
                 "updated_at": r["updated_at"].isoformat() if hasattr(r["updated_at"], "isoformat") else str(r["updated_at"])}
                for r in pool.facts.values()]
        FACTS_CACHE.write_text(json.dumps(dump, ensure_ascii=False), encoding="utf-8")
        return len(dump)

    _hb(0, "start")
    for i, t in enumerate(turns, 1):
        text = f"{t['speaker']}: {t['text']}".strip()
        if not text:
            _hb(i); continue
        try:
            await fe.add(USER_ID, text, "", llm, date_context=str(t["date"]))
        except RuntimeError as e:
            print(f"  ABORT at turn {i}: {e}", flush=True)
            _hb(i, f"ABORT {e}")
            break
        except Exception as e:
            print(f"  turn {i} err {type(e).__name__}: {e}", flush=True)
        _hb(i)
        if i % 10 == 0:
            n = _dump()  # periodic partial cache (survives crash)
            print(f"  ...{i}/{len(turns)} turns, facts={n}, calls={_calls['ingest']}", flush=True)

    n = _dump()
    _hb(len(turns), "DONE")
    print(f"INGEST done: {n} facts cached -> {FACTS_CACHE.name}; ingest_calls={_calls['ingest']}", flush=True)
    return pool


def _load_pool_from_cache():
    pool = InMemoryPool()
    rows = json.loads(FACTS_CACHE.read_text(encoding="utf-8"))
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["updated_at"])
        except Exception:
            ts = datetime.now(timezone.utc)
        pool.facts[r["id"]] = {"id": r["id"], "user_id": USER_ID, "fact": r["fact"],
                               "embedding": np.asarray(r["embedding"], dtype=float),
                               "source": "explicit", "updated_at": ts}
    return pool


# ── F1 (SQuAD-style token F1) ──
def _normalize(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def f1(pred, gold):
    p, g = _normalize(pred), _normalize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec = same / len(p)
    rec = same / len(g)
    return 2 * prec * rec / (prec + rec)


# ── ANSWER + SCORE ──
async def answer_and_score():
    from fact_engine import FactEngine
    pool = _load_pool_from_cache()
    fe = FactEngine(pool)
    conv = _conv0()
    qa = _conv0_qa()
    print(f"facts in memory: {len(pool.facts)}; QA total: {len(qa)}", flush=True)

    async def llm(p):
        return await _llm(p, "answer")

    cat_names = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    results = []
    for j, q in enumerate(qa, 1):
        cat = int(q.get("category", 0))
        question = str(q.get("question", ""))
        gold = str(q.get("answer", q.get("adversarial_answer", "")))
        try:
            facts = await fe.search(USER_ID, question, limit=SEARCH_K)
            fact_block = "\n".join(f"- {f}" for f in facts) or "(no facts)"
            pred = await llm(ANSWER_PROMPT.format(facts=fact_block, q=question))
            pred = str(pred).strip()
        except RuntimeError as e:
            print(f"  ABORT at QA {j}: {e}", flush=True)
            break
        except Exception as e:
            pred = ""
            print(f"  QA {j} err {type(e).__name__}: {e}", flush=True)
        results.append({"category": cat, "question": question, "gold": gold,
                        "pred": pred, "f1": f1(pred, gold), "n_facts": len(facts)})
        PROGRESS.write_text(
            f"{datetime.now(timezone.utc).isoformat()} ANSWER qa={j}/{len(qa)} "
            f"calls={_calls['answer']}\n", encoding="utf-8")
        if j % 10 == 0:
            OUT.write_text(json.dumps({"calls": _calls, "results": results}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            print(f"  ...{j}/{len(qa)} QA, answer_calls={_calls['answer']}", flush=True)

    OUT.write_text(json.dumps({"calls": _calls, "results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    _report(results, cat_names)


def _conv0_qa():
    return _conv0().get("qa", [])


def _report(results, cat_names):
    import statistics as st
    print("\n" + "=" * 70)
    print("LoCoMo — TBG FactEngine — token-F1 (no judge)")
    print("=" * 70)
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["f1"])

    main_cats = [1, 2, 3, 4]
    main_f1 = [r["f1"] for r in results if r["category"] in main_cats]
    print(f"{'category':<16}{'n':>5}{'F1':>9}")
    for c in main_cats:
        v = by_cat.get(c, [])
        if v:
            print(f"{cat_names.get(c, c):<16}{len(v):>5}{st.mean(v):>9.3f}")
    overall = st.mean(main_f1) if main_f1 else 0.0
    print("-" * 30)
    print(f"{'OVERALL (1-4)':<16}{len(main_f1):>5}{overall:>9.3f}")
    if 5 in by_cat:
        print(f"{'adversarial(5)':<16}{len(by_cat[5]):>5}{st.mean(by_cat[5]):>9.3f}  (отдельно, ТЗ не включает)")

    print(f"\ncalls: ingest={_calls['ingest']} answer={_calls['answer']}")
    lo, hi = MEM0_F1_BAND
    print(f"\nMem0 опубликованная F1-полоса: {lo:.2f}–{hi:.2f}")
    verdict = ("В ПОЛОСЕ Mem0 (приличен для standalone)" if lo <= overall <= hi else
               "ВЫШЕ полосы Mem0" if overall > hi else
               "НИЖЕ полосы Mem0")
    print(f"TBG FactEngine overall(1-4) = {overall:.3f}  ->  {verdict}")
    print("\nОговорка: LoCoMo — fact-QA. Belief-слой TBG (контрадикция/decay/confidence) "
          "LoCoMo НЕ меряет — это граница бенча, не провал.")


async def main():
    mode_ingest = "--ingest" in sys.argv[1:]
    mode_answer = "--answer" in sys.argv[1:]
    if not mode_ingest and not mode_answer:
        mode_ingest = mode_answer = True
    if mode_ingest:
        await ingest()
    if mode_answer:
        if not FACTS_CACHE.exists():
            print("no facts cache — run --ingest first"); return
        await answer_and_score()


if __name__ == "__main__":
    asyncio.run(main())
