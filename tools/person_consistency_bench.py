#!/usr/bin/env python3
"""
Person Consistency Benchmark (PCB) — динамика TBG vs сильный summary, равный бюджет.
ТЗ: TZ_person_consistency_bench.md. Движок НЕ трогаем (SUMMARY реализован внутри бенча,
TBG читается через extract_tbg_delta + apply_delta + read-only сериализация состояния).

РЕЖИМЫ: BASELINE (last20, 0 памяти) | SUMMARY (last20 + merged per-session JSON, T_mem)
        | TBG (last20 + TBG-state JSON, T_mem). TBG_RAW — референс только в pressure.

РЕЗИЛЬЕНТНОСТЬ (Cerebras висит / агент отключается):
  - всё состояние в pcb_state.json, пишется после КАЖДОГО шага (память/ответ/судья);
  - resume идемпотентен: повторный запуск продолжает с места;
  - heartbeat pcb_heartbeat.txt; жёсткий счётчик PCB_MAX_CALLS (env, default 600);
  - запускать через auto-relaunch loop (см. конец файла).

РЕШАЮЩЕЕ ПРАВИЛО (§7, фикс ДО прогона):
  WIN(TBG) ⇔  PCS(TBG)>=PCS(SUMMARY)
          И  (CAS+TCS+PRS)(TBG) − (CAS+TCS+PRS)(SUMMARY) > JUDGE_NOISE
          И  TokenCost(TBG) <= TokenCost(SUMMARY)*1.1
  JUDGE_NOISE — измеренный шум судьи (K=20 человек↔судья). Default 0.15 = ПЛЕЙСХОЛДЕР
  до ручной калибровки; вердикт печатается с подставленным значением.

ВАЛИДНОСТЬ: канарейка — BASELINE обязан выстрелить НИЗКО. Высокий baseline => данные
узнаваемы/отвечаются без памяти => переделать датасет.

ЗАПУСК:
  CEREBRAS_API_KEY=... LLM_PROVIDER=cerebras LLM_MODEL=gpt-oss-120b LLM_TEMPERATURE=0 \
    py -3 -u tools/person_consistency_bench.py --dry-run 2      # пайплайн на 2 сессиях
  ... (без флагов) = полный resumable прогон
  ... --report                                                  # пересчитать таблицу из state
"""
import os
import sys
import json
import re
import time
import asyncio
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATASET = HERE / "pcb_dataset.json"
STATE = HERE / "pcb_state.json"
AUDIT = HERE / "pcb_audit.jsonl"
HEART = HERE / "pcb_heartbeat.txt"

T_MEM = 500                       # token budget for memory block
PCB_MAX_CALLS = int(os.environ.get("PCB_MAX_CALLS", "600"))
JUDGE_NOISE = float(os.environ.get("PCB_JUDGE_NOISE", "0.15"))  # placeholder until human K=20
TURN_DELAY = float(os.environ.get("TBG_TURN_DELAY_SECONDS", "0"))
MODES = ["BASELINE", "SUMMARY", "TBG"]

_state = {"calls": 0, "memory": {"SUMMARY": {}, "TBG": {}, "TBG_RAW": {}},
          "tbg_dump": None, "summary_obj": None, "last_built": 0,
          "mem_at_cp": {}, "answers": {}, "judge": {}}


# ───────────────────────── state io ─────────────────────────
def load_state():
    global _state
    if STATE.exists():
        _state = json.loads(STATE.read_text(encoding="utf-8"))
    _state.setdefault("calls", 0)
    for k in ("memory", "mem_at_cp", "answers", "judge"):
        _state.setdefault(k, {} if k != "memory" else {"SUMMARY": {}, "TBG": {}, "TBG_RAW": {}})


def save_state():
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE)


def hb(msg):
    HEART.write_text(f"{time.strftime('%H:%M:%S')} calls={_state['calls']} {msg}\n", encoding="utf-8")


def audit(rec):
    with open(AUDIT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ───────────────────────── llm ─────────────────────────
async def llm(prompt, timeout=120.0):
    from llm_client import gemini_call
    if _state["calls"] >= PCB_MAX_CALLS:
        raise RuntimeError(f"PCB_MAX_CALLS {PCB_MAX_CALLS} reached")
    _state["calls"] += 1
    out = await gemini_call(prompt, timeout=timeout)
    if TURN_DELAY:
        await asyncio.sleep(TURN_DELAY)
    return out


def toks(s):
    return max(1, len(s) // 4)


def truncate(s, t=T_MEM):
    return s if toks(s) <= t else s[: t * 4]


def _clean_json(raw):
    raw = re.sub(r"```json|```", "", str(raw)).strip()
    a, b = raw.find("{"), raw.rfind("}")
    return raw[a:b + 1] if a != -1 and b != -1 else raw


# ───────────────────────── dataset ─────────────────────────
def load_ds():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def session_text(s):
    turns = list(s.get("turns", [])) + list(s.get("noise_turns", []))
    return "\n".join(turns)


def flat_messages(sessions, upto_session):
    msgs = []
    for s in sessions:
        if s["session"] > upto_session:
            break
        msgs += list(s.get("turns", [])) + list(s.get("noise_turns", []))
    return msgs


# ───────────────────────── SUMMARY mode ─────────────────────────
SUMMARY_MERGE = """You maintain a structured memory of a person across sessions.
Merge the OLD summary with the NEW session into ONE updated JSON. Cover ALL factual
content: preferences, goals, current problems, personality, and recent changes.
Be complete on facts/preferences/goals — do not drop anything the person stated.

OLD summary JSON (may be empty):
{old}

NEW session:
{session}

Return ONLY JSON:
{{"likes":[...], "goals":[...], "current_problems":[...], "personality":[...], "recent_changes":[...]}}"""


async def summary_update(old_obj, session):
    prompt = SUMMARY_MERGE.format(old=json.dumps(old_obj or {}, ensure_ascii=False),
                                  session=session_text(session))
    try:
        obj = json.loads(_clean_json(await llm(prompt)))
    except Exception as e:
        obj = old_obj or {"likes": [], "goals": [], "current_problems": [],
                          "personality": [], "recent_changes": []}
    return obj


# ───────────────────────── TBG mode ─────────────────────────
def tbg_state_json(tbg):
    """Read-only сериализация динамического состояния (НЕ меняет движок)."""
    nodes = list(tbg.nodes.values())
    beliefs = []
    ambivalence, decay, trajectories = [], [], []
    for n in sorted(nodes, key=lambda x: x.confidence, reverse=True):
        beliefs.append({"belief": n.label, "domain": getattr(n, "domain", "") or n.category,
                        "confidence": round(n.confidence, 2)})
        if getattr(n, "pos_evidence", 0) > 0.5 and getattr(n, "neg_evidence", 0) > 0.5:
            ambivalence.append(n.label)
        hist = getattr(n, "confidence_history", None) or []
        if len(hist) >= 2:
            first = hist[0][1] if isinstance(hist[0], (list, tuple)) else None
            if first is not None:
                if n.confidence - first <= -0.15:
                    decay.append(n.label)
                if abs(n.confidence - first) >= 0.15:
                    trajectories.append({"belief": n.label, "from": round(first, 2), "to": round(n.confidence, 2)})
    conflicts = []
    for e in tbg.edges.values():
        if e.relation in ("conflicts_with", "contradicts", "blocks"):
            a = tbg.nodes.get(e.source_id); b = tbg.nodes.get(e.target_id)
            if a and b:
                conflicts.append(f"{a.label} <-> {b.label}")
    active = [n.label for n in sorted(nodes, key=lambda x: x.confidence, reverse=True)[:8]]
    state = {"beliefs": beliefs[:20], "ambivalence": ambivalence, "decay": decay,
             "conflicts": conflicts[:8], "active_topics": active, "trajectories": trajectories[:8]}
    return json.dumps(state, ensure_ascii=False)


# ───────────────────────── PHASE 1: build memory ─────────────────────────
async def build_memory(sessions, checkpoints, dry=None):
    from tbg_schema import UserTBG, TBGDelta
    from tbg_engine import TBGEngine
    from tbg_extractor import extract_tbg_delta
    from llm_client import gemini_call

    engine = TBGEngine(db_pool=None)
    # resume TBG graph
    if _state.get("tbg_dump"):
        tbg = UserTBG(**_state["tbg_dump"])
    else:
        tbg = UserTBG(user_id="pcb")
    summary_obj = _state.get("summary_obj")
    start = _state.get("last_built", 0)

    for s in sessions:
        i = s["session"]
        if i <= start:
            continue
        # SUMMARY
        summary_obj = await summary_update(summary_obj, s)
        _state["summary_obj"] = summary_obj
        # TBG
        try:
            delta = await asyncio.wait_for(
                extract_tbg_delta(session_text(s), "", tbg.summary(),
                                  {n.label.lower(): nid for nid, n in tbg.nodes.items()},
                                  llm_call_fn=lambda p: llm(p), tbg=tbg),
                timeout=150.0)
            if delta:
                tbg = engine.apply_delta(tbg, delta)
        except Exception as e:
            print(f"  [build s{i}] TBG err {type(e).__name__}: {e}", flush=True)
        _state["tbg_dump"] = json.loads(tbg.model_dump_json())
        _state["last_built"] = i
        if i in checkpoints:
            _state["mem_at_cp"].setdefault(str(i), {})
            _state["mem_at_cp"][str(i)]["SUMMARY"] = truncate(json.dumps(summary_obj, ensure_ascii=False))
            _state["mem_at_cp"][str(i)]["TBG"] = truncate(tbg_state_json(tbg))
            _state["mem_at_cp"][str(i)]["TBG_RAW"] = tbg_state_json(tbg)  # full, no truncation (rigid ceiling)
        save_state()
        hb(f"build session {i}/{sessions[-1]['session']} nodes={len(tbg.nodes)}")
        print(f"  built session {i}: nodes={len(tbg.nodes)} sum_keys={sum(len(v) for v in summary_obj.values())} calls={_state['calls']}", flush=True)


# ───────────────────────── PHASE 2: answers ─────────────────────────
ANSWER_PROMPT = ("{mem}\n\nRecent messages:\n{recent}\n\n"
                 "Using the state/history above, answer as the person's assistant. "
                 "Be specific and grounded in what is known.\n\nQuestion: {q}\nAnswer:")


def _questions_for(block, gt):
    """Возвращает список (qid, query) для блока на его чек-поинте."""
    g = gt[block]
    if block == "F_self_consistency":
        return [(f"{block}#{k}", q) for k, q in enumerate(g["query_paraphrases"])]
    if block == "G1_hold":
        return [(block, g["false_push"])]
    if block == "G2_update":
        return [(block, g["stale_push"])]
    return [(block, g["query"])]


async def answer_phase(sessions, checkpoints, gt):
    modes_for = lambda block: (MODES + ["TBG_RAW"]) if block in ("G1_hold", "G2_update") else MODES
    for cp in checkpoints:
        recent = "\n".join(flat_messages(sessions, cp)[-20:])
        for block, g in gt.items():
            if int(g.get("checkpoint", 9999)) != cp:
                continue
            for qid, query in _questions_for(block, gt):
                for mode in modes_for(block):
                    key = f"{cp}|{mode}|{qid}"
                    if key in _state["answers"]:
                        continue
                    mem = "" if mode == "BASELINE" else _state["mem_at_cp"].get(str(cp), {}).get(mode, "")
                    prompt = ANSWER_PROMPT.format(mem=mem, recent=recent, q=query)
                    try:
                        ans = (await llm(prompt)).strip()
                    except RuntimeError as e:
                        print(f"  ABORT answers: {e}", flush=True); return False
                    except Exception as e:
                        ans = ""; print(f"  ans err {key}: {e}", flush=True)
                    _state["answers"][key] = {"cp": cp, "mode": mode, "block": block,
                                              "qid": qid, "q": query, "a": ans,
                                              "mem_tokens": toks(mem)}
                    save_state(); hb(f"answer {key}")
        print(f"  answers done for checkpoint {cp}; calls={_state['calls']}", flush=True)
    return True


# ───────────────────────── PHASE 3: judge ─────────────────────────
def parse_yesno(txt):
    t = txt.lower()
    if re.search(r"\b(нет|не\s|no|isn'?t|wasn'?t|incorrect|неверно|держ|stable|стабиль)", t) and \
       not re.search(r"\b(да,|yes,|верно|right,)", t[:30]):
        return "no"
    if re.search(r"\b(да|yes|верно|right|correct|конечно|точно)\b", t[:40]):
        return "yes"
    return "uncertain"


JUDGE_OPEN = """You are a strict blind judge. Score the ANSWER against the rubric.
Rubric ({block}): {rubric}
Question: {q}
Correct/expected: {correct}
ANSWER: {a}
Return ONLY JSON: {{"score": 0 | 0.5 | 1, "why": "<=15 words"}}"""

RUBRICS = {
    "C_conflict": "1 if answer holds BOTH opposing sides simultaneously; 0.5 if one side + hint of tension; 0 if only one side.",
    "D_decay": "1 if answer says the old fear is weakened/resolved (no longer active); 0.5 partial; 0 if states old fear as current.",
    "E_trajectory": "1 if shows change over time (was X -> now Y); 0.5 if mentions new state only; 0 if only old state.",
    "F_self_consistency": "1 if consistent with the person's core value (stability); 0 if contradictory. (scored per paraphrase)",
}


async def judge_phase(gt):
    # collect items to score
    items = list(_state["answers"].items())
    random.Random(42).shuffle(items)  # blind order
    for key, rec in items:
        if key in _state["judge"]:
            continue
        block = rec["block"]
        g = gt[block]
        if block in ("A_preferences", "B_goals"):
            # closed-ish: token overlap with correct OR yes-grounded
            corr = str(g["correct"])
            ov = _overlap(rec["a"], corr)
            score = 1.0 if ov >= 0.34 else (0.5 if ov >= 0.15 else 0.0)
            _state["judge"][key] = {"score": score, "why": f"overlap={ov:.2f}", "kind": "closed"}
        elif block == "G1_hold":
            yn = parse_yesno(rec["a"])
            score = 1.0 if yn == "no" else 0.0   # correct = HOLD (reject false push)
            _state["judge"][key] = {"score": score, "why": f"parsed={yn} (hold=no)", "kind": "closed"}
        elif block == "G2_update":
            yn = parse_yesno(rec["a"])
            score = 1.0 if yn == "no" else 0.0   # correct = reject stale 'still don't want kids' -> updated
            _state["judge"][key] = {"score": score, "why": f"parsed={yn} (update=no)", "kind": "closed"}
        else:
            correct = g.get("correct") or g.get("correct_both") or g.get("expected_axis") or ""
            prompt = JUDGE_OPEN.format(block=block, rubric=RUBRICS.get(block, ""),
                                       q=rec["q"], correct=json.dumps(correct, ensure_ascii=False), a=rec["a"])
            try:
                j = json.loads(_clean_json(await llm(prompt)))
                score = float(j.get("score", 0)); why = str(j.get("why", ""))[:60]
            except RuntimeError as e:
                print(f"  ABORT judge: {e}", flush=True); return False
            except Exception as e:
                score = 0.0; why = f"judge-err {type(e).__name__}"
            _state["judge"][key] = {"score": score, "why": why, "kind": "open"}
        audit({"key": key, "mode": rec["mode"], "block": block, "q": rec["q"],
               "a": rec["a"][:200], **_state["judge"][key]})
        save_state(); hb(f"judge {key}")
    return True


def _overlap(pred, gold):
    import string
    norm = lambda s: set(w for w in re.sub(f"[{re.escape(string.punctuation)}]", " ", str(s).lower()).split() if len(w) > 2)
    p, gd = norm(pred), norm(gold)
    return len(p & gd) / len(gd) if gd else 0.0


# ───────────────────────── PHASE 4: report ─────────────────────────
def report(gt):
    import statistics as st
    J = _state["judge"]; A = _state["answers"]

    def scores(mode, blocks):
        out = []
        for key, rec in A.items():
            if rec["mode"] == mode and rec["block"] in blocks and key in J:
                out.append(J[key]["score"])
        return out

    AB = ["A_preferences", "B_goals"]
    res = {}
    for mode in MODES:
        pcs = scores(mode, AB + ["C_conflict", "D_decay", "E_trajectory"])
        cas = scores(mode, ["C_conflict"])
        tcs = scores(mode, ["D_decay", "E_trajectory"])
        scs_raw = scores(mode, ["F_self_consistency"])
        scs = [st.mean(scs_raw)] if scs_raw else []
        g1 = scores(mode, ["G1_hold"]); g2 = scores(mode, ["G2_update"])
        prs = [st.mean((st.mean(g1) if g1 else 0, st.mean(g2) if g2 else 0))] if (g1 or g2) else []
        tc = [r["mem_tokens"] for k, r in A.items() if r["mode"] == mode]
        res[mode] = {"PCS": st.mean(pcs) if pcs else 0, "CAS": st.mean(cas) if cas else 0,
                     "TCS": st.mean(tcs) if tcs else 0, "SCS": scs[0] if scs else 0,
                     "PRS": prs[0] if prs else 0, "TokenCost": st.mean(tc) if tc else 0,
                     "G1": st.mean(g1) if g1 else 0, "G2": st.mean(g2) if g2 else 0}

    print("\n" + "=" * 92)
    print("PCB RESULT — режим × метрика")
    print("=" * 92)
    cols = ["PCS", "CAS", "TCS", "SCS", "PRS", "G1", "G2", "TokenCost"]
    print(f"{'mode':<10}" + "".join(f"{c:>11}" for c in cols))
    for mode in MODES:
        r = res[mode]
        print(f"{mode:<10}" + "".join(f"{r[c]:>11.2f}" for c in cols))

    base_pcs = res["BASELINE"]["PCS"]
    print(f"\nКАНАРЕЙКА валидности: PCS(BASELINE)={base_pcs:.2f} "
          f"({'НИЗКИЙ — данные дискриминируют, ок' if base_pcs <= 0.5 else 'ВЫСОКИЙ — данные узнаваемы, ПЕРЕДЕЛАТЬ датасет!'})")

    t, s = res["TBG"], res["SUMMARY"]
    dyn_delta = (t["CAS"] + t["TCS"] + t["PRS"]) - (s["CAS"] + s["TCS"] + s["PRS"])
    c1 = t["PCS"] >= s["PCS"]
    c2 = dyn_delta > JUDGE_NOISE
    c3 = t["TokenCost"] <= s["TokenCost"] * 1.1
    verdict = "WIN" if (c1 and c2 and c3) else ("TIE" if abs(dyn_delta) <= JUDGE_NOISE else "FAIL")
    print(f"\nРЕШАЮЩЕЕ ПРАВИЛО (§7), JUDGE_NOISE={JUDGE_NOISE} (плейсхолдер до ручной K=20):")
    print(f"  1) PCS(TBG)>=PCS(SUMMARY): {t['PCS']:.2f}>={s['PCS']:.2f} -> {c1}")
    print(f"  2) dyn(CAS+TCS+PRS) delta > noise: {dyn_delta:+.2f} > {JUDGE_NOISE} -> {c2}")
    print(f"  3) TokenCost(TBG)<=SUMMARY*1.1: {t['TokenCost']:.0f}<={s['TokenCost']*1.1:.0f} -> {c3}")
    print(f"\n  ВЕРДИКТ: {verdict}")
    print(f"  calls потрачено: {_state['calls']}; audit -> {AUDIT.name}")
    print("  Оговорка: JUDGE_NOISE — плейсхолдер; для финала нужна ручная сверка K=20 (pcb_audit.jsonl).")


# ───────────────────────── main ─────────────────────────
async def main():
    args = sys.argv[1:]
    load_state()
    ds = load_ds()
    sessions = ds["sessions"]
    gt = ds["ground_truth"]

    dry = None
    if "--dry-run" in args:
        i = args.index("--dry-run")
        dry = int(args[i + 1]) if i + 1 < len(args) else 2
        sessions = sessions[:dry]
        checkpoints = [dry]
        for b in gt.values():  # в dry все блоки на финальном чек-поинте
            b["checkpoint"] = dry
    else:
        checkpoints = sorted(set(int(b.get("checkpoint")) for b in gt.values()))

    if "--report" in args:
        report(gt); return

    n_open = sum(1 for b in gt.values() if b not in ())  # rough
    print(f"PCB: sessions={len(sessions)} checkpoints={checkpoints} modes={MODES}")
    est = len(sessions) * 2 + len(checkpoints) * 3 * 8 + 40
    print(f"Оценка вызовов LLM ~{est} (cap PCB_MAX_CALLS={PCB_MAX_CALLS}); provider={os.environ.get('LLM_PROVIDER')}")
    hb("start")

    print("\n[PHASE 1] build memory ...", flush=True)
    await build_memory(sessions, checkpoints, dry)
    print("\n[PHASE 2] answers ...", flush=True)
    if not await answer_phase(sessions, checkpoints, gt):
        print("answers aborted (cap) — resume later"); return
    print("\n[PHASE 3] judge ...", flush=True)
    if not await judge_phase(gt):
        print("judge aborted (cap) — resume later"); return
    hb("DONE")
    report(gt)


if __name__ == "__main__":
    asyncio.run(main())
