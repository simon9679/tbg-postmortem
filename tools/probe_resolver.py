#!/usr/bin/env python3
"""
E3a-3 (финал) — concept-identity resolver probe. БЕЗ правок прода.

Вопрос: можно ли РАЗДЕЛИТЬ работу эмбеддера на две независимые?
  (A) ТЕМА/тождество концепта — решает косинус (e5) ПО ГЛОССУ, а не по голому лейблу.
  (B) ПОЛЯРНОСТЬ/оппозиция — решает EPA (детерм.) и/или NLI, отдельно от косинуса.

Глосс = нейтральное англ. предложение про концепт, сгенерённое LLM
(cerebras gpt-oss-120b, temp=0), 5 ранов для проверки стабильности.

Метрики на пару:
  Q1 ТЕМА:       cos_label  = e5(label_a, label_b)        [контраст, уже знаем]
                 cos_gloss  = e5(gloss_a, gloss_b)         [главный: разделяет ли глосс?]
  Q2 ПОЛЯРНОСТЬ: epa_dot    = dot(EPA(label_a), EPA(label_b))  [знак - => оппозиция]
                 nli_label  = NLI(label_a, label_b)  -> relation + contradiction-score
                 nli_gloss  = NLI(gloss_a, gloss_b)  -> relation + contradiction-score
  Q3 СТАБИЛЬНОСТЬ: по 5 ранам глоссов — флипает ли класс решения
                 (nli_gloss relation; cos_gloss band по провизорным порогам).
  + латентность NLI, мс/пара.

e5: симметричная STS — ОБА текста префиксятся "query: ".
NLI: cross-encoder/nli-deberta-v3-small, биднаправленно (max по направлениям).
EPA: tbg_axes.get_belief_axes() (MiniLM, детерминирован).

ЗАПУСК:
  py -3 tools/probe_resolver.py                 # глоссы (если нет) + анализ
  py -3 tools/probe_resolver.py --gloss-only    # только сгенерить глоссы
  py -3 tools/probe_resolver.py --analyze-only  # только анализ (глоссы должны быть)
  py -3 tools/probe_resolver.py --regloss       # пересгенерить все глоссы
Глоссы кладутся в calib_pairs_v2.json -> "_glosses" (резюмируемо, по-лейблово).
"""
import os
import sys
import json
import time
import asyncio

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# Консоль Windows = cp1251, ломается на не-кириллических символах (× и т.п.).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PAIRS_PATH = os.path.join(HERE, "calib_pairs_v2.json")
OUT_PATH = os.path.join(HERE, "probe_resolver_out.json")

CLASSES = ("paraphrase", "affective_opposite", "structural_opposite", "adjacent", "unrelated")
# Для эмбеддера «тот же топик» = парафраз + любая оппозиция (полярность — не его работа).
MERGE_GROUP = ("paraphrase", "affective_opposite", "structural_opposite")

N_RUNS = 5
GLOSS_PROMPT = (
    "Describe this psychological/behavioral concept as ONE neutral English sentence "
    "stating what the person prefers, feels, or does. Concept: {label}. "
    "Output only the sentence."
)

E5_NAME = "intfloat/multilingual-e5-small"
NLI_NAME = "cross-encoder/nli-deberta-v3-small"
# Cerebras free tier = 30 RPM. Бьём строго последовательно с паузой,
# иначе параллельные вызовы выжирают лимит и ретраи 429 истощаются.
GLOSS_CONCURRENCY = 1
PACING_SEC = 2.2


def _valid_runs(runs):
    """Только успешные глоссы (без __ERROR__ ячеек)."""
    return [r for r in runs if r and not str(r).startswith("__ERROR__")]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_data():
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    """Atomic write with retries: a cloud-synced folder occasionally locks the
    file during sync — without retries this crashed the whole run."""
    import time as _t
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp = PAIRS_PATH + ".tmp"
    last = None
    for attempt in range(6):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, PAIRS_PATH)
            return True
        except OSError as e:  # PermissionError (cloud-sync lock) etc.
            last = e
            _t.sleep(1.5 * (attempt + 1))
    print(f"  [warn] save_data failed after retries: {last}; continuing (data in memory)")
    return False


def unique_labels(data):
    seen = []
    for cls in CLASSES:
        for a, b in data[cls]:
            for x in (a, b):
                if x not in seen:
                    seen.append(x)
    return seen


# ---------------------------------------------------------------------------
# Глоссы (LLM, cerebras gpt-oss-120b temp=0). Резюмируемо.
# ---------------------------------------------------------------------------
async def ensure_glosses(data, force=False):
    os.environ.setdefault("LLM_PROVIDER", "cerebras")
    os.environ.setdefault("LLM_MODEL", "gpt-oss-120b")
    os.environ.setdefault("LLM_TEMPERATURE", "0")
    os.environ.setdefault("LLM_MAX_TOKENS", "3000")
    from llm_client import gemini_call

    glosses = {} if force else dict(data.get("_glosses", {}))
    labels = unique_labels(data)
    # __ERROR__ ячейки считаем НЕзаполненными -> их перегенерим.
    todo = [l for l in labels if len(_valid_runs(glosses.get(l, []))) < N_RUNS]
    print(f"Уникальных лейблов: {len(labels)}; нужно догенерить: {len(todo)} "
          f"(до {N_RUNS} валидных ранов; concurrency={GLOSS_CONCURRENCY}, pacing={PACING_SEC}s)")
    if not todo:
        print("Все глоссы уже есть.")
        return glosses

    done = {"n": 0}

    async def one(label):
        runs = _valid_runs(glosses.get(label, []))  # стартуем с уже валидных
        while len(runs) < N_RUNS:
            try:
                g = await gemini_call(GLOSS_PROMPT.format(label=label), timeout=90)
                g = " ".join(str(g).split()).strip()
                if g and not g.startswith("__ERROR__"):
                    runs.append(g)
                else:
                    runs.append("__ERROR__:empty")
                    break
            except Exception as e:
                runs.append(f"__ERROR__:{type(e).__name__}")
                break
            await asyncio.sleep(PACING_SEC)
        glosses[label] = runs
        done["n"] += 1
        if done["n"] % 3 == 0 or done["n"] == len(todo):
            data["_glosses"] = glosses
            save_data(data)
            print(f"  ... {done['n']}/{len(todo)} лейблов (сохранено)")

    # строго последовательно (concurrency=1) — самый надёжный режим под 30 RPM
    for lbl in todo:
        await one(lbl)
    data["_glosses"] = glosses
    save_data(data)
    n_err = sum(1 for v in glosses.values() if len(_valid_runs(v)) < N_RUNS)
    print(f"Готово. Лейблов с неполными глоссами: {n_err}")
    return glosses


# ---------------------------------------------------------------------------
# Анализ
# ---------------------------------------------------------------------------
def stats(vals):
    a = np.array(vals, dtype=float)
    return float(a.min()), float(a.mean()), float(a.max())


def main_analyze(data, glosses):
    from sentence_transformers import SentenceTransformer
    from tbg_axes import get_belief_axes
    from tbg_nli import NLIContradictionDetector

    print(f"\nЗагрузка e5 ({E5_NAME}) ...")
    e5 = SentenceTransformer(E5_NAME)

    def e5cos(a, b):
        emb = e5.encode([f"query: {a}", f"query: {b}"], normalize_embeddings=True, show_progress_bar=False)
        return float(emb[0] @ emb[1])

    print("Инициализация EPA-осей (MiniLM) ...")
    axes = get_belief_axes()
    axes.init()

    def epa_dot(a, b):
        pa, pb = axes.project_batch([a, b])
        va = np.array([pa[k] for k in pa])
        vb = np.array([pb[k] for k in pb])
        return float(np.dot(va, vb))

    def epa_eval(a, b):
        """Знаковый сигнал оппозиции по оси evaluation — то, чем реально
        оперирует прод _is_opposition (а не dot всех осей).
        eval_prod < 0  => противоположная валентность (оппозиция)."""
        pa, pb = axes.project_batch([a, b])
        return float(pa["evaluation"] * pb["evaluation"])

    print(f"Загрузка NLI ({NLI_NAME}) — первая загрузка ~280МБ ...")
    nli = NLIContradictionDetector(model_name=NLI_NAME)

    nli_times = []

    def nli_rel(a, b):
        t0 = time.perf_counter()
        ab = nli.score_pair(a, b)
        ba = nli.score_pair(b, a)
        nli_times.append((time.perf_counter() - t0) * 1000.0)
        contra = max(ab["contradiction"], ba["contradiction"])
        entail = max(ab["entailment"], ba["entailment"])
        neutral = max(ab["neutral"], ba["neutral"])
        rel = max((("contradiction", contra), ("entailment", entail), ("neutral", neutral)),
                  key=lambda x: x[1])[0]
        return rel, contra

    def g(label, run):
        runs = _valid_runs(glosses.get(label, []))
        if not runs:
            return label  # глосса нет — деградируем к лейблу (помечаем в отчёте)
        return runs[run % len(runs)]

    # --- собрать метрики по парам -----------------------------------------
    per_class = {cls: {"cos_label": [], "cos_gloss": [], "epa_dot": [], "epa_eval": [],
                       "nli_label": [], "nli_gloss": []} for cls in CLASSES}
    rel_breakdown = {cls: {"label": {}, "gloss": {}} for cls in CLASSES}
    pair_records = []  # для стабильности

    for cls in CLASSES:
        for a, b in data[cls]:
            cl = e5cos(a, b)
            ga1, gb1 = g(a, 0), g(b, 0)
            cg = e5cos(ga1, gb1)
            ed = epa_dot(a, b)
            ee = epa_eval(a, b)
            rl, sl = nli_rel(a, b)
            rg, sg = nli_rel(ga1, gb1)

            per_class[cls]["cos_label"].append(cl)
            per_class[cls]["cos_gloss"].append(cg)
            per_class[cls]["epa_dot"].append(ed)
            per_class[cls]["epa_eval"].append(ee)
            per_class[cls]["nli_label"].append(sl)
            per_class[cls]["nli_gloss"].append(sg)
            rel_breakdown[cls]["label"][rl] = rel_breakdown[cls]["label"].get(rl, 0) + 1
            rel_breakdown[cls]["gloss"][rg] = rel_breakdown[cls]["gloss"].get(rg, 0) + 1

            pair_records.append({"cls": cls, "a": a, "b": b})

    # --- стабильность по 5 ранам ------------------------------------------
    # провизорные пороги по cos_gloss run-1 (для band-flip)
    mg = []
    for cls in MERGE_GROUP:
        mg += per_class[cls]["cos_gloss"]
    adj = per_class["adjacent"]["cos_gloss"]
    unr = per_class["unrelated"]["cos_gloss"]
    MERGE_g = (max(adj) + min(mg)) / 2.0
    FLAG_g = (max(unr) + min(adj)) / 2.0

    def band(c):
        return "MERGE" if c >= MERGE_g else ("FLAG" if c >= FLAG_g else "NONE")

    rel_flips = 0
    band_flips = 0
    cos_stds = []
    for rec in pair_records:
        a, b = rec["a"], rec["b"]
        rels, bands, coses = [], [], []
        for r in range(N_RUNS):
            ga, gb = g(a, r), g(b, r)
            cg = e5cos(ga, gb)
            coses.append(cg)
            bands.append(band(cg))
            rr, _ = nli_rel(ga, gb)
            rels.append(rr)
        if len(set(rels)) > 1:
            rel_flips += 1
        if len(set(bands)) > 1:
            band_flips += 1
        cos_stds.append(float(np.std(coses)))

    n_pairs = len(pair_records)
    rel_flip_rate = rel_flips / n_pairs
    band_flip_rate = band_flips / n_pairs

    # --- seps --------------------------------------------------------------
    def seps(metric):
        mg_lo = min(min(per_class[c][metric]) for c in MERGE_GROUP)
        sep_merge = mg_lo - max(per_class["adjacent"][metric])
        sep_flag = min(per_class["adjacent"][metric]) - max(per_class["unrelated"][metric])
        return sep_merge, sep_flag

    sm_lbl, sf_lbl = seps("cos_label")
    sm_gls, sf_gls = seps("cos_gloss")

    # =====================================================================
    # ОТЧЁТ
    # =====================================================================
    print("\n" + "=" * 78)
    print("ТАБЛИЦА: класс x метрика (min / mean / max)")
    print("=" * 78)
    cols = ["cos_label", "cos_gloss", "epa_dot", "nli_label", "nli_gloss"]
    head = f"{'класс':<20}" + "".join(f"{c:>22}" for c in cols)
    print(head)
    print("-" * len(head))
    for cls in CLASSES:
        row = f"{cls:<20}"
        for c in cols:
            mn, me, mx = stats(per_class[cls][c])
            row += f"  {mn:>5.2f}/{me:>5.2f}/{mx:>5.2f}"
        print(row)

    print("\nNLI relation breakdown (по лейблам | по глоссам run-1):")
    for cls in CLASSES:
        lb = ", ".join(f"{k}:{v}" for k, v in sorted(rel_breakdown[cls]["label"].items()))
        gb = ", ".join(f"{k}:{v}" for k, v in sorted(rel_breakdown[cls]["gloss"].items()))
        print(f"  {cls:<20} label[{lb}]  ||  gloss[{gb}]")

    print("\nEPA по оси evaluation (eval_prod = proj_a.eval * proj_b.eval; <0 => оппозиция валентности):")
    for cls in CLASSES:
        mn, me, mx = stats(per_class[cls]["epa_eval"])
        neg = sum(1 for x in per_class[cls]["epa_eval"] if x < 0)
        print(f"  {cls:<20} mean={me:+.4f}  min={mn:+.4f}  max={mx:+.4f}  "
              f"(пар с eval_prod<0: {neg}/{len(per_class[cls]['epa_eval'])})")

    print("\nРАЗДЕЛИМОСТЬ (топическая, merge-group = paraphrase∪affective∪structural):")
    print(f"  по cos_label : sep_MERGE={sm_lbl:+.3f}  sep_FLAG={sf_lbl:+.3f}")
    print(f"  по cos_gloss : sep_MERGE={sm_gls:+.3f}  sep_FLAG={sf_gls:+.3f}")

    print(f"\nСТАБИЛЬНОСТЬ по {N_RUNS} ранам глоссов ({n_pairs} пар):")
    print(f"  nli_gloss relation-flip-rate = {rel_flip_rate:.3f}  ({rel_flips}/{n_pairs} пар сменили класс)")
    print(f"  cos_gloss band-flip-rate     = {band_flip_rate:.3f}  ({band_flips}/{n_pairs}; "
          f"провизорные пороги MERGE_g={MERGE_g:.3f}, FLAG_g={FLAG_g:.3f})")
    print(f"  cos_gloss std по ранам: mean={np.mean(cos_stds):.4f}, max={np.max(cos_stds):.4f}")

    if nli_times:
        print(f"\nЛАТЕНТНОСТЬ NLI: {np.mean(nli_times):.1f} мс/пара "
              f"(2 прохода/пара, n={len(nli_times)}), max={np.max(nli_times):.1f} мс")

    # --- явные ответы ------------------------------------------------------
    print("\n" + "=" * 78)
    print("ОТВЕТЫ")
    print("=" * 78)
    print(f"1) Глосс разделяет тему лучше лейбла? "
          f"sep_MERGE: {sm_lbl:+.3f} -> {sm_gls:+.3f}; sep_FLAG: {sf_lbl:+.3f} -> {sf_gls:+.3f}. "
          f"{'ДА' if (sm_gls > sm_lbl and sm_gls > 0) else 'не полностью' if sm_gls > sm_lbl else 'НЕТ'}")

    def evalstat(cls):
        vals = per_class[cls]["epa_eval"]
        return float(np.mean(vals)), sum(1 for x in vals if x < 0), len(vals)
    aff_m, aff_neg, aff_n = evalstat("affective_opposite")
    str_m, str_neg, str_n = evalstat("structural_opposite")
    par_m, par_neg, par_n = evalstat("paraphrase")
    epa_aff = stats(per_class["affective_opposite"]["epa_dot"])
    print(f"2) EPA (правильный показатель — eval_prod по оси evaluation): "
          f"affective mean={aff_m:+.4f} (оппозиций {aff_neg}/{aff_n}); "
          f"structural mean={str_m:+.4f} ({str_neg}/{str_n}); "
          f"paraphrase mean={par_m:+.4f} ({par_neg}/{par_n}). "
          f"[справочно epa_dot всех осей у affective={epa_aff[1]:+.3f} — вырожден ~0]. "
          f"EPA {'ловит' if aff_neg > par_neg and aff_m < par_m else 'НЕ ловит чисто'} affective; "
          f"structural {'молчит' if str_neg <= aff_neg else 'тоже сигналит'}")

    nlg_str = rel_breakdown["structural_opposite"]["gloss"]
    nll_str = rel_breakdown["structural_opposite"]["label"]
    print(f"3) NLI structural_opposite: label{dict(nll_str)} -> gloss{dict(nlg_str)}. "
          f"contradiction по глоссам {'вытягивается' if nlg_str.get('contradiction',0) > nll_str.get('contradiction',0) else 'не лучше лейблов'}")

    print(f"4) Стабильность: nli relation-flip-rate={rel_flip_rate:.3f}, "
          f"cos band-flip-rate={band_flip_rate:.3f}. "
          f"{'конвейер стабилен' if rel_flip_rate <= 0.1 and band_flip_rate <= 0.1 else 'есть дрейф — глоссы шумят'}")

    if nli_times:
        print(f"5) NLI: {np.mean(nli_times):.1f} мс/пара (CPU, 2 прохода).")

    # --- сохранить сырьё ---------------------------------------------------
    out = {
        "per_class_stats": {cls: {c: stats(per_class[cls][c]) for c in cols} for cls in CLASSES},
        "rel_breakdown": rel_breakdown,
        "seps": {"cos_label": [sm_lbl, sf_lbl], "cos_gloss": [sm_gls, sf_gls]},
        "stability": {"nli_relation_flip_rate": rel_flip_rate,
                      "cos_band_flip_rate": band_flip_rate,
                      "provisional_thresholds": {"MERGE_g": MERGE_g, "FLAG_g": FLAG_g},
                      "cos_std_mean": float(np.mean(cos_stds))},
        "nli_latency_ms_per_pair": float(np.mean(nli_times)) if nli_times else None,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nСырьё сохранено: {OUT_PATH}")
    print("Прод не трогался. run_drift не запускался.")


def main():
    args = sys.argv[1:]
    data = load_data()

    if "--analyze-only" in args:
        glosses = data.get("_glosses", {})
        if not glosses:
            print("Нет глоссов в calib_pairs_v2.json. Сначала без --analyze-only.")
            return 1
        main_analyze(data, glosses)
        return 0

    force = "--regloss" in args
    glosses = asyncio.run(ensure_glosses(data, force=force))

    if "--gloss-only" in args:
        print("Глоссы готовы (--gloss-only).")
        return 0

    main_analyze(data, glosses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
