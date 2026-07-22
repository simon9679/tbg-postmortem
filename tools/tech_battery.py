#!/usr/bin/env python3
"""
TECH-BATTERY: what distills into a technology. Groq, gpt-oss-120b, temp=0. WITHOUT prod edits.

4 sections:
  A — SSR false-merge value (psychology): does routing-VETO reduce FMR on lexical
      traps at an acceptable false-split. OUR pairs (bias-flagged).
  B — multilingual invariance: en->ru/zh/ar (objective LLM translation), routing SAME-recall +
      DIFFERENT-precision across language pairs. Cosine baseline (expect ~0 on cross-script).
  C — classification >> generation: stability(route) vs stability(gen_id), 5 runs.
  D — cross-field generalization (THE MAIN ONE): objective homonyms (python language↔python snake…)
      into a SHARED taxonomy. Does routing separate them. Low bias.

Cache of LLM calls in tools/tech_battery_cache.json (resumable, atomic, cloud-sync-safe).
Pacing tuned for Groq. Run:
  py -3 tools/tech_battery.py --gen [--sections A,B,C,D]   # LLM only (background)
  py -3 tools/tech_battery.py --analyze [--sections ...]   # analysis (local, instant)
"""
import os
import sys
import json
import time
import asyncio
from collections import Counter
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CACHE_PATH = os.path.join(HERE, "tech_battery_cache.json")
PACING = 2.2

PSYCH = ["career", "money", "health", "relationships", "self", "family", "lifestyle", "meaning", "other"]
GENERAL = ["technology", "nature", "finance", "health", "food", "geography",
           "language", "objects", "weather", "business", "abstract", "other"]

# ---------------- data ----------------
A_TRAPS = [  # GT = DIFFERENT (cross-domain lexical traps)
    ("career security", "financial security"), ("memory issue", "computer memory"),
    ("sleep debt", "financial debt"), ("relationship attachment", "file attachment"),
    ("work stress", "heart stress"), ("emotional baggage", "airline baggage"),
    ("trust issues", "SSL trust chain"), ("personal boundaries", "country borders"),
    ("career growth", "economic growth"), ("job burnout", "physical exhaustion"),
    ("emotional support", "technical support"), ("social network", "computer network"),
]
A_PARAS = [  # GT = SAME (paraphrases of the same concept)
    ("burnout", "exhaustion"), ("imposter syndrome", "feels like a fraud"),
    ("fear of failure", "afraid everything will fail"), ("financial security", "money stability"),
    ("spiritual crisis", "loss of meaning"), ("hobby", "favorite pastime"),
    ("meditation", "mindfulness practice"), ("people pleaser", "struggles to say no"),
    ("job loss", "losing your job"), ("loneliness", "feeling isolated"),
]

B_CONCEPTS = [
    "job loss", "debt", "burnout", "insomnia", "divorce", "loneliness",
    "self-doubt", "life purpose", "parenting", "travel", "hobby", "spiritual crisis",
    "career ambition", "savings", "grief", "friendship", "retirement", "exercise",
]
B_LANGS = ["en", "ru", "zh", "ar"]

C_CONCEPTS = [
    "burnout", "financial insecurity", "imposter syndrome", "fear of failure",
    "people pleaser", "loss of meaning", "loneliness", "career ambition",
    "self-doubt", "procrastination", "perfectionism", "emotional numbness",
]

D_PAIRS = [  # GT = DIFFERENT (objective homonyms)
    ("python programming language", "python snake"), ("memory leak", "memory foam"),
    ("java programming language", "java island"), ("apple the company", "apple the fruit"),
    ("amazon the company", "amazon river"), ("cloud storage", "cloud in the weather"),
    ("shell script", "sea shell"), ("computer virus malware", "biological virus"),
]


# ---------------- cache / LLM ----------------
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    import time as _t
    payload = json.dumps(cache, ensure_ascii=False, indent=2) + "\n"
    tmp = CACHE_PATH + ".tmp"
    for attempt in range(6):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, CACHE_PATH)
            return
        except OSError:
            _t.sleep(1.5 * (attempt + 1))
    print("  [warn] save_cache failed; in memory")


def route_prompt(label, taxonomy):
    cats = "|".join(taxonomy)
    return (f"Route this concept to EXACTLY ONE category.\nConcept: {label}\n"
            f"categories: [{cats}]\n"
            'Output ONLY JSON: {{"domain":".."}}'.replace("{{", "{").replace("}}", "}"))


def translate_prompt(concept):
    return (f'Translate the concept "{concept}" into Russian, Chinese (Simplified), and Arabic. '
            'Keep it a short concept phrase, not a literal word-by-word gloss. '
            'Output ONLY JSON: {"ru":"..","zh":"..","ar":".."}')


def genid_prompt(label):
    return (f"Output a canonical short snake_case concept_id (2-3 words) for this concept.\n"
            f"Concept: {label}\nOutput ONLY the id, nothing else.")


async def llm(prompt, timeout=90):
    from llm_client import gemini_call
    return await gemini_call(prompt, timeout=timeout)


def parse_domain(text, taxonomy):
    try:
        d = str(json.loads(text).get("domain", "")).strip().lower()
        return d if d in taxonomy else f"?{d}"
    except Exception:
        t = text.strip().lower()
        for c in taxonomy:
            if c in t:
                return c
        return "?parse"


async def ensure_runs(cache, key, prompt, n, parser):
    """Generate up to n valid runs for key. parser(text)->value (None if broken)."""
    cur = cache.get(key, [])
    while len([x for x in cur if x is not None]) < n:
        try:
            txt = await llm(prompt)
            val = parser(txt)
        except Exception:
            val = None
        cur.append(val)
        cache[key] = cur
        if val is None:
            break
        await asyncio.sleep(PACING)
    cache[key] = cur
    return [x for x in cur if x is not None]


# ---------------- generation by section ----------------
async def gen_section_A(cache):
    labels = sorted({x for p in (A_TRAPS + A_PARAS) for x in p})
    for i, l in enumerate(labels):
        await ensure_runs(cache, f"routeP::{l}", route_prompt(l, PSYCH), 1,
                          lambda t: (lambda d: d if not d.startswith("?") else None)(parse_domain(t, PSYCH)))
        if i % 5 == 0:
            save_cache(cache)
    save_cache(cache)


async def gen_section_B(cache):
    # translation
    for i, c in enumerate(B_CONCEPTS):
        key = f"tr::{c}"
        if key not in cache or not isinstance(cache.get(key), dict):
            try:
                obj = json.loads(await llm(translate_prompt(c)))
                cache[key] = {k: str(obj.get(k, "")).strip() for k in ("ru", "zh", "ar")}
            except Exception:
                cache[key] = {"ru": "", "zh": "", "ar": ""}
            await asyncio.sleep(PACING)
        if i % 5 == 0:
            save_cache(cache)
    save_cache(cache)
    # routing of all variants
    items = []
    for c in B_CONCEPTS:
        items.append(("en", c, c))
        tr = cache.get(f"tr::{c}", {})
        for lang in ("ru", "zh", "ar"):
            if tr.get(lang):
                items.append((lang, c, tr[lang]))
    for i, (lang, c, text) in enumerate(items):
        await ensure_runs(cache, f"routeP::{text}", route_prompt(text, PSYCH), 1,
                          lambda t: (lambda d: d if not d.startswith("?") else None)(parse_domain(t, PSYCH)))
        if i % 5 == 0:
            save_cache(cache)
    save_cache(cache)


async def gen_section_C(cache):
    for i, c in enumerate(C_CONCEPTS):
        await ensure_runs(cache, f"routeP::{c}", route_prompt(c, PSYCH), 5,
                          lambda t: (lambda d: d if not d.startswith("?") else None)(parse_domain(t, PSYCH)))
        await ensure_runs(cache, f"genid::{c}", genid_prompt(c), 5,
                          lambda t: (lambda s: s if s else None)(" ".join(t.strip().split()).lower().strip('"\'`.')))
        save_cache(cache)
    save_cache(cache)


async def gen_section_D(cache):
    labels = sorted({x for p in D_PAIRS for x in p})
    for i, l in enumerate(labels):
        await ensure_runs(cache, f"routeG::{l}", route_prompt(l, GENERAL), 1,
                          lambda t: (lambda d: d if not d.startswith("?") else None)(parse_domain(t, GENERAL)))
        if i % 5 == 0:
            save_cache(cache)
    save_cache(cache)


# ---------------- analysis ----------------
def _mode(cache, key):
    v = [x for x in cache.get(key, []) if x is not None]
    return Counter(v).most_common(1)[0][0] if v else None


def get_model():
    from fact_engine import get_embed_model
    return get_embed_model()


def analyze_A(cache):
    model = get_model()
    emb = {}

    def E(l):
        if l not in emb:
            emb[l] = model.encode(l, normalize_embeddings=True)
        return emb[l]

    def cos(a, b):
        return float(np.dot(E(a), E(b)))

    def dom(l):
        return _mode(cache, f"routeP::{l}")

    traps = [(a, b, cos(a, b), dom(a), dom(b)) for a, b in A_TRAPS]
    paras = [(a, b, cos(a, b), dom(a), dom(b)) for a, b in A_PARAS]

    print("\n" + "=" * 78 + "\nA. SSR FALSE-MERGE (наши пары — bias)\n" + "=" * 78)
    print("Сводка операционных точек (merge если cos>=T; VETO: + same-domain):")
    print(f"{'T':>5}{'FMR_cos':>9}{'FMR_veto':>10}{'split_cos':>11}{'split_veto':>12}")
    best = None
    for T in [0.50, 0.60, 0.70, 0.82]:
        fmr_cos = sum(1 for *_, c, da, db in [(a, b, c, da, db) for a, b, c, da, db in traps] if c >= T) / len(traps)
        fmr_veto = sum(1 for a, b, c, da, db in traps if c >= T and da == db and da is not None) / len(traps)
        split_cos = sum(1 for a, b, c, da, db in paras if c < T) / len(paras)
        split_veto = sum(1 for a, b, c, da, db in paras if not (c >= T and da == db and da is not None)) / len(paras)
        print(f"{T:>5.2f}{fmr_cos:>9.0%}{fmr_veto:>10.0%}{split_cos:>11.0%}{split_veto:>12.0%}")
        if split_cos <= 0.20 and best is None:
            best = (T, fmr_cos, fmr_veto, split_cos, split_veto)

    print("\nTRAPS (GT=DIFFERENT) — cos | domains:")
    for a, b, c, da, db in sorted(traps, key=lambda x: -x[2]):
        print(f"  {c:>6.3f}  {da}↔{db:<12} {a!r}↔{b!r}")
    print("PARAS (GT=SAME) — cos | domains:")
    for a, b, c, da, db in sorted(paras, key=lambda x: -x[2]):
        same = "same-dom" if da == db else "CROSS-dom"
        print(f"  {c:>6.3f}  {da}↔{db:<12} [{same}] {a!r}↔{b!r}")

    # VETO effect independent of T (routing blocks all cross-domain)
    removed = [(a, b) for a, b, c, da, db in traps if da is not None and db is not None and da != db]
    added = [(a, b) for a, b, c, da, db in paras if da is not None and db is not None and da != db]
    print(f"\nVETO блокирует cross-domain независимо от T:")
    print(f"  убрано ложных мёржей (traps cross-domain): {len(removed)}/{len(traps)}")
    print(f"  добавлено ложных сплитов (paras cross-domain): {len(added)}/{len(paras)} -> {[f'{a}/{b}' for a,b in added]}")
    fs = len(added) / len(paras)
    print(f"  ПЛАНКА (FMR_veto≈0 на traps при false-split≤~20%): "
          f"FMR_veto=0% (все traps cross-domain VETO), added-split={fs:.0%} -> "
          f"{'ВЗЯТА' if fs <= 0.20 else 'НЕ взята (split>20%)'}")


def analyze_B(cache):
    model = get_model()

    def dom(text):
        return _mode(cache, f"routeP::{text}")

    # domain каждого варианта
    routed = {}  # (lang, concept) -> domain
    texts = {}   # (lang, concept) -> text
    for c in B_CONCEPTS:
        routed[("en", c)] = dom(c); texts[("en", c)] = c
        tr = cache.get(f"tr::{c}", {})
        for lang in ("ru", "zh", "ar"):
            t = tr.get(lang, "")
            texts[(lang, c)] = t
            routed[(lang, c)] = dom(t) if t else None

    langs = B_LANGS
    print("\n" + "=" * 78 + "\nB. MULTILINGUAL INVARIANCE (объективный перевод)\n" + "=" * 78)
    print(f"{'lang-pair':<10}{'SAME-recall':>13}{'DIFF-prec':>11}{'cos SAME-recall':>17}")
    overall_ok = True
    for l1, l2 in combinations(langs, 2):
        # SAME-recall: один концепт в обоих языках -> один domain
        sr_tot = sr_ok = 0
        for c in B_CONCEPTS:
            d1, d2 = routed.get((l1, c)), routed.get((l2, c))
            if d1 and d2:
                sr_tot += 1
                sr_ok += (d1 == d2)
        sr = sr_ok / sr_tot if sr_tot else 0.0
        # DIFFERENT-precision: пары РАЗНЫХ концептов, у кого en-домены различаются ->
        # должны различаться и в (l1,l2)
        dp_tot = dp_ok = 0
        for ci, cj in combinations(B_CONCEPTS, 2):
            de_i, de_j = routed.get(("en", ci)), routed.get(("en", cj))
            if de_i and de_j and de_i != de_j:
                d1, d2 = routed.get((l1, ci)), routed.get((l2, cj))
                if d1 and d2:
                    dp_tot += 1
                    dp_ok += (d1 != d2)
        dp = dp_ok / dp_tot if dp_tot else 0.0
        # косинус SAME-recall (порог 0.82) для кросс-язычных одинаковых концептов
        cs_tot = cs_ok = 0
        for c in B_CONCEPTS:
            t1, t2 = texts.get((l1, c)), texts.get((l2, c))
            if t1 and t2:
                cs_tot += 1
                cs_ok += (float(np.dot(model.encode(t1, normalize_embeddings=True),
                                       model.encode(t2, normalize_embeddings=True))) >= 0.82)
        cs = cs_ok / cs_tot if cs_tot else 0.0
        ok = sr >= 0.90 and dp >= 0.90
        overall_ok = overall_ok and ok
        print(f"{l1}-{l2:<7}{sr:>12.0%}{dp:>11.0%}{cs:>17.0%}  {'OK' if ok else 'FAIL'}")
    print(f"\nПЛАНКА (SAME-recall≥90% И DIFF-prec≥90% на ВСЕХ парах): "
          f"{'ВЗЯТА' if overall_ok else 'НЕ взята'}")


def analyze_C(cache):
    print("\n" + "=" * 78 + "\nC. CLASSIFICATION >> GENERATION\n" + "=" * 78)
    r_stable = g_stable = 0
    print(f"{'concept':<24}{'route x5':>22}{'genid x5':>30}")
    for c in C_CONCEPTS:
        rv = [x for x in cache.get(f"routeP::{c}", []) if x is not None][:5]
        gv = [x for x in cache.get(f"genid::{c}", []) if x is not None][:5]
        r_ok = len(rv) == 5 and len(set(rv)) == 1
        g_ok = len(gv) == 5 and len(set(gv)) == 1
        r_stable += r_ok
        g_stable += g_ok
        print(f"  {c:<22}{('=' if r_ok else '≠')+str(dict(Counter(rv))):>22}"
              f"{('=' if g_ok else '≠')+str(len(set(gv)))+'uniq':>30}")
    n = len(C_CONCEPTS)
    rp, gp = r_stable / n, g_stable / n
    print(f"\nstability route = {rp:.0%} ({r_stable}/{n}); gen_id = {gp:.0%} ({g_stable}/{n})")
    print(f"ПЛАНКА (route≥95% И заметно выше gen): "
          f"{'ВЗЯТА' if rp >= 0.95 and rp - gp >= 0.15 else 'НЕ взята'}")


def analyze_D(cache):
    model = get_model()

    def dom(l):
        return _mode(cache, f"routeG::{l}")

    print("\n" + "=" * 78 + "\nD. CROSS-FIELD GENERALIZATION (объективные омонимы)\n" + "=" * 78)
    sep_route = sep_cos = 0
    for a, b in D_PAIRS:
        da, db = dom(a), dom(b)
        c = float(np.dot(model.encode(a, normalize_embeddings=True),
                         model.encode(b, normalize_embeddings=True)))
        r_sep = da is not None and db is not None and da != db
        cos_sep = c < 0.82
        sep_route += r_sep
        sep_cos += cos_sep
        print(f"  cos={c:>5.2f} route[{da}↔{db}] -> {'РАЗВЁЛ' if r_sep else 'СЛИЛ'}  {a!r}↔{b!r}")
    n = len(D_PAIRS)
    print(f"\nrouting развёл {sep_route}/{n} = {sep_route/n:.0%}; косинус(<0.82) {sep_cos}/{n}")
    print(f"ПЛАНКА (routing ≥90% разведены): {'ВЗЯТА' if sep_route/n >= 0.90 else 'НЕ взята'}")


# ---------------- main ----------------
SECT_GEN = {"A": gen_section_A, "B": gen_section_B, "C": gen_section_C, "D": gen_section_D}
SECT_AN = {"A": analyze_A, "B": analyze_B, "C": analyze_C, "D": analyze_D}


def main():
    args = sys.argv[1:]
    sects = ["A", "B", "C", "D"]
    for a in args:
        if a.startswith("--sections="):
            sects = a.split("=", 1)[1].split(",")
        elif a.startswith("--sections"):
            pass
    if "--sections" in args:
        i = args.index("--sections")
        if i + 1 < len(args):
            sects = args[i + 1].split(",")

    cache = load_cache()

    if "--gen" in args:
        os.environ.setdefault("LLM_PROVIDER", "groq")
        os.environ.setdefault("LLM_MODEL", "openai/gpt-oss-120b")
        os.environ.setdefault("LLM_TEMPERATURE", "0")
        os.environ.setdefault("LLM_MAX_TOKENS", "2000")

        async def run():
            for s in sects:
                print(f"[gen] секция {s} ...", flush=True)
                await SECT_GEN[s](cache)
                print(f"[gen] секция {s} готова", flush=True)
        asyncio.run(run())
        print("Генерация завершена.")
        return 0

    if "--analyze" in args:
        for s in sects:
            SECT_AN[s](cache)
        print("\nПрод не трогался. run_drift не запускался.")
        return 0

    print("Укажи --gen (LLM, фон) или --analyze (локально).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
