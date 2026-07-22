#!/usr/bin/env python3
"""
L1 probe: domain-gated routing. WITHOUT prod edits. Offline.

Idea (after closing L0): don't measure pairwise label similarity (impossible — the E3 series),
but ROUTE each concept into a discrete cell (domain × stance) via LLM. Then
merge/conflict are decided WITHIN a cell, not by a global cosine.

Taxonomy — standard wheel-of-life, NOT tuned to DriftBench:
  domain ∈ {career, money, health, relationships, self, family, lifestyle, meaning, other}
  stance ∈ {approach, avoid, neutral}

Router (cerebras gpt-oss-120b, temp=0), 5 runs per label.

BARS (fixed BEFORE the run):
  1. domain STABILITY: % of labels with the same domain across all 5 runs. BAR ≥90%.
     (stance — separately, softer threshold, for reference.)
  2. SAFETY: catastrophic pairs -> DIFFERENT domain.
     career security vs financial security; mortgage payments vs doctor partner. BAR: separated.
  3. RECALL: paraphrases -> ONE cell (domain+stance).
     burnout vs exhaustion; imposter syndrome vs feels like a fraud. BAR: co-located.
  4. Borderline leak: list of labels that float across domain between runs.

RUN:
  py -3 tools/probe_routing.py               # routing (if absent) + analysis
  py -3 tools/probe_routing.py --route-only
  py -3 tools/probe_routing.py --analyze-only
"""
import os
import sys
import json
import asyncio
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PAIRS_PATH = os.path.join(HERE, "calib_pairs_v2.json")
RUNS_PATH = os.path.join(HERE, "routing_runs.json")
CEREBRAS24_PATH = os.path.join(HERE, "routing_runs_cerebras24.json")
CLASSES = ("paraphrase", "affective_opposite", "structural_opposite", "adjacent", "unrelated")

DOMAINS = ["career", "money", "health", "relationships", "self", "family", "lifestyle", "meaning", "other"]
STANCES = ["approach", "avoid", "neutral"]
N_RUNS = 5
PACING_SEC = 2.2

HARDCASES = [
    "financial security", "career security", "wants promotion", "wants stability", "wants change",
    "spiritual crisis", "духовный кризис", "hobby", "хобби", "fear of boss",
    "work-life balance", "side hustle", "caring for aging parents", "meditation practice",
    "saving for retirement",
]

ROUTER_PROMPT = (
    "Route this concept to EXACTLY ONE domain and ONE stance.\n"
    "Concept: {label}\n"
    "domain: [career|money|health|relationships|self|family|lifestyle|meaning|other]\n"
    "stance: [approach|avoid|neutral]\n"
    'Output ONLY JSON: {{"domain":"..","stance":".."}}'
)

# Pairs for SAFETY/RECALL checks.
SAFETY_PAIRS = [("career security", "financial security"), ("mortgage payments", "doctor partner")]
RECALL_PAIRS = [("burnout", "exhaustion"), ("imposter syndrome", "feels like a fraud")]


# --------------------------------------------------------------------------
def load_labels():
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    seen = []
    for c in CLASSES:
        for a, b in d[c]:
            for x in (a, b):
                if x not in seen:
                    seen.append(x)
    for h in HARDCASES:
        if h not in seen:
            seen.append(h)
    return seen


def load_runs():
    if os.path.exists(RUNS_PATH):
        with open(RUNS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_runs(runs):
    """Atomic + retries (a cloud-synced folder locks the file during sync)."""
    import time as _t
    payload = json.dumps(runs, ensure_ascii=False, indent=2) + "\n"
    tmp = RUNS_PATH + ".tmp"
    for attempt in range(6):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, RUNS_PATH)
            return True
        except OSError:
            _t.sleep(1.5 * (attempt + 1))
    print("  [warn] save_runs не удался; данные в памяти")
    return False


def _valid(runs_for_label):
    out = []
    for r in runs_for_label or []:
        if isinstance(r, dict) and r.get("domain") in DOMAINS and r.get("stance") in STANCES:
            out.append(r)
    return out


def parse_route(text):
    try:
        obj = json.loads(text)
        dom = str(obj.get("domain", "")).strip().lower()
        st = str(obj.get("stance", "")).strip().lower()
        if dom in DOMAINS and st in STANCES:
            return {"domain": dom, "stance": st}
        return {"domain": dom or "?", "stance": st or "?", "_invalid": True}
    except Exception as e:
        return {"domain": "?", "stance": "?", "_error": type(e).__name__}


async def ensure_routes(force=False):
    os.environ.setdefault("LLM_PROVIDER", "cerebras")
    os.environ.setdefault("LLM_MODEL", "gpt-oss-120b")
    os.environ.setdefault("LLM_TEMPERATURE", "0")
    os.environ.setdefault("LLM_MAX_TOKENS", "3000")
    from llm_client import gemini_call

    labels = load_labels()
    runs = {} if force else load_runs()
    todo = [l for l in labels if len(_valid(runs.get(l, []))) < N_RUNS]
    print(f"Лейблов: {len(labels)}; догенерить роуты: {len(todo)} "
          f"(до {N_RUNS} валидных; pacing={PACING_SEC}s)")
    if not todo:
        print("Все роуты есть.")
        return runs

    done = 0
    for lbl in todo:
        cur = _valid(runs.get(lbl, []))
        while len(cur) < N_RUNS:
            try:
                resp = await gemini_call(ROUTER_PROMPT.format(label=lbl), timeout=90)
                r = parse_route(resp)
            except Exception as e:
                r = {"domain": "?", "stance": "?", "_error": type(e).__name__}
            if r.get("domain") in DOMAINS and r.get("stance") in STANCES:
                cur.append(r)
            else:
                cur.append(r)
                break  # broken response — don't loop; we'll top up on the next run
            await asyncio.sleep(PACING_SEC)
        runs[lbl] = cur
        done += 1
        if done % 5 == 0 or done == len(todo):
            save_runs(runs)
            print(f"  ... {done}/{len(todo)} (сохранено)")
    save_runs(runs)
    return runs


# --------------------------------------------------------------------------
def majority(items):
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def analyze(runs):
    labels = load_labels()
    full = {l: _valid(runs.get(l, [])) for l in labels}
    complete = {l: v for l, v in full.items() if len(v) >= N_RUNS}
    incomplete = [l for l in labels if len(full[l]) < N_RUNS]

    # --- 1. stability ---
    dom_stable = [l for l, v in complete.items()
                  if len(set(r["domain"] for r in v[:N_RUNS])) == 1]
    st_stable = [l for l, v in complete.items()
                 if len(set(r["stance"] for r in v[:N_RUNS])) == 1]
    cell_stable = [l for l, v in complete.items()
                   if len(set((r["domain"], r["stance"]) for r in v[:N_RUNS])) == 1]
    n = len(complete)
    dom_pct = 100.0 * len(dom_stable) / n if n else 0.0
    st_pct = 100.0 * len(st_stable) / n if n else 0.0
    cell_pct = 100.0 * len(cell_stable) / n if n else 0.0

    def maj_dom(l):
        return majority([r["domain"] for r in full.get(l, [])]) if full.get(l) else None

    def maj_cell(l):
        v = full.get(l, [])
        return majority([(r["domain"], r["stance"]) for r in v]) if v else None

    # --- 4. borderline leak ---
    leaky = []
    for l, v in complete.items():
        doms = [r["domain"] for r in v[:N_RUNS]]
        if len(set(doms)) > 1:
            leaky.append((l, dict(Counter(doms))))

    print("\n" + "=" * 80)
    print(f"РОУТИНГ: {len(labels)} лейблов, полных (5 ранов): {n}, неполных: {len(incomplete)}")
    if incomplete:
        print(f"  неполные: {incomplete}")
    print("=" * 80)

    print(f"\n1) СТАБИЛЬНОСТЬ (по {n} полным лейблам):")
    print(f"   domain одинаков во всех 5: {len(dom_stable)}/{n} = {dom_pct:.1f}%   "
          f"[ПЛАНКА ≥90% -> {'ВЗЯТА' if dom_pct >= 90 else 'НЕ взята'}]")
    print(f"   stance одинаков во всех 5: {len(st_stable)}/{n} = {st_pct:.1f}%   (справочно)")
    print(f"   ячейка (domain+stance) одинакова: {len(cell_stable)}/{n} = {cell_pct:.1f}%")

    print("\n2) SAFETY (катастрофические пары -> разные domain, majority):")
    safety_ok = True
    for a, b in SAFETY_PAIRS:
        da, db = maj_dom(a), maj_dom(b)
        ok = da is not None and db is not None and da != db
        safety_ok = safety_ok and ok
        print(f"   {a!r}[{da}]  vs  {b!r}[{db}]  -> {'РАЗДЕЛЕНЫ' if ok else 'СЛИЛИСЬ (провал)'}")
    print(f"   [ПЛАНКА: все разделены -> {'ВЗЯТА' if safety_ok else 'НЕ взята'}]")

    print("\n3) RECALL (парафразы -> одна ячейка domain+stance, majority):")
    recall_ok = True
    for a, b in RECALL_PAIRS:
        ca, cb = maj_cell(a), maj_cell(b)
        ok = ca is not None and cb is not None and ca == cb
        recall_ok = recall_ok and ok
        print(f"   {a!r}{ca}  vs  {b!r}{cb}  -> {'СО-ЛОКОВАНЫ' if ok else 'РАЗОШЛИСЬ (провал)'}")
    print(f"   [ПЛАНКА: со-локованы -> {'ВЗЯТА' if recall_ok else 'НЕ взята'}]")

    print(f"\n4) BORDERLINE-УТЕЧКА (плавают по domain, {len(leaky)} лейблов):")
    if leaky:
        for l, dist in sorted(leaky):
            print(f"   {l!r}: {dist}")
    else:
        print("   нет — domain стабилен у всех.")

    # hardcases — a separate list to eyeball
    print("\nХАРДКЕЙСЫ (majority domain/stance):")
    for h in HARDCASES:
        v = full.get(h, [])
        if v:
            print(f"   {h!r:<28} -> {maj_dom(h)} / {majority([r['stance'] for r in v])}")
        else:
            print(f"   {h!r:<28} -> (нет данных)")

    # --- 5. PARAPHRASE-LEAK (arbiter VETO vs PENALTY) ---
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        para_pairs = json.load(f)["paraphrase"]
    leak_ok = 0
    leak_total = 0
    over_partition = []
    for a, b in para_pairs:
        va, vb = full.get(a, []), full.get(b, [])
        if len(va) < N_RUNS or len(vb) < N_RUNS:
            continue
        leak_total += 1
        # strict: in EACH of the 5 runs domain(a)==domain(b)
        per_run_equal = all(va[i]["domain"] == vb[i]["domain"] for i in range(N_RUNS))
        if per_run_equal:
            leak_ok += 1
        else:
            over_partition.append((a, b,
                                   dict(Counter(r["domain"] for r in va[:N_RUNS])),
                                   dict(Counter(r["domain"] for r in vb[:N_RUNS]))))
    leak_pct = 100.0 * leak_ok / leak_total if leak_total else 0.0
    print(f"\n5) PARAPHRASE-LEAK (оба лейбла пары -> один domain во ВСЕХ 5 ранах):")
    print(f"   совпали: {leak_ok}/{leak_total} = {leak_pct:.1f}%")
    if leak_pct >= 80:
        print(f"   -> ВЫСОКИЙ: over-partition редок, domain безопасен как VETO.")
    else:
        print(f"   -> НИЗКИЙ: парафразы часто разъезжаются, domain только как PENALTY.")
    print(f"\n   OVER-PARTITION (разъехавшиеся пары paraphrase):")
    if over_partition:
        for a, b, da, db in over_partition:
            print(f"     {a!r}{da}  ↮  {b!r}{db}")
    else:
        print("     нет — все парафразы со-домены.")

    # --- 6. CROSS-PROVIDER (24 Cerebras vs Groq) ---
    if os.path.exists(CEREBRAS24_PATH):
        with open(CEREBRAS24_PATH, "r", encoding="utf-8") as f:
            cer = json.load(f)
        match = 0
        diffs = []
        for l, cv in cer.items():
            cvv = _valid(cv)
            gvv = full.get(l, [])
            if not cvv or not gvv:
                continue
            cd = majority([r["domain"] for r in cvv])
            gd = majority([r["domain"] for r in gvv])
            if cd == gd:
                match += 1
            else:
                diffs.append((l, cd, gd))
        tot = len(cer)
        print(f"\n6) CROSS-PROVIDER (24 Cerebras vs Groq, majority domain):")
        print(f"   совпало: {match}/{tot} = {100.0*match/tot:.1f}%")
        if diffs:
            for l, cd, gd in diffs:
                print(f"     {l!r}: Cerebras[{cd}] -> Groq[{gd}]")
        else:
            print("     все домены совпали (провайдер-инвариантно).")

    print("\n" + "=" * 80)
    print("ИТОГ ПО ПЛАНКАМ")
    print("=" * 80)
    print(f"  1 стабильность domain ≥90%: {'ДА' if dom_pct >= 90 else 'НЕТ'} ({dom_pct:.1f}%)")
    print(f"  2 safety разделены:         {'ДА' if safety_ok else 'НЕТ'}")
    print(f"  3 recall со-локованы:       {'ДА' if recall_ok else 'НЕТ'}")
    all_ok = dom_pct >= 90 and safety_ok and recall_ok
    print(f"\n  ВЕРДИКТ: domain-routing {'ПРОХОДИТ все планки' if all_ok else 'НЕ проходит все планки'}.")
    print("\nПрод не трогался. run_drift не запускался.")


def main():
    args = sys.argv[1:]
    if "--analyze-only" in args:
        runs = load_runs()
        if not runs:
            print("Нет routing_runs.json. Сначала без --analyze-only.")
            return 1
        analyze(runs)
        return 0
    runs = asyncio.run(ensure_routes(force="--reroute" in args))
    if "--route-only" in args:
        print("Роуты готовы (--route-only).")
        return 0
    analyze(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
