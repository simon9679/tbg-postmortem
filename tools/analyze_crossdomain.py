#!/usr/bin/env python3
"""
VETO vs PENALTY — arbitration over existing data. WITHOUT LLM, WITHOUT prod edits.

Question: if a pair's domain doesn't match, how do we handle it —
  VETO    (never merge cross-domain) — simple, but we lose cross-domain synonyms;
  PENALTY (penalize, but at very high cosine still merge) — saves synonyms,
          IF a threshold T exists: cos(synonym) >= T > cos(catastrophe/unrelated).

Data:
  - domain per label = mode of 5 runs from routing_runs.json (Groq, ready).
  - cos = MiniLM (fact_engine), normalize, dot.

Types of cross-domain pairs:
  synonym (SAVE)              : 2 over-partition pairs from L1
       fear of failure↔scared it will fail; people pleaser↔struggles to say no
  catastrophe (DON'T save)    : career security↔financial security; mortgage↔doctor partner
  unrelated (DON'T save)      : the whole unrelated class from calib_pairs_v2

Verdict:
  separable (min(synonym) > max(catastrophe∪unrelated)) -> PENALTY viable, T from data;
  overlap                                               -> PENALTY doesn't save -> VETO, tolerate duplicates.

SELF-SAFETY: is there any catastrophe pair with domain=self? If not —
  the rule "don't penalize pairs where one domain is self" is safe for them.

Caveat: n of synonyms is tiny (2) -> direction, NOT final calibration.

RUN: py -3 tools/analyze_crossdomain.py
"""
import os
import sys
import json
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PAIRS_PATH = os.path.join(HERE, "calib_pairs_v2.json")
RUNS_PATH = os.path.join(HERE, "routing_runs.json")
DOMAINS = ["career", "money", "health", "relationships", "self", "family", "lifestyle", "meaning", "other"]
STANCES = ["approach", "avoid", "neutral"]

SYNONYM_PAIRS = [("fear of failure", "scared it will fail"),
                 ("people pleaser", "struggles to say no")]
CATASTROPHE_PAIRS = [("career security", "financial security"),
                     ("mortgage payments", "doctor partner")]


def domain_of(runs, label):
    v = [r for r in runs.get(label, [])
         if isinstance(r, dict) and r.get("domain") in DOMAINS]
    if not v:
        return None
    return Counter(r["domain"] for r in v).most_common(1)[0][0]


def main():
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(RUNS_PATH, "r", encoding="utf-8") as f:
        runs = json.load(f)

    from fact_engine import get_embed_model
    model = get_embed_model()
    cache = {}

    def emb(l):
        if l not in cache:
            cache[l] = model.encode(l, normalize_embeddings=True)
        return cache[l]

    def cos(a, b):
        return float(np.dot(emb(a), emb(b)))

    # collect pairs: unrelated + synonym + catastrophe (+ domains)
    records = []  # (cos, a, b, da, db, cross, type)

    def add(a, b, typ):
        da, db = domain_of(runs, a), domain_of(runs, b)
        cross = (da is not None and db is not None and da != db)
        records.append({"cos": cos(a, b), "a": a, "b": b, "da": da, "db": db,
                        "cross": cross, "type": typ})

    for a, b in SYNONYM_PAIRS:
        add(a, b, "synonym")
    for a, b in CATASTROPHE_PAIRS:
        add(a, b, "catastrophe")
    for a, b in data["unrelated"]:
        add(a, b, "unrelated")

    # ---- main question: threshold T among CROSS-domain ----
    cross = [r for r in records if r["cross"]]
    syn = [r for r in cross if r["type"] == "synonym"]
    bad = [r for r in cross if r["type"] in ("catastrophe", "unrelated")]

    print("=" * 84)
    print("КРОСС-ДОМЕННЫЕ ПАРЫ, отсортировано по косинусу (MiniLM)")
    print("=" * 84)
    print(f"{'cos':>6}  {'type':<12} {'domains':<22} пара")
    print("-" * 84)
    for r in sorted(cross, key=lambda x: -x["cos"]):
        dom = f"{r['da']}↔{r['db']}"
        print(f"{r['cos']:>6.3f}  {r['type']:<12} {dom:<22} {r['a']!r} ↔ {r['b']!r}")

    # pairs that are NOT cross-domain (for transparency — their domain matched)
    same = [r for r in records if not r["cross"]]
    if same:
        print("\n(не кросс-доменные — domain совпал, не участвуют в арбитраже:)")
        for r in same:
            print(f"  {r['cos']:>6.3f}  {r['type']:<12} {r['da']}={r['db']:<14} {r['a']!r}↔{r['b']!r}")

    print("\n" + "=" * 84)
    print("АРБИТРАЖ: существует ли порог T (cos(synonym) >= T > cos(catastrophe∪unrelated))?")
    print("=" * 84)
    if not syn or not bad:
        print("Недостаточно кросс-доменных пар в одной из групп.")
        return
    min_syn = min(r["cos"] for r in syn)
    max_bad = max(r["cos"] for r in bad)
    argmin_syn = min(syn, key=lambda x: x["cos"])
    argmax_bad = max(bad, key=lambda x: x["cos"])
    gap = min_syn - max_bad
    print(f"  min(synonym)            = {min_syn:.3f}   ({argmin_syn['a']!r}↔{argmin_syn['b']!r})")
    print(f"  max(catastrophe∪unrel.) = {max_bad:.3f}   ({argmax_bad['a']!r}↔{argmax_bad['b']!r})")
    print(f"  gap = {gap:+.3f}")
    separable = gap > 0
    if separable:
        T = (min_syn + max_bad) / 2.0
        print(f"  -> SEPARABLE. Порог T≈{T:.3f} (из данных, не угадан). "
              f"PENALTY жизнеспособен: cos≥T мёржит вопреки разнице доменов.")
    else:
        print(f"  -> OVERLAP. Порога нет: '{argmax_bad['a']}↔{argmax_bad['b']}' (катастрофа/unrel) "
              f"имеет косинус ВЫШЕ слабейшего синонима. PENALTY по косинусу не спасает -> VETO, "
              f"дубли терпим, приоритет L1.")

    # ---- SELF-SAFETY ----
    print("\n" + "=" * 84)
    print("SELF-SAFETY: участвует ли domain=self в catastrophe-парах?")
    print("=" * 84)
    cat_self = [r for r in records if r["type"] == "catastrophe"
                and (r["da"] == "self" or r["db"] == "self")]
    print(f"  catastrophe-пар с domain=self: {len(cat_self)}")
    if not cat_self:
        print("  -> НЕТ. Правило «не штрафовать пары с domain=self» для катастроф БЕЗОПАСНО.")
    else:
        for r in cat_self:
            print(f"     ОПАСНО: {r['a']!r}[{r['da']}]↔{r['b']!r}[{r['db']}]")

    print("\n  Все self-involving пары (из рассмотренных) и тип:")
    self_pairs = [r for r in records if r["da"] == "self" or r["db"] == "self"]
    if self_pairs:
        for r in sorted(self_pairs, key=lambda x: -x["cos"]):
            print(f"     {r['cos']:>6.3f}  {r['type']:<12} {r['da']}↔{r['db']:<14} {r['a']!r}↔{r['b']!r}")
    else:
        print("     нет.")

    print("\n" + "=" * 84)
    print("ВЕРДИКТ")
    print("=" * 84)
    print(f"  Порог штрафа существует (PENALTY по косинусу): {'ДА' if separable else 'НЕТ'}")
    print(f"  self-meta-domain безопасен для катастроф:       {'ДА' if not cat_self else 'НЕТ'}")
    print("  Оговорка: n синонимов = 2 -> направление, не финальная калибровка.")
    print("\nПрод не трогался. run_drift не запускался.")


if __name__ == "__main__":
    main()
