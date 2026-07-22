#!/usr/bin/env python3
"""
K=20 judge validation — aggregation. Zero LLM calls. Read-only.
Run AFTER k20_manual_review.md is filled with МОЯ ОЦЕНКА values.

Computes:
  1. judge_noise  = disagreement rate (fraction manual != auto) AND mean |auto-manual|/2.
  2. per-arm bias = mean(auto - manual) per arm  (positive => judge more lenient than human).
                    Focus: is the judge lenient to C(rag) vs D(tracker)/E(tbg)?
  3. survival     = which final aggregate gaps still exceed the MEASURED judge_noise,
                    incl. a rag-bias-adjusted E-C.
"""
import json, re
from collections import defaultdict

# Final aggregate numbers from the full n=60 run (REPORT_esmemeval.md), for survival check.
FINAL = {"tbg": 0.68, "tracker": 0.55, "rag": 1.03,
         "gaps": {"overall E-D": +0.13, "conflict E-D": +0.40, "long E-D": +0.60, "E-C(bm25)": -0.35}}
DISP2ARM = {"a trunc": "trunc", "b summ": "summary", "c rag(bm25)": "rag",
            "d tracker": "tracker", "e tbg": "tbg"}


def parse_manual(path="k20_manual_review_ru.md"):
    txt = open(path, encoding="utf-8").read()
    blocks = re.split(r"^\[(\d+)\] ", txt, flags=re.M)[1:]  # [n, body, n, body, ...]
    out = []
    for i in range(0, len(blocks), 2):
        n = int(blocks[i]); body = blocks[i + 1]
        arm = re.search(r"arm=([^\n|]+)", body).group(1).strip().lower()
        # tolerant: accept "0", " 0", "_0__", "_ 0" etc — first 0/1/2 after the label,
        # skipping spaces/underscores. Blank "___" (no digit) stays unlabeled (None).
        m = re.search(r"МОЯ ОЦЕНКА:[ _]*([0-2])", body)
        manual = int(m.group(1)) if m else None
        out.append({"n": n, "arm": DISP2ARM.get(arm, arm), "manual": manual})
    return out


def main():
    manual = parse_manual()
    sel = json.load(open("_k20_selected.json", encoding="utf-8"))
    # align by order (both are the 20 selected, same order as written)
    if len(manual) != len(sel):
        print(f"WARN: {len(manual)} blocks vs {len(sel)} selected");
    rows = []
    for mu, su in zip(manual, sel):
        rows.append({"arm": su["arm"], "auto": su["auto"], "manual": mu["manual"]})
    unlabeled = [r for r in rows if r["manual"] is None]
    if unlabeled:
        print(f"NOT READY: {len(unlabeled)}/{len(rows)} blocks still blank "
              f"(fill all >>> МОЯ ОЦЕНКА: with 0/1/2, then re-run).")
        return

    # 1. judge_noise
    dis = sum(1 for r in rows if r["manual"] != r["auto"])
    jn_disagree = dis / len(rows)
    jn_norm = sum(abs(r["auto"] - r["manual"]) for r in rows) / len(rows) / 2  # 0..1 scale
    print("=" * 60)
    print("K=20 JUDGE VALIDATION")
    print("=" * 60)
    print(f"n={len(rows)} | disagreements(auto!=manual)={dis} -> judge_noise(disagree rate)={jn_disagree:.2f}")
    print(f"mean |auto-manual|/2 = {jn_norm:.3f}  (same 0..1 scale as score gaps)")

    # 2. per-arm bias
    print("\nper-arm bias  mean(auto - manual)  [+ = judge more lenient than you]:")
    byarm = defaultdict(list)
    for r in rows:
        byarm[r["arm"]].append(r["auto"] - r["manual"])
    for a in ["trunc", "summary", "rag", "tracker", "tbg"]:
        if byarm[a]:
            b = sum(byarm[a]) / len(byarm[a])
            print(f"  {a:<9} n={len(byarm[a])}  bias={b:+.2f}")
    rag_bias = (sum(byarm['rag']) / len(byarm['rag'])) if byarm['rag'] else 0.0
    print(f"  -> RAG leniency suspicion: judge is {'LENIENT to RAG' if rag_bias>0.15 else 'not notably lenient to RAG'} "
          f"(rag bias {rag_bias:+.2f})")

    # 3. survival vs measured judge_noise (use the normalized noise on the 0..1 gap scale)
    jn = jn_norm
    print(f"\nsurvival of final gaps vs MEASURED judge_noise={jn:.3f}:")
    for name, g in FINAL["gaps"].items():
        print(f"  {name:<14} {g:+.2f}  -> {'SURVIVES' if abs(g) > jn else 'within noise (was over-claimed)'}")
    # SYMMETRIC bias correction: subtract EACH arm's own leniency bias (estimate of
    # the human-judged score), then recompute the decisive gaps. Adjusting only RAG
    # would overstate the closing — the judge inflates tracker/tbg too.
    bias = {a: (sum(byarm[a]) / len(byarm[a]) if byarm[a] else 0.0) for a in ['rag', 'tracker', 'tbg']}
    adj = {a: FINAL[a] - bias[a] for a in ['rag', 'tracker', 'tbg']}
    ed_adj = adj['tbg'] - adj['tracker']
    ec_adj = adj['tbg'] - adj['rag']
    print("\nSYMMETRIC bias-corrected (each arm minus its own leniency):")
    print(f"  tbg {FINAL['tbg']:.2f}->{adj['tbg']:.2f} | tracker {FINAL['tracker']:.2f}->{adj['tracker']:.2f} "
          f"| rag {FINAL['rag']:.2f}->{adj['rag']:.2f}")
    print(f"  E-D(vs tracker) {FINAL['gaps']['overall E-D']:+.2f} -> {ed_adj:+.2f}  "
          f"({'SURVIVES' if abs(ed_adj) > jn else 'within noise'})")
    print(f"  E-C(vs bm25)    {FINAL['gaps']['E-C(bm25)']:+.2f} -> {ec_adj:+.2f}  "
          f"({'RAG still ahead' if ec_adj < -jn else 'within noise (tie)'})")


if __name__ == "__main__":
    main()
