#!/usr/bin/env python3
"""
Final L0-resolver batch. WITHOUT prod edits. Offline on calib_pairs_v2.json.

4 similarity methods per pair:
  1. cos_pooled  — MiniLM sentence-embedding, cosine (baseline).
  2. cos_white   — MiniLM + PCA-whitening (multidimensional, NOT the scalar re-centering
                   from E3a-2). μ,W are fit on the set's unique labels + ~40 generic
                   belief phrases. emb'=(emb−μ)·W (top-k components), cosine on whitened.
                   NOTE: the fit is transductive + the corpus is small (n<dim) → top-k reduction
                   is mandatory, overfitting risk. This is a probe, not a prod transform.
  3. maxsim      — MiniLM token-embeddings, mean-of-max, symmetrized (ColBERT-style).
  4. xenc_sts    — STS/paraphrase cross-encoder (cross-encoder/stsb-roberta-base, NOT NLI),
                   similarity-score. Measures ms/pair.

Acceptance bar (FIXED BEFORE THE RUN):
  a method wins  <=>  sep_MERGE > 0  (min(paraphrase) > max(adjacent))
                          AND synonyms preserved: M&A attorney↔corporate lawyer and
                          burnout↔exhaustion stay above max(adjacent) (merge band).
  sep_MERGE = min(paraphrase) − max(adjacent)
  sep_FLAG  = min(paraphrase∪opposite∪adjacent) − max(unrelated)   [topic boundary]
If none clears the bar — L0-resolver is closed.

RUN: py -3 tools/probe_L0.py
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PAIRS_PATH = os.path.join(HERE, "calib_pairs_v2.json")
CLASSES = ("paraphrase", "affective_opposite", "structural_opposite", "adjacent", "unrelated")
TOPIC_GROUP = ("paraphrase", "affective_opposite", "structural_opposite", "adjacent")  # everything is "topic" vs unrelated

# Synonym check (spot-check; burnout↔exhaustion — cross-pair synonym, not among the set's pairs).
SYN_PAIRS = [("M&A attorney", "corporate lawyer"), ("burnout", "exhaustion")]

# ~40 generic belief phrases to stabilize whitening covariance (outside the test pairs).
GENERIC = [
    "values honesty", "feels anxious", "wants success", "fears change", "enjoys learning",
    "needs approval", "avoids conflict", "seeks adventure", "values loyalty", "feels lonely",
    "wants recognition", "fears rejection", "loves challenge", "needs structure", "values privacy",
    "feels overwhelmed", "wants balance", "fears uncertainty", "enjoys solitude", "values fairness",
    "seeks meaning", "feels confident", "wants connection", "fears judgement", "values tradition",
    "needs control", "loves spontaneity", "feels grateful", "wants independence", "fears commitment",
    "values curiosity", "feels restless", "wants validation", "seeks comfort", "values courage",
    "feels hopeful", "wants respect", "fears stagnation", "enjoys teaching", "values discipline",
]


def load_pairs():
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {c: d[c] for c in CLASSES}


def unique_labels(pairs):
    seen = []
    for c in CLASSES:
        for a, b in pairs[c]:
            for x in (a, b):
                if x not in seen:
                    seen.append(x)
    return seen


def fit_whitening(X, k):
    """BERT-whitening (Su et al. 2021), top-k components. X: (n,d) raw embeddings."""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / Xc.shape[0]
    U, S, _ = np.linalg.svd(cov)
    k = min(k, np.sum(S > 1e-8))
    W = U[:, :k] @ np.diag(1.0 / np.sqrt(S[:k] + 1e-8))
    return mu, W


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def maxsim(toks_a, toks_b):
    """mean-of-max over normalized tokens, symmetrized."""
    A = toks_a / (np.linalg.norm(toks_a, axis=1, keepdims=True) + 1e-9)
    B = toks_b / (np.linalg.norm(toks_b, axis=1, keepdims=True) + 1e-9)
    sim = A @ B.T  # (na, nb)
    a2b = sim.max(axis=1).mean()
    b2a = sim.max(axis=0).mean()
    return float(0.5 * (a2b + b2a))


def stats_min(vals):
    return float(np.min(vals))


def stats_max(vals):
    return float(np.max(vals))


def main():
    pairs = load_pairs()
    labels = unique_labels(pairs)
    print(f"Пар: {sum(len(pairs[c]) for c in CLASSES)}; уникальных лейблов: {len(labels)}")

    from fact_engine import get_embed_model
    from sentence_transformers import CrossEncoder
    model = get_embed_model()

    # --- precomputations per label ---
    print("Кодирую sentence-embeddings (pooled, raw) ...")
    pooled = {l: model.encode(l, normalize_embeddings=True) for l in labels}
    raw = {l: model.encode(l, normalize_embeddings=False) for l in labels}

    print("Фит whitening (лейблы + generic) ...")
    corpus_labels = labels + [g for g in GENERIC if g not in labels]
    Xc = np.stack([model.encode(l, normalize_embeddings=False) for l in corpus_labels])
    k = min(80, Xc.shape[0] - 1)
    mu, W = fit_whitening(Xc, k)
    white = {l: ((raw[l][None, :] - mu) @ W)[0] for l in labels}
    print(f"  whitening: corpus={Xc.shape[0]}, dim={Xc.shape[1]}, top-k={W.shape[1]} "
          f"(n<dim -> редукция обязательна; фит трансдуктивный, проба)")

    print("Кодирую token-embeddings (для MaxSim) ...")
    toks = {}
    for l in labels:
        t = model.encode(l, output_value="token_embeddings", convert_to_numpy=False)
        toks[l] = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)

    print("Загрузка STS cross-encoder (cross-encoder/stsb-roberta-base, ~500МБ при первом запуске) ...")
    xenc = CrossEncoder("cross-encoder/stsb-roberta-base")

    # --- compute the 4 methods by class ---
    methods = ["cos_pooled", "cos_white", "maxsim", "xenc_sts"]
    by_class = {m: {c: [] for c in CLASSES} for m in methods}
    xenc_times = []

    for c in CLASSES:
        for a, b in pairs[c]:
            by_class["cos_pooled"][c].append(cos(pooled[a], pooled[b]))
            by_class["cos_white"][c].append(cos(white[a], white[b]))
            by_class["maxsim"][c].append(maxsim(toks[a], toks[b]))
            t0 = time.perf_counter()
            sc = float(xenc.predict([[a, b]])[0])
            xenc_times.append((time.perf_counter() - t0) * 1000.0)
            by_class["xenc_sts"][c].append(sc)

    # --- synonym check ---
    syn = {m: [] for m in methods}
    for (a, b) in SYN_PAIRS:
        ea_p, eb_p = model.encode(a, normalize_embeddings=True), model.encode(b, normalize_embeddings=True)
        ea_r, eb_r = model.encode(a, normalize_embeddings=False), model.encode(b, normalize_embeddings=False)
        wa = ((ea_r[None, :] - mu) @ W)[0]
        wb = ((eb_r[None, :] - mu) @ W)[0]
        ta = model.encode(a, output_value="token_embeddings", convert_to_numpy=False)
        tb = model.encode(b, output_value="token_embeddings", convert_to_numpy=False)
        ta = ta.cpu().numpy() if hasattr(ta, "cpu") else np.asarray(ta)
        tb = tb.cpu().numpy() if hasattr(tb, "cpu") else np.asarray(tb)
        syn["cos_pooled"].append(cos(ea_p, eb_p))
        syn["cos_white"].append(cos(wa, wb))
        syn["maxsim"].append(maxsim(ta, tb))
        syn["xenc_sts"].append(float(xenc.predict([[a, b]])[0]))

    # --- seps + verdict ---
    def sep_merge(m):
        return stats_min(by_class[m]["paraphrase"]) - stats_max(by_class[m]["adjacent"])

    def sep_flag(m):
        topic = []
        for c in TOPIC_GROUP:
            topic += by_class[m][c]
        return stats_min(topic) - stats_max(by_class[m]["unrelated"])

    print("\n" + "=" * 92)
    print("ПО КЛАССАМ (min / mean / max)")
    print("=" * 92)
    print(f"{'метод':<12}" + "".join(f"{c[:10]:>16}" for c in CLASSES))
    for m in methods:
        row = f"{m:<12}"
        for c in CLASSES:
            v = by_class[m][c]
            row += f"  {np.min(v):>4.2f}/{np.mean(v):>4.2f}/{np.max(v):>4.2f}"
        print(row)

    print("\n" + "=" * 92)
    print("ИТОГ: метод × {sep_MERGE, sep_FLAG, синонимы, мс/пара, ВЕРДИКТ по планке}")
    print("=" * 92)
    print(f"{'метод':<12}{'sep_MERGE':>11}{'sep_FLAG':>10}{'maxAdj':>9}"
          f"{'syn:M&A':>9}{'syn:burn':>9}{'syn_ok':>8}{'мс/пара':>9}  вердикт")
    print("-" * 92)
    any_win = False
    for m in methods:
        sm = sep_merge(m)
        sf = sep_flag(m)
        max_adj = stats_max(by_class[m]["adjacent"])
        s1, s2 = syn[m][0], syn[m][1]
        syn_ok = (s1 > max_adj) and (s2 > max_adj)
        ms = float(np.mean(xenc_times)) if m == "xenc_sts" else None
        win = (sm > 0) and syn_ok
        any_win = any_win or win
        ms_str = f"{ms:>7.0f}  " if ms is not None else f"{'~0':>7}  "
        print(f"{m:<12}{sm:>+11.3f}{sf:>+10.3f}{max_adj:>9.2f}"
              f"{s1:>9.2f}{s2:>9.2f}{('ДА' if syn_ok else 'нет'):>8}{ms_str:>9}"
              f"  {'BERET ПЛАНКУ' if win else '-'}")

    print("\n" + "=" * 92)
    print("ВЕРДИКТ ПО ПЛАНКЕ (зафиксирована до прогона: sep_MERGE>0 И синонимы>max(adjacent))")
    print("=" * 92)
    if any_win:
        print("Как минимум один метод ВЗЯЛ планку — впервые чистый L0-resolver. Детали выше.")
    else:
        print("Ни один метод не взял планку. L0-resolver закрыт окончательно.")
        print("Baseline cos_pooled для контекста: sep_MERGE="
              f"{sep_merge('cos_pooled'):+.3f}, sep_FLAG={sep_flag('cos_pooled'):+.3f}.")
    print("\nПрод не трогался. run_drift не запускался.")


if __name__ == "__main__":
    main()
