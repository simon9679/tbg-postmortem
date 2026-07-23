# Evaluation Reliability: Three Measured Layers of Noise on a Memory Benchmark

*Central methodological finding. We measured our own reproducibility on ES-MemEval and
quantified three independent noise sources between a QA item and its score. The headline:
**subset-level capability claims at n ≈ 20 are dominated by ingest stochasticity, while the
aggregate verdict is robust.** This holds regardless of whether TBG ships — it is a
statement about how memory benchmarks must be read.*

A score on this benchmark passes through three stochastic stages:
**ingest** (dialogue → belief graph) → **answer** (memory → answer) → **judge** (answer →
0/1/2). We isolated and measured each.

---

## Layer 1 — Judge noise (smallest)
**Method:** blind manual relabel, K = 20. A stratified sample of (question, gold, arm-answer)
triples hand-scored 0/1/2 **without seeing the judge's score**, weighted toward the decisive
arms and toward items the judge scored 2. Aggregation `k20_aggregate.py` (0 LLM).

**Numbers:**
- Overall judge noise = **0.10** (mean |auto − manual| / 2; raw disagreement 4/20 = 0.20).
- Per-arm leniency bias (mean auto − manual): **rag +0.33**, tracker +0.25, tbg +0.17,
  trunc/summary +0.00.
- **Not a length effect:** TBG has the *longest* answers (220 chars vs RAG's 139) yet is
  inflated *least* — the bias tracks retrieval / state-digest **style** (source-quoting,
  confidently declarative), not verbosity.

**Consequence:** an LLM judge applies a *retrieval-style premium*. Symmetric bias correction
moves rag 1.03→0.70, tbg 0.68→0.51, tracker 0.55→0.30 — a chunk of RAG's apparent lead is a
judge artifact, and the raw leaderboard reorders once leniency is subtracted.

---

## Layer 2 — Answerer noise (small, symmetric)
**Method:** re-answer ablation. Feed the **byte-identical** frozen v1.0 render strings to a
fresh answerer + judge pass (temp = 0, but not byte-exact on this provider). Input held
constant; only the answerer+judge stage re-rolled. Dumps: `_reans/`.

**Numbers (v1.0 input, original → re-answer):**

| capability | original | re-answer | Δ |
|---|---|---|---|
| temporal | 0.35 | 0.40 | +0.05 |
| conflict | 0.85 | 0.90 | +0.05 |
| user_model | 0.85 | 0.85 | 0.00 |
| **aggregate** | 0.68 | 0.72 | +0.03 |

**Consequence:** on identical input the answerer is stable to **≈ ±0.05**. It is *not* a
major noise source — a capability score does not swing meaningfully from re-answering alone.

---

## Layer 3 — Ingest noise (largest, dominates subsets)
**Method:** controlled full re-ingest (16 conversations re-extracted, v1.1 config) **plus** a
render-format ablation to prove the swing is the graph, not the presentation. The format
ablation re-renders the *same* v1.1 graphs with the old threshold format (`TBG_RANK_RENDER=0`)
and re-answers — isolating render format as a variable. Dumps: `_rankoff/`, `gate_graphs/`.

**Numbers (v1.0 → re-ingest v1.1, and format toggled on the v1.1 graphs):**

| capability | v1.0 | re-ingest (v1.1) | Δ | same graph, old format |
|---|---|---|---|---|
| temporal | 0.35 | 0.35 | 0.00 (flat) | 0.20 |
| **conflict** | 0.85 | **0.45** | **−0.40** | **0.45** |
| user_model | 0.85 | 1.00 | +0.15 | 1.00 |
| aggregate | 0.68 | 0.60 | −0.08 | 0.55 |

**Consequence — the decisive result:**
- A single re-ingest swung capability subsets by **−0.40 (conflict) and +0.15 (user_model)**
  — **in both directions**, far beyond judge (0.10) or answerer (0.05) noise.
- **Format contribution to the conflict collapse = 0.00:** toggling the render format on the
  *same* re-ingested graph left conflict at 0.45 → 0.45. (Format is not globally inert — on the
  same graph it moved *temporal* 0.35 → 0.20; the claim is specifically about the conflict swing.)
  The conflict swing is the **graph produced by re-extraction**, not the answerer, not the judge,
  not the presentation. (This also exonerates rank-based rendering for conflict.)
- **4 of the 5** collapsed conflict items recover on the v1.0 graph (Layer 2); the fifth flips
  under answerer noise on identical input. All 5 stay collapsed under a format change on the
  v1.1 graph (Layer 3) — a clean triangulation.
- **Confound flagged honestly:** the re-ingest was bundled with three engineering fixes (E5
  changes prune → graph composition), so "fixes vs pure lottery" is a **pre-registered open
  question** (separation experiment pending — one more controlled ingest). But the bidirectional
  swings (−0.40 / +0.15), the zero format contribution, and unchanged belief/turning-point
  counts in the collapsed conversations point to **extraction stochasticity as the dominant
  component**.

---

## Conclusion
| noise source | magnitude (0..2 subset scale) | method |
|---|---|---|
| judge | ~0.10 | K=20 blind relabel |
| answerer | ~0.05 | re-answer on byte-identical input |
| **ingest** | **~0.40 swing (2 ingests; not yet a band)** | controlled re-ingest + format ablation |

**Subset-level capability claims on this memory benchmark at n ≈ 20 are dominated by ingest
stochasticity, not by judge or answerer noise.** A single-ingest "+0.40 on conflict" is one
draw from a high-variance distribution and must be reported with that caveat.

**Aggregate verdicts are robust.** Across two independent ingests and four full QA passes
(v1.0, re-answer, v1.1, rank-OFF), the n = 60 verdict is unchanged: **NO aggregate edge** —
TBG ties the prompt state-tracker (within noise) and trails BM25-RAG; the pre-registered
product rule fails both times.

**Practical rule for memory benchmarks:** report subset claims only with an ingest-variance
estimate. We observed a subset **swing of up to 0.40 between two ingests** — this is a single
pair of draws, **not** a measured variance band; establishing one requires **≥ 5 ingests**
(see scale-up). Ranking memory systems on single-ingest capability subsets is unsafe.

---

## Novelty & scale-up
- **Novelty:** LLM-judge length/style bias is documented in prior work; our contributions are
  (1) **per-arm judge-bias quantification** on an official memory benchmark, with the resulting
  **leaderboard reordering** and a **refutation of the naive length mechanism** (the bias tracks
  retrieval *style*, not answer length), and (2) an explicit **three-layer noise decomposition**
  on that benchmark showing ingest — not the judge — is the dominant reproducibility risk at
  subset granularity.
- **Scale-up:** K = 200 blind relabels (tighten per-arm bias from n≈6 to n≈40); ≥ 3 judge
  families; **≥ 5 ingests per system** to place an empirical variance band on every subset
  claim; report ingest-variance as a first-class benchmark metric, not a footnote.
