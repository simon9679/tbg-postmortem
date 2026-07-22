# TBG on ES-MemEval — Benchmark Results

**System.** TBG (Temporal Belief Graph): a compact, engine-computed model of a user's
evolving beliefs — confidence per belief, confidence trajectories, opposition edges,
turning points, and ambivalence (a single belief holding both positive and negative
evidence). This document reports how it scores against strong baselines on a public
benchmark. No product claims beyond what the numbers support.

## Benchmark
- **ES-MemEval / EvoEmo** — peer-reviewed (ACM Web Conference 2026, arXiv 2602.01885;
  Zenodo DOI 10.5281/zenodo.18338564). 18 evolving-state conversations, 13–33 sessions each.
- **Official judge, used verbatim** from the paper's repo (`src/lib/qa/qa_experiment.py`):
  GPT-4o-style, `0 = wrong / 1 = partial / 2 = correct`, `Score: X` regex-extracted.
- Subset scored: **n = 60 QA** drawn from the 3 evolving-state abilities (temporal
  reasoning, conflict detection, user modeling); the full evolving-state pool is 857.

## Methodology
- **5 arms, one shared answerer** (cerebras `gpt-oss-120b`, temp = 0):
  A truncation · B summary · C RAG (lexical BM25, 0 LLM) · **D prompt state-tracker (the bar)** · **E TBG**.
- **Pre-registered decision rule (G4), fixed in config before any score existed:**
  TBG is a *product* iff score(E) beats *both* D *and* C by more than judge noise.
  Beating C but tying D → parity, **not** a product.
- **Budgets equalized pre-run, not tuned.** An honesty audit before any QA score found
  the input budget skewed *toward* TBG and the representation budget skewed *against* it;
  both were equalized (identical seeker text to every arm; a shared `REPR_BUDGET` for all
  answers). Limits are marked "equalized pre-run, not tuned" in code.
- **Resumable free-tier runs.** Ingest cached offline once per conversation; the harness
  aggregates only *complete* items (all 5 arms + judges scored), so every partial cut is
  apples-to-apples. Real wall was a 5-req/min rate bucket, paced around, not a daily quota.

## Scores (5 arms × 3 abilities, n = 60, official judge)
| arm | temporal | conflict | user_model | **AGG** |
|---|---|---|---|---|
| A trunc | 0.45 | 0.85 | 0.75 | 0.68 |
| B summ | 0.25 | 0.65 | 0.90 | 0.60 |
| **C rag (bm25)** | 0.90 | 1.30 | 0.90 | **1.03** |
| D tracker | 0.45 | 0.45 | 0.75 | 0.55 |
| **E tbg** | 0.35 | 0.85 | 0.85 | **0.68** |

## Verdict — no aggregate edge (robust across two ingests and four full QA passes)
- **E − D = +0.13**, within judge noise → TBG ≈ the prompt state-tracker on aggregate.
- **E − C = −0.35** → BM25-RAG takes the aggregate (mostly via temporal recall).
- Pre-registered G4 ("beat both D and C") → **not a product**. RAG blocks. Reported as-is.
- **The aggregate verdict is reproducible:** a full re-ingest (v1.1) and three further QA
  passes (re-answer, v1.1, rank-OFF) leave it unchanged — NO-EDGE holds every time. (Subset
  scores are *not* this stable — see the ingest-variance caveat below and
  `EVAL_RELIABILITY_NOTE.md`.)

## Validated pockets (single-ingest; carry an ingest-variance caveat)
> **Caveat — single-ingest; ingest variance substantial (observed ±0.40 on capability
> subsets).** These pocket numbers come from one ingest. A controlled re-ingest swung the
> conflict subset by −0.40 and user_model by +0.15 (both directions), far beyond judge
> (0.10) or answerer (0.05) noise; a format ablation showed the swing is the re-extracted
> graph, not the rendering. Read each pocket number as one draw from a high-variance
> distribution, not a fixed value. Full decomposition: `EVAL_RELIABILITY_NOTE.md`.

- **Conflict detection: E − D = +0.40** (single ingest). Item-level TBG **5-0-15** vs the
  tracker — but a re-ingest put this at +0.00, so treat +0.40 as within the observed ±0.40 swing.
- **Long-horizon: E − D = +0.60** (single ingest), gap growing with length
  (short +0.23 → mid −0.03 → long +0.60) — same ingest-variance caveat applies.
- Qualitatively, E wins "does the user feel / believe X" and "how has X evolved" (confidence
  nodes + trajectories); it loses pointed-fact recall (the compressed belief-state abstracts
  the verbatim detail away).

## v1.1 (engineering-correctness release)
A follow-up release fixed three engine/render correctness issues (decay-aware turning
points, prune tie-break, rank-based conflict rendering). Measured on a full re-ingest:
**subset deltas fell within the measured ingest variance** (conflict −0.40, user_model +0.15,
aggregate 0.68→0.60) and the aggregate verdict stayed NO-EDGE. A **format ablation exonerated
rank rendering** (conflict 0.45 with and without it — identical), locating the subset movement
in ingest stochasticity, not the fixes' presentation. v1.1 is **not** presented as an
improvement; v1.0 remains the citable baseline.

## Human judge validation (K = 20, blind manual relabel)
- **judge_noise = 0.10** (mean |auto − manual| / 2, on the 0..1 gap scale; raw
  disagreement rate 0.20 over 20 items).
- **Systematic leniency toward the retrieval / state-digest arms: RAG bias = +0.33** (judge
  scores RAG above the human); tracker +0.25, TBG +0.17, truncation/summary 0.00. Note this
  is **not** a raw-length effect — TBG produces the longest answers (220 chars vs RAG's 139)
  yet is inflated least; the bias tracks retrieval/declarative *style*, not verbosity. See
  `EVAL_RELIABILITY_NOTE.md`.
- **After symmetric bias correction** (each arm minus its own leniency): tbg 0.68→0.51,
  tracker 0.55→0.30, rag 1.03→0.70. Corrected **E − D = +0.21** (survives noise);
  corrected E − C = −0.19 (0.51 − 0.70; RAG still ahead, but the gap roughly halves).
  [The read-only aggregation script reports −0.18, computed on unrounded per-arm biases.]
  See `EVAL_RELIABILITY_NOTE.md`.

## Extraction-sign diagnostic
Across 8 conversations / 194 turns: **253 contradict decisions, 100% lowered confidence,
0 sign errors** (never "contradict → confidence up"). The sign mechanism is sound, so the
conflict-detection edge rests on a healthy foundation. Per-turn dumps: `tbg_signs_p*.txt`.

## Caveats (read before citing "significant")
- RAG here is **lexical BM25**, not embedding retrieval — read every RAG comparison as
  "vs BM25-RAG", not "vs RAG in general". The paper's "RAG fails on evolving states" did
  **not** replicate with BM25 + a strong answerer.
- **n = 60 of 857** evolving-state QA; subgroup n (conflict / long) is 10–20.
- Bias correction is **coarse at n ≈ 6 per arm** — directional, not a precise estimate.
- **Single answerer model** (`gpt-oss-120b`); judge is a single model. Multi-model
  replication is future work.
- The conflict edge is achieved by **node-level dynamics, not rendered conflict edges** —
  the rendered states shown to the answering model contained no conflict lines (see the
  report's attribution note); a rank-based conflict render was added *after* the gate and is an **unmeasured**
  potential upside for v1.1.
