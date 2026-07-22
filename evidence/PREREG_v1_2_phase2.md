# v1.2 Pre-Registration — Phase 2 (history cap on the long horizon, p8)

*Registered 2026-07-10, BEFORE any run. Both prediction sets recorded.*

## Question (one)
On the long, contradiction-rich p8 (33 sessions, 2024-09 → 2025-09) do the signals throttled
by `history_cap=5` revive at cap=25: `amf_ambiv`, oscillations, the trajectory layer? Last
test of "the intended cognitive layer was throttled by config, not absent."

## Step-0 map
- **Input:** `esmemeval/.../data/evo_emo.json`, `id="p8"`, 33 sessions. AD.ingest = 1 LLM call
  per session (canonicalization OFF) → **33 calls** for one ingest.
- **v1.1 ingest config reproduced:** `TBG_OPREF=1`, `TBG_EVIDENCE_TYPE=1`, `TBG_FIX_DECAY_TP=1`,
  temp=0, MAX_TOKENS=2500.
- **Replay substrate (stronger than the TZ's raw_facts):** capture the raw LLM **responses**
  per turn → replay through the real `extract_tbg_delta` with a stub llm_fn = **full fidelity**
  (not a 0.7-confidence approximation). Replay-cap5 MUST byte-match the capture (determinism
  control). `TBG_HISTORY_CAP` is import-time → each replay is a separate process.
- **Clock:** `TBG_DECAY_USE_LOGICAL_CLOCK=1` in capture + both replays (deterministic; p8's
  year-long session timestamps are not honestly reproducible in a minutes-long replay, and
  Phase-1 A showed wall-clock ≈0 at realistic timing anyway).

## Design: one LLM ingest + two 0-LLM replays
Isolates the cap effect from ingest stochasticity (±0.40) — two independent LLM ingests would
measure noise, not the cap. Capture responses once; replay them at cap=5 and cap=25.

## Predictions (before numbers)

### P1 — cap binds
- **Architect:** at cap=25, nodes with >5 history points appear on ≥30% of surviving nodes.
  ~85% (33 sessions almost guarantee repeat touches).
- **Code:** **agree directionally, mild doubt on the 30% line.** op/ref concentrates updates on
  core concepts; a long tail of one-off nodes stays at 1. I expect binding on the core but
  ~20–40% of *all* surviving nodes over 5. ~75%.

### P2 — AMF revives (the real unknown)
- **Architect:** `amf_ambiv` nonzero on more than 4/21, but **max stays < 0.2** — partial,
  not core-worthy. ~55%.
- **Code:** **agree it stays < 0.2, slightly higher confidence.** The `amf_ambiv = v/(1+v)`
  form saturates low — even {0,1} alternation caps ~0.23 (amf_filter's own note), so wider
  nonzero yes, but max < 0.2 likely. ~60%.

### P3 — oscillations
- **Architect:** oscillating-node count at cap=25 ≥ 2× the cap=5 count (Thompson sees more
  history). ~60%.
- **Code:** **agree more, doubt the exact 2×.** More history → more up/down pairs detected, but
  the multiple depends on how many nodes have long enough histories. I expect a clear increase,
  possibly < 2×. ~60%.

### P4 — ingest-variance bonus
- **Architect:** the fresh p8 ingest vs the frozen v1.1 p8 graph shows ≥20% node-set symdiff by
  label — another ingest-stochasticity data point. ~70%.
- **Code:** **agree, higher confidence.** We have already shown ingest variance is large
  (±0.40 on conflict subsets); ≥20% label symdiff on a 33-session graph is very likely. ~80%.

## What each outcome means
- P2 revives meaningfully (max ≥ 0.2, many nonzero) → first live AMF on a real long dialogue →
  a concrete case for returning the motivational profile to the experimental tier.
- P2 stays flat/low → "throttled by cap" hypothesis closes honestly, with numbers; the only
  remaining road to a cognitive engine is Path A (the L1 layer).
- P4 ≥ 20% → reinforces the reliability note: ingest variance dominates even at 33 sessions.
