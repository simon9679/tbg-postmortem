# TBG — `ballast` branch (productization attempt)

This branch is **not** the main study — start with [`main`](../../tree/main) and its
`FULL_HISTORY.md`. This is the later, cleaned *productization* fork of the Temporal Belief Graph
engine: an anti-sycophancy **belief anchor + policy layer** carved out of the full research
codebase.

- `policy.py` — the two-sided anti-sycophancy layer (`HOLD` / `ALLOW_UPDATE` / `PASS`),
  deterministic 0-LLM decision logic.
- `ballast_signals.py` — a read-only, structured JSON snapshot of a belief graph (0 LLM): the
  "give the model the user's *state*, not the whole transcript" idea.
- `esmem_gate.py` / `esmem_adapter.py` — the ES-MemEval gate harness.
- `evidence/` — reports for this branch (`REPORT_phase1_policy.md`,
  `REPORT_pressure_gate.md`, …).

**Why it exists, and why it's here.** On the main branch, the anti-sycophancy experiment is the
pressure-gate (`FULL_HISTORY.md` §12) — it came back **inconclusive** because the benchmark
dataset could not exert real pressure on a strong model (a pre-registered canary caught this).
The cognitive/policy layer was built and unit-tested but **never shipped into the product
pipeline** (§6). This branch is the cleaned code those sections describe, published for
completeness — not as a working product. The honest headline and the transferable result (the
falsification protocol) are on `main`.
