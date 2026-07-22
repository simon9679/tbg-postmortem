# v1.2 Phase-1 experiment results (0 LLM, demo replay)

Substrate: replay of the 14-turn demo from `demo_prov_opref1.json` (raw_facts/raw_edges) +
`demo_scenario_ru.json`. Engine unmodified; module globals varied (≡ the env flags, which are
separately byte-identical at OFF). Predictions in `PREREG_v1_2.md` (registered before these).
Dumps: `_v12/exp{A,B,C}.json`.

## Experiment A — clock non-determinism

| comparison | max \|Δconf\| | mean | node symdiff |
|---|---|---|---|
| wall 0s vs 60s | 0.0000 | 0.0000 | 0 |
| wall 0s vs 600s | 0.0012 | 0.0002 | 0 |
| **logical vs logical (determinism)** | **0.0000** | 0.0000 | 0 |
| wall 0s vs **5-day gap** (extreme) | 0.2433 | 0.0996 | 0 |

- Conflicts stable across realistic wall gaps: **true**.
- **Verdict:** at realistic inter-turn timing (< the ~14-min decay interval) clock contributes
  **≈ 0** (max 0.0012). Divergence appears only at large gaps (5-day → 0.24) where decay fires.
  Logical clock is **exactly deterministic** (0.0) — the flag is correct.

| prediction | result |
|---|---|
| **Architect:** <0.05 aggregate, ~70% | **confirmed** (realistic max 0.0012) |
| **Code:** exactly 0 realistic + logical 0, ~85% | **confirmed, exact** (logical 0.0; realistic ~0) |

→ **Clock is NOT a fourth noise source at realistic timing.** Keep logical clock for
determinism guarantees, not for noise reduction. (It becomes relevant only if ingest turns are
genuinely days apart in wall time — which the gate ingest never was: it ran in minutes.)

## Experiment B — history cap 5 → 25

| metric | cap=5 | cap=25 |
|---|---|---|
| amf_ambiv max | 0.0319 | 0.0319 |
| amf_ambiv nonzero | 4 | 4 |
| oscillating nodes | 1 | 1 |
| max history length | 4 | 4 |
| **nodes with >5 history points** | **0** | **0** |

- **Verdict:** cap 5 vs 25 are **identical** — because on a 14-turn demo **no node ever reaches
  5 history points** (max = 4), so the cap never binds. AMF stays ≈ 0 (max 0.032). The demo is
  structurally too short to test this lever.

| prediction | result |
|---|---|
| **Architect:** revival but max < 0.2, ~60% | direction confirmed (no revival); cap didn't bind |
| **Code:** max < 0.15, cap barely binds, demo too short, ~65% | **confirmed, and stronger — cap did not bind AT ALL** |

→ **History cap is NOT testable on the demo.** Verdict deferred to **Phase 2**: a paid re-ingest
of the long p8 (33 sessions) is the only substrate where nodes accrue >5 updates and the cap
5→25 can actually change amf_ambiv. Do not promote AMF or claim a cap win on this evidence.

## Experiment C — label collisions

- Gate graphs (Step 0): **0 collisions**, 0/188 edge resolutions exposed (**0%**).
- Demo replay base vs `TBG_FIX_LABEL_COLLISION=1`: 12 edges vs 12, **edge set identical by
  label: true**, conflicts identical: true.
- **Verdict:** the fix is **inert on clean data** — concept_id dedup + label-echo already
  prevent collisions. Real bug in code, empirically never fires.

| prediction | result |
|---|---|
| **Architect:** <5% affected, ~65% | **confirmed with room** (0%) |
| **Code:** 0% (Step 0) + fix diffs 0 edges, ~90% | **confirmed, exact** |

→ **Ship the collision flag as "correct but inert"** — safe, no behavioral change on real data.

## Bottom line for v1.2
- **A:** clock ≈ 0 at realistic timing — drop as a noise lever; logical clock kept for determinism.
- **B:** history cap untestable on demo; the one genuine "revive AMF" hypothesis **survives**
  but needs the **Phase-2 paid p8 re-ingest** to get a real number.
- **C:** collision fix inert — shippable at zero risk.
- The only Phase-2 spend that could still return "TBG as intended" is the p8 re-ingest for the
  history cap. Clock and collisions are closed here, for free.
