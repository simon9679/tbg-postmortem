# v1.2 Phase-2 results — history cap on long p8 (33 sessions)

One LLM ingest (33 responses captured) + two 0-LLM replays. Predictions in `PREREG_v1_2_phase2.md`.

**Determinism control:** capture(cap5,real) == replay(cap5,stub): **True**

## Cap effect (replay cap5 vs cap25, same responses)

| metric | cap=5 | cap=25 |
|---|---|---|
| nodes | 50 | 50 |
| max history length | 5 | 18 |
| nodes with >5 history | 0 (0%) | 2 (4%) |
| amf_ambiv max | 0.0198 | 0.0198 |
| amf_ambiv nonzero | 17 | 17 |
| oscillating nodes | 4 | 6 |
| conflicts identical (cap5==cap25) | True | |

## P4 — ingest variance (fresh p8 ingest vs frozen v1.1 p8)
- fresh nodes=50, frozen nodes=50, symdiff by label=90 → **94.7%** of union differs.

## Verdict vs predictions
- **P1 (cap binds):** nodes>5 at cap25 = 2/50 (4%). <30% -> below architect P1.
- **P2 (AMF revives, max<0.2):** cap25 amf max = 0.0198, nonzero 17 (cap5: 0.0198/17). max<0.2 holds
- **P3 (oscillations 2x):** cap5=4 cap25=6 (<2x).
- **P4 (>=20% symdiff):** 94.7% (holds).
## KEY MECHANISTIC FINDING (new — predictions missed this)
`amf_ambiv` is **identical** (0.0198, 17 nonzero) at cap=5 and cap=25 — even though 2 nodes
now carry 18-point histories. Reason: `amf_filter.compute_node_amf` reads
`confidence_history[-AMF_WINDOW:]` with **`AMF_WINDOW = 5`** (amf_filter:14,33). AMF only ever
looks at the last 5 points, so the history cap is **irrelevant to AMF**. 

→ The Phase-2 hypothesis ("the intended cognitive layer / AMF was throttled by history_cap")
is **FALSE at the mechanism level**: AMF is throttled by its OWN 5-point window, not the cap.
Raising `TBG_HISTORY_CAP` 5→25 changes AMF by exactly 0. The correct knob, if any, is
`AMF_WINDOW` — outside this pre-registration's scope, and itself dubious given AMF's formula
saturates low (max achievable ~0.23).

Oscillation detection DOES use full history (not the 5-window), so it moved (4→6, +50%); AMF
did not. Nodes-over-5 is only 4% (2/50) even on 33 sessions — MAX_NODES churn + a broad,
one-off-heavy belief set mean concepts rarely get revisited >5 times.

## Bottom line
- **P1 fails** (4% « 30%): even 33 sessions rarely revisit a concept >5×.
- **P2 fails, with a mechanistic reason:** AMF is window-5 internally; history-cap cannot revive
  it. This **closes** the "throttled by config" hypothesis honestly, with a code-level cause.
- **P3 partial:** oscillations +50% (cap-sensitive), not 2×.
- **P4 holds hard:** 94.7% label symdiff on a fresh vs frozen p8 ingest — ingest variance on
  long conversations is enormous at the surface (label) level. (Caveat: label-level overstates
  concept-level divergence via paraphrase; but the surface the answerer/judge sees genuinely
  differs ~95%.)
- **Consequence:** the last cheap lever to "TBG as intended" is closed. AMF revival is not a
  history-cap question. The remaining road to a cognitive engine is Path A (the L1 layer), not
  a config flip. Reliability note gains a 5th data point (ingest variance ~95% at 33 sessions).
