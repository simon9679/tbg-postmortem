# v1.2 Pre-Registration — Phase 1 (flags + free replay experiments)

*Registered 2026-07-10, BEFORE any experiment numbers. Both prediction sets recorded; where
they differ, that is the methodological point (opposite bets settled by data).*

## Substrate (from Step 0)
- **Gate cache is final-only** — `esmem_gate_state.json` `mem[cid]` holds render STRINGS
  (`trunc/summary/rag_corpus/tracker/tbg`), no per-turn raw responses/deltas. A full gate
  replay is impossible.
- **The demo scenario IS 0-LLM replayable** — `demo_prov_opref1.json` holds 13 RAW turns with
  `raw_facts` (28) + `raw_edges` (14); replay = `_sdl.resolve(raw_facts, user_text, tbg,
  raw_edges)` + `engine.apply_delta`, no LLM. Experiments run on the demo (TZ option b).
- **Caveat baked in:** the demo is 14 turns. It shows *direction*; magnitude claims (esp. the
  history cap) need the long p8 (33 sessions) — that is Phase 2 (a paid re-ingest).

## Flags shipped (Part A, frozen files, default OFF = byte-identical)
- `TBG_HISTORY_CAP` (default 5) — `CONFIDENCE_HISTORY_MAX` now `int(os.getenv(...,"5"))`.
- `TBG_FIX_LABEL_COLLISION` (default 0) — on duplicate lowercased label in edge resolution,
  keep higher-confidence node (tie → older `created_at`) + warn; off = original last-wins.
- `TBG_DECAY_USE_LOGICAL_CLOCK` — pre-existing, unchanged; enters the matrix.

## Predictions (registered before numbers)

### Experiment A — clock non-determinism
- **Architect:** wall-clock non-determinism contributes ≈ negligibly to graph differences on
  replay (**<0.05 aggregate**) — half-lives are in days, runs in minutes. Confidence ~70%.
- **Code (reviewer):** **agree, stronger.** Decay needs `dt ≥ MIN_DECAY_INTERVAL_DAYS = 0.01`
  (~14 min); realistic inter-turn gaps (seconds–minutes) never cross it, so the wall-clock
  series should be **exactly 0** diff, not just <0.05. The logical-clock series must be
  **exactly 0** by construction (a correctness test of the flag). Divergence should appear
  ONLY when injected gaps exceed ~14 min. Confidence ~85%.

### Experiment B — history cap 5 → 25
- **Architect:** `amf_ambiv` revives on some nodes but **max < 0.2** — partial revival,
  insufficient to promote AMF to core. Confidence ~60%.
- **Code (reviewer):** **agree on "insufficient", more pessimistic on magnitude.** On the
  14-turn demo few nodes accrue >5 updates, so cap 25 barely binds — I expect **max amf_ambiv
  < 0.15** and most nodes unchanged. The demo is too short to fairly test the cap; a real
  verdict needs the 33-session p8 (Phase 2). Confidence ~65%.

### Experiment C — label collisions
- **Architect:** collisions rare in real graphs (**<5% of edge resolutions affected**) —
  concept_id dedup catches most. Confidence ~65%.
- **Code (reviewer):** **agree, already confirmed with room to spare.** Step 0 measured the 16
  gate graphs: **0 collisions, 0/188 edge resolutions exposed (0%)**. So the fix flag changes
  nothing on real data; the demo replay with the flag ON should diff **0 edges**. The bug is
  real in code but empirically never fires (concept_id + label-echo). Confidence ~90%.

## What each outcome means
- A ≈ 0 → clock is **not** a fourth noise source; drop it as a v1.2 lever (keep logical clock
  only for determinism guarantees, not noise reduction).
- A > 0 at realistic gaps → new measured noise source for the reliability note.
- B revives AMF meaningfully (max ≥ 0.2 with more nonzero nodes) → history cap is a real v1.2
  win; schedule the paid p8 re-ingest to confirm at length.
- B stays ≈ flat → cap is not the lever on short data; verdict deferred to Phase 2.
- C = 0 diff → close the collision fix as "correct but inert"; no pre-registration cost to ship.
