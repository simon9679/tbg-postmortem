# Ballast primitives — what is proven (internal; NOT published)

The boundary for any future change. Columns: **mechanics proven?** (does the math do what it
claims) · **stable under re-ingest?** · **external usefulness measured?** · **tier / verdict**.
Each cell cites the experiment or says "not measured".

| primitive | mechanics proven? | stable under re-ingest? | external usefulness? | tier / verdict |
|---|---|---|---|---|
| **confidence (log-odds update)** | yes — sign clean 253/0, 0 sign errors (sign diagnostic) | **no** — values swing ±0.40, 94.7% label symdiff (Phase 2 P4) | not measured in isolation; part of graph that lost aggregate (gate NO-EDGE) | mechanics-sound, ingest-unstable, external-unproven |
| **evidence mass (pos+neg)** | yes (accumulation math) | inherits ingest instability | not measured | mechanics-sound, unproven |
| **trajectory (confidence_history)** | yes, but window-limited | no (ingest-unstable + cap-throttled) | **negative** — fed the ranking that a blind reader could not confirm (Test 1 T1-B 2/5) | not externally useful as ranked |
| **turning point (cascade magnitude)** | yes; decay-induced false TPs exist, fixed under `TBG_FIX_DECAY_TP` (Phase 1 E1) | not measured directly | negative via ranking (Test 1) | mechanics-fixed(flag), external-negative |
| **conflict edge (contradicts, rank-render)** | yes — surfaces sub-0.5 edges; renders correctly | edges land ~0.41, one draw; content varies by ingest | **parity/redundant** — state block over RAG +0.10, answerer cited it 0/60 (Gate E2) | redundant with retrieval |
| **ambivalence (tanh min(pos,neg)/scale)** | yes (Priester&Petty form) | inherits ingest instability | not measured | mechanics-sound, unproven |
| **oscillation (Thompson signature)** | yes — uses full history (cap-sensitive: 4→6 at cap 5→25) | not measured | not measured | mechanics-sound, unproven |
| **decay (log-odds → baseline)** | yes | clock contributes ≈0 at realistic timing (Phase 1 A) | not measured | mechanics-sound, negligible-noise |
| **AMF (variance filter)** | filter math ok, but **AMF_WINDOW=5** makes it read only 5 points | irrelevant — bit-identical 0.0198 at cap 5 vs 25 (Phase 2) | ≈0 signal on all data seen | **dead** — mechanistic; not a cap question |
| **ranking heuristic (Σ down-traj+conflicts+tp+osc)** | deterministic | not measured | **split**: separates trivial-vs-nontrivial (T1-A PASS, dummies sank) but a blind reader could NOT confirm the *ordering* among real dialogues (T1-B FAIL) | discriminates trivial only; not a useful ranking |
| **extraction contract (polarity/attribution/birth)** | **yes — 0/191 errors** on shipped config (Phase 0), sign 253/0 | n=2 dialogues, 1 model — boundary of applicability | n/a (correctness, not a product) | **proven-clean** (narrow) |

## Reading
- The only column that is broadly green is **mechanics**. **Stability-under-re-ingest is red or
  unmeasured everywhere**, and **external usefulness is negative (trajectory/TP/ranking/conflict)
  or unmeasured**. Two primitives are closed hard: **AMF (dead)**, **conflict-as-added-signal
  (redundant)**.
- What is genuinely proven and durable is **not a primitive** — it is the *measurement of these
  primitives' instability* (the reliability layer). See `MEASUREMENTS.md`.
- Any future change touching a primitive must state which cell it moves and cite a new experiment;
  a mechanics improvement that does not move the stability or usefulness cell is not a product win.
