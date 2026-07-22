# Pre-Registration — Gate E2 vs C (state-block over RAG)

*Co-authored (architect + Code). Registered BEFORE the prompt design / any E2 LLM call.
Both bet sets fixed. 0 new ingest.*

## Basis (updated after Phase 0)
Phase 0 measured the extraction contract at **0% P/A/B error** (191 decisions, op/ref +
evidence_type). So E2 does **not** compete on "we extract more accurately." E2 tests one thing:
**does a compact conflict/state block — what RAG cannot compute — raise conflict-detection
without hurting aggregate?** Both of C's weak spots are measured, not asserted (ingest variance
94.7% label-symdiff at 33 sessions; judge retrieval-style premium +0.33).

## Design (isolates exactly Δ = the state block)
- **C arm:** BM25 retrieval over the cached `rag_corpus` → answer. **Reused from `gate_state`
  (60 scored, incl. 20 conflict) — zero new calls.**
- **E2 arm:** the SAME BM25 retrieval over the SAME `rag_corpus` + a **state block**, inside the
  SAME `REPR_BUDGET = 4500`, split **retrieval 3700 / state 800**. → new answer → new judge.
- **State block = 0 new ingest:** rendered from the **frozen v1.1 graph** `gate_graphs/<cid>.json`
  (rank-render CONFLICTS + trajectories). A fixed PRIOR draw of the same sessions — not a live
  extraction (which would be a new ingest / lottery). Same answerer prompt, same official judge,
  temp = 0.
- **Fixed knobs (no tuning):** `S = 800` is the ONLY split point — no post-hoc S search;
  S = 400/1200 would be SEPARATE pre-registered runs, not tuning inside this one. **One split
  (3700/800) for ALL abilities** — no per-ability adaptivity (temporal/user_model get the same
  state block appended, so the AGG guard is computable).

## Lower-bound framing (mandatory)
E2 here uses the frozen graph's draw, which can diverge from the bench's gold conflicts
(ingest variance 94.7%). The PRODUCT scenario computes state on the *same* ingest as memory
(one draw for both). Therefore **E2 in this design is a LOWER BOUND on the state block's value.**
- WIN even on a non-own draw → strong signal.
- PARITY → part of the blame may be draw/gold desync; say that honestly, but do **not** turn it
  into an excuse (report it as one contributing factor, weighed against redundancy).

## Outcomes (n = 20 conflict, same judge, same items)
- **WIN:** E2 − C ≥ **+0.30**
- **LOSS:** E2 − C ≤ **−0.10**
- else **PARITY**
- **AGG guard:** AGG(E2) ≥ AGG(C) − 0.10 over all 60. If violated, a conflict WIN is **nullified**
  (we must not kill C's fact-recall to buy conflict).

## Bets (fixed before prompt design)
| | E2−C ≥ +0.30 (conflict) | AGG guard holds | named main risk |
|---|---|---|---|
| **Architect** | ~50% (honest fifty-fifty post-Phase-0) | ~70% | judge **style** premium (+0.33) eats the state block's contribution |
| **Code** | **~30%** (WIN 30 / PARITY 50 / LOSS 20) | ~60% | **redundancy + wrong-draw**: BM25 already retrieves the raw conflicting lines → answerer often detects the conflict from text → state block redundant; where non-redundant, the frozen draw diverged from gold (the 94.7%) → misleads. The split also steals raw text from temporal → AGG risk. |

Opposition is live and the risks are DIFFERENT (architect = judge premium; Code = redundancy +
wrong-draw). The per-item dump resolves both.

## Style-premium control (mandatory metric)
Our bias is **style, not length** (K=20: TBG longest yet least inflated). The state block may
push E2 answers toward a declarative/state-digest style — exactly what the judge rewards
(tracker +0.25). So the per-item dump records, per E2 answer: **cites state-block / cites
retrieval / both**, and RESULTS breaks the score down by this field. (Answer-length E2 vs C is
also logged, but the style field is the primary control.)

## Pre-declared tie-breaker
If the outcome is **borderline** (narrow WIN just over +0.30, or PARITY near either edge), a
**mini K = 10 blind manual relabel** of conflict items (same K-relabel mechanics as before,
scores hidden) decides. Declared now so it is not invented after seeing numbers.

## Cost / hygiene
~120 calls (E2: 60 answers + 60 judges; C reused, 0 calls). Detached, resumable, dumps to
`_v2gateE2/`. Frozen states (`gate_graphs`, `.v1_0.frozen`, `gate_state` C entries) read-only.

## Addendum (0-LLM pre-count, before launch) — third interpretation axis
Content of the frozen state blocks on the 10 conflict-convs:
- **7/10 convs have explicit opposition edges** (p12,p13,p15,p17,p18,p6,p7); edges 0.49–0.71
  (rank-render surfaces sub-0.5, e.g. p17@0.49). **3/10 are trajectories-only** (p14,p5,p8 —
  physically 0 opposition edges, verified: NOT a render filter; rank-render has no threshold).
- **Conflict items: 16/20 sit on with-edges convs, 4/20 on trajectories-only.**

New per-item axis in RESULTS: **"state block had an explicit CONFLICTS section vs trajectories-
only."** This is CONTEXT, not a gate. It pre-separates two PARITY readings:
- PARITY even on the 16 with-conflict-edges items → "the state block itself doesn't help."
- PARITY only on the 4 trajectories-only items → "a block without conflict sections doesn't
  help" (leaves the conflict-edge value open) — but note the counter-precedent below.

**Symmetry of our bets, made explicit:** the v1.0 attribution found TBG's conflict win (+0.40)
was achieved by **node dynamics with NO rendered conflict lines** (gate renders had none). So
"trajectories without explicit conflict edges" is exactly the configuration in which E once won
conflict. **Code's bet is reinforced by block content (16/20 do have edges, yet redundancy);
Architect's is reinforced by the v1.0 precedent (trajectories alone once sufficed). Data decides.**
