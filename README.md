# Temporal Belief Graph (TBG)

> **This repository is a negative result.** The original hypothesis failed. The value of the
> project is the evaluation methodology that emerged from that failure.

> **Built with Claude Code, by someone who isn't an engineer.** I had the idea and the
> evaluation discipline; an AI assistant wrote the code. If you find a bug or a math error,
> you're probably right — tell me. Numbers are reproducible from frozen artifacts so they can
> be checked. Full note at the top of [FULL_HISTORY.md](FULL_HISTORY.md).

## What is this?

**TBG** is a deterministic belief-memory layer for LLMs. Instead of storing *facts* (what the
user said), it tracks *how a user's beliefs change over time* — confidence per belief,
supporting/contradicting evidence, contradictions, trajectories, and turning points. Built
solo over ~3 months (2026).

## What was I trying to prove?

That a belief-centric memory layer lets an LLM reason about a changing user **better than
conventional retrieval (RAG)**.

## Did it work?

**No.** Four independent, pre-registered evaluations against strong baselines — not one
produced a reproducible win.

```
        Idea
          │
   Implementation
          │
  4 pre-registered evaluations
          │
       NO edge
          │
     Investigation
          │
 extraction variance dominates evaluation
          │
 a falsification protocol for memory systems
```

## The demo that fooled me — and why that's the whole point

The engine turns a dialogue into a *live belief trajectory*. The one that convinced me:
Ebenezer Scrooge, verbatim Dickens, on a strong model — the cynical beliefs hold, then collapse
on exactly the right lines:

```
[ 1] Bah! Humbug! Christmas is a fraud.
     Christmas is a fraud                     85%  NEW
[ 3] Let the poor die and decrease the surplus population.
     social responsibility is not my concern 91%  NEW
[ 8] My clerk's family is poor, I gave them nothing. I see what I am.
     social responsibility is not my concern 55%  ▼37% ⚡
[10] I am not the man I was. I can change. I will.
     Christmas is a fraud                     54%  ▼38% ⚡
     social responsibility is not my concern  6%  ▼31% ⚡
     commits to year-round joy                89%  NEW
```

It *looked like understanding*. It was **rented from the LLM**: **100% of the opposition edges
were proposed by the model and 0% by my deterministic graph — and those edges account for ~85%
of every confidence drop** (the rest is passive decay). That gap — a convincing demo vs what
measurement actually showed — is the whole story.

*(The trace above and the provenance numbers are two separate ingests of the same dialogue, so
their exact figures and even the belief labels differ — see [FULL_HISTORY.md](FULL_HISTORY.md)
§5. That they differ is itself the finding: with no canonical concept identity, every ingest
invents its own vocabulary.)*

## Key findings

- **ES-MemEval (primary benchmark):** TBG was statistically indistinguishable from a plain
  prompt state-tracker and *behind* lexical BM25 retrieval in aggregate — **NO-EDGE**.
- **Both headline wins evaporated on re-extraction.** A conflict-detection edge (+0.40) and a
  long-conversation edge (+0.60) collapsed to ~0 when the belief graph was rebuilt from the
  *same* dialogue — they were single draws from a noisy process, not stable effects.
- **Evaluation variance is dominated by *extraction*, not the judge or the answering model.**
  Ingest noise ≈ ±0.40 (on a 0–2 scale) — roughly **4–8×** larger than judge noise (≈0.10) or
  answerer noise (≈0.05).
- **Most of the "reasoning" was the LLM's, not the graph's.** Provenance tracing: **85%** of
  belief-confidence drops came from contradiction edges the LLM itself proposed; the
  deterministic Python opposition machinery produced **0**.
- **What held up under attack:** constant-cost memory at oracle-equivalent quality; a
  sign-consistent, semantically clean extraction contract; and — most durably — the
  **evaluation protocol** itself.

## The transferable result: a falsification protocol for memory systems

The project's most reusable output is not the architecture — it is a cheap-to-expensive
procedure for deciding whether a memory system's advantage deserves trust *before* running an
expensive comparison:

```
1. dataset validity (cheap canary)   →
2. reproducibility (re-ingest)        →
3. noise decomposition (judge/answerer/ingest) →
4. provenance of intelligence (LLM vs architecture) →
5. judge calibration (blind human relabel) →
6. only now — architecture comparison
```

Each step can stop the process before money is spent on the next. In this project step 1
killed one experiment, step 2 erased both headline wins, and step 4 showed the semantics were
the LLM's. The 8 rules with the number behind each, standalone and ~5 minutes:
**[`FALSIFICATION_PROTOCOL.md`](FALSIFICATION_PROTOCOL.md)**. The full case study that produced
them: **[`FULL_HISTORY.md`](FULL_HISTORY.md)**.

## Read the full story

**[`FULL_HISTORY.md`](FULL_HISTORY.md)** — the complete write-up: every experiment, number,
method, and closed research branch, with the evaluation protocol in full.

## Take it further

I couldn't make TBG beat retrieval — but the measurements leave **two doors open**, and both are
cheap to *falsify* with the protocol, so you'd know fast rather than guessing from a demo:

- **Train the extractor instead of renting it.** A small model fine-tuned on in-domain dialogue
  could give the deterministic core the *canonical* concept identity an off-the-shelf embedder
  can't — the thing whose absence makes the whole graph unstable (why:
  [FULL_HISTORY §4](FULL_HISTORY.md)). Needs a labeled corpus + compute I didn't have.
- **Run the one regime that should favour it.** Constant-cost belief memory should beat retrieval
  on very long, multi-session dialogues that overflow the context window — the single case never
  run (thousands of calls, out of budget; see [FULL_HISTORY §15 / §20.1](FULL_HISTORY.md)).

If you open either door, use the same procedure that falsified the original idea: pre-register →
re-ingest → decompose the noise → trace the provenance ([`FALSIFICATION_PROTOCOL.md`](FALSIFICATION_PROTOCOL.md)).
Forks, PRs, and "I tried it and it still doesn't work" reports are equally welcome — that's the
point of publishing a negative result with the code attached.

## Repository layout

```
tbg_engine / tbg_extractor / tbg_schema     the belief-graph engine (deterministic core)
tbg_axes / tbg_nli / fact_engine / ...       legacy extraction/resolution — present but NOT wired
                                             into the shipped pipeline (failed the concept-identity
                                             wall; see FULL_HISTORY §4 / §17)
api.py, mode_/dissonance_/intervention_      cognitive layer — present, but wired only into a
                                             standalone api.py, never into the product/demo/gate
                                             pipeline (see FULL_HISTORY §6)
memory_bench / pressure_matrix / ...         benchmarks and experiments
tools/                                        probes (embedder / resolver / attribution)
evidence/                                     frozen artifacts behind every cited number
test_*.py                                     engine regression + drift tests (repo root)
```

The **`ballast`** branch holds the later *productization* attempt (an anti-sycophancy belief
anchor + policy layer) — a cleaned fork of this engine.

## Reproduce

Nearly every number in `FULL_HISTORY.md` reproduces from frozen artifacts without re-running
the LLM pipeline. For example, the headline provenance result regenerates with **no LLM calls**:

```
py -3 tools/attribution_run.py --analyze    # recomputes the 85/0 split from evidence/attribution_{A,B}.json
```

Live runs need an LLM provider key in the environment (see `env.example`); every environment
flag (defaults, read-time vs import-time) is documented in [`evidence/FLAGS.md`](evidence/FLAGS.md).

## Limitations

Small n throughout (subset n ≈ 10–20; some experiments n = 1 dialogue / 1 model family). Single
answerer/judge model on the main gate. These bound how far each result generalizes — read the
caveats in `FULL_HISTORY.md`; do not over-generalize any single number.

## Citation

No formal publication or DOI — this is a repo, not a paper. If the **evaluation protocol** or
these findings are useful in your own work, a link back is plenty:
`github.com/simon9679/tbg-postmortem` (TBG: a negative-result study and a falsification protocol
for memory systems, 2026).

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Support

No product or framework is being sold here — this is a negative-result research write-up.
If the evaluation protocol, or the write-up, was useful, you can toss a coffee toward future
experiments (entirely optional). GitHub Sponsors / Ko-fi are unavailable to recipients in
Ukraine, so the frictionless channel is crypto — send from any wallet or exchange:

- ₿ **USDT (TRC-20):** `TWS9EdrEx8A34bnAdDrDznywigWdNfJgt3`

<!-- optional card channel (Ukrainian donation page, pay by card in any currency):
☕ https://donatello.to/<YOUR-NAME>
-->

A tip from a curious reader just means the work was useful to someone — which, for a repo like
this, is the real signal.
