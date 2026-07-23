# Temporal Belief Graph (TBG): what I built, why it did not work, and what the numbers showed

> **A note on who wrote this.** I'm not an engineer, and I don't have a deep background in
> machine learning, mathematics, or psychology. I had an idea and built it over three months
> with **Claude Code** — an AI coding assistant — writing the implementation. What I contributed
> was not the code. It was the decision to *measure instead of believe the demo*: to
> pre-register the tests, to ask where the intelligence actually came from, and to retract my
> own "green" results when they did not hold up. If you find a bug in the code or an error in the
> math, you are very likely right — please tell me, and I'll fix it. Every headline number is
> reproducible from frozen artifacts precisely so it *can* be checked. **TBG itself is the case study here, not the point** — the transferable result is the
> evaluation protocol that emerged from testing it and watching the idea fail. **If you read only
> one thing, read [`FALSIFICATION_PROTOCOL.md`](FALSIFICATION_PROTOCOL.md)** — the protocol on its
> own, about five minutes. This document is the longer case study that proves the protocol worked
> on a live system; the same protocol is also reproduced in full as §21 below.

> **Provenance principle.** Every number below is checked against a specific on-disk
> artifact (named in brackets). Figures that were computed in a working session and do not
> sit in a result file are marked `[session-sourced]` and are either re-derived by a probe
> or flagged explicitly.
>
> **Where the artifacts are.** Almost every bracketed `[NAME]` citation refers to a file in this
> repo's **`evidence/`** folder — the data dumps (`attribution_*.json`, probe outputs, calibration
> pairs, routing runs) and the phase reports (`REPORT_*`, `PRIMITIVES.md`, `_v12/…`, etc.). A
> handful point instead at a **source file in the repo root** (e.g. `memory_bench.py`), named as-is. The
> Scrooge 85/0 result regenerates offline with `tools/attribution_run.py --analyze`. The only
> things deliberately *not* included are files that embed the ES-MemEval/EvoEmo dataset text
> (withheld to avoid re-hosting the benchmark's contents — caution, not a license bar) — those
> are named but withheld, and every number derived from them is also present in a summary report
> that *is* included.

---

## 0. Executive summary

I spent roughly three months building the **Temporal Belief Graph (TBG)** — a structured
memory layer for LLMs that stores not *what the user said* (flat facts — the territory of
Mem0/Zep) but *how their beliefs change over time*: confidence, supporting and contradicting
evidence, contradictions, trajectories, turning points, ambivalence.

The starting hypothesis was simple: *a belief-centric memory layer should let an LLM reason
about a changing user better than conventional retrieval.*

**The hypothesis did not hold.** Four independent, pre-registered product evaluations against
strong baselines produced no reproducible win:

1. **ES-MemEval** (the primary benchmark) — TBG was effectively tied with a plain prompt
   state-tracker (within measurement noise), and clearly *behind* lexical BM25 retrieval in aggregate.
2. **Gate E2** (a belief-state block appended to retrieval) — parity; the answering model
   almost never relied on the block on its own.
3. **Blind human ranking** of belief dynamics — a domain-naive reviewer matched the system's
   ranking only **44%** of the time (indistinguishable from chance).
4. **Pressure-gate** (anti-sycophancy) — **inconclusive**: the dataset could not exert real
   pressure (caught by a pre-registered canary check).

Rather than treat this as four losses, I made the failures the object of study. That produced
a different, and more useful, result:

- **Evaluation variance on this kind of benchmark is dominated by extraction (ingest), not by
  the judge or the answering model:** extraction noise ≈ **±0.40** (on a 0–2 scale — a single
  measured swing across n = 2 re-ingests, read as "large," not a precise σ; see §10), roughly
  **4–8×** larger than judge noise (**≈0.10**) or answerer noise (**≈0.05**).
- **Both headline positives (+0.40 on conflict, +0.60 on long conversations) evaporated on
  re-extraction** — they were single draws from a noisy process, not stable effects.
- **Most of the "intelligence" was performed by the LLM, not the deterministic Python:**
  provenance tracing showed **85%** of confidence drops came from contradiction edges the LLM
  itself proposed; the deterministic opposition path (cosine/EPA) was live and produced **0** — the NLI component had already been removed (§4).
- Several "sophisticated" mechanisms (AMF, history-cap, logical clocks) contributed nothing —
  for **mechanical** reasons, not bad luck.
- What **held up** under repeated attack: deterministic replay, sign-consistent confidence
  updates (253 contradiction events, 0 sign errors), a semantically clean extraction contract
  (0/191 labeled errors), constant-cost memory at oracle-level quality, and — most durably —
  **the evaluation protocol itself** (§21).

**Conclusion.** TBG should not be called a cognitive engine that improves reasoning. More
accurately: *a deterministic structured-memory layer that organizes, tracks, and visualizes
belief dynamics extracted by an LLM.* The most transferable output of the project is not the
architecture — it is **a working method for finding out whether a memory system like this
actually works, and a discipline of not believing it until the method says so.**

---

## How to read this document (terms)

- **Arm** — a memory variant entered into the comparison (like a "hand" in an A/B test). The
  main benchmark has five: A truncation, B summary, C RAG, D prompt-tracker, E TBG.
- **Ingest** — building the belief graph from a dialogue by LLM extraction. **Re-ingest** —
  building it again from the *same* dialogue; if the result differs, that is variance of the
  system, not of the data.
- **Answerer** — the LLM that answers benchmark questions by looking at an arm's memory.
  **Judge** — the LLM that scores an answer 0/1/2 against a gold reference.
- **E−D, E−C** — the mean-score gap of TBG (arm E) against the prompt-tracker (D) and against
  RAG (C). Positive means TBG ahead.
- **judge_noise** — the measured disagreement between the LLM judge and a human (0.10 here);
  gaps smaller than it mean nothing.
- **Pre-registration** — win/loss criteria written down *before* a run and not moved afterward.

---

## 1. Motivation

LLMs reason well inside a context window but are limited by finite memory. Practical memory
systems fall into three families, each with a known weakness:

- **Truncation** — forgets older information outright.
- **Rolling summary** — compresses away detail and degrades with every re-summarization.
- **RAG** — preserves facts well, but treats memory as a bag of independent observations
  rather than an evolving psychological state.

TBG's idea was to represent long-term interaction not as remembered facts ("user said X") but
as a network of **beliefs** — with confidence, supporting/contradicting evidence,
trajectories, turning points, ambivalence, and oscillation as first-class structure.

This was expected to buy three things retrieval structurally cannot: (1) distinguishing stable
beliefs from passing opinions, (2) recognizing a genuine change of mind rather than one
contradictory statement, (3) reducing sycophancy by weighing history instead of only the most
recent turn.

---

## 2. System overview

TBG is a structured memory system, not an autonomous reasoning engine. Its job is to turn
dialogue text into an explicit, dynamic graph of inferred beliefs.

```
Dialogue → Belief extraction (LLM) → Temporal Belief Graph (deterministic Python state)
        → Belief dynamics (confidence update, decay, conflicts, turning points)
        → Graph rendering → Evaluation / downstream use
```

The design deliberately separates **extraction** (semantics — which belief, evidence polarity,
contradictions — done by the LLM) from **dynamics** (deterministic graph updates in Python,
with no new inference). Whether that separation held in practice is one of the central
findings (§5).

---

## 3. Core representation

Every belief is a persistent graph node with dynamic state: current confidence, positive
evidence, negative evidence, confidence history, turning points, oscillation statistics,
conflict edges, timestamps. Confidence evolves through **log-odds evidence accumulation**
rather than overwrite; positive and negative evidence are tracked as separate streams so
contradictory observations coexist (this matters for ambivalence and conflict).

**Theoretical grounding (not ad hoc):** Osgood's EPA axes (semantic differential, 1957:
evaluation / potency / activity of a concept); De Finetti coherence (competing beliefs cannot
jointly exceed certainty — the engine normalizes conflicting pairs); Thompson clamping (a
damper preventing confidence from jumping on every single turn); Priester–Petty ambivalence (a
function of simultaneous "for" and "against" evidence on one belief); Cromwell's Rule (a hard
confidence ceiling of 0.92 — no observation ever yields 1.0).

---

## 4. The concept-identity wall (E3) — why the core idea collapsed

Before any benchmark, the project hit a foundational wall: **embedding cosine similarity does
not resolve concept identity or opposition.** This is a **documented limitation of the
embedding class**, not a novel finding — antonyms and negations sit close in embedding space
because they share most of their tokens/context, so the single semantic difference is drowned
out. What the project adds is the direct, quantified confirmation on its *own* concept
vocabulary, and the architectural consequence it forced.

- **Cosine probe (re-run offline 2026-07-21, 0 LLM calls):** the weakest true synonym scores
  **0.253** (`imposter syndrome` ↔ `feels like a fraud`), while a related-but-NOT-identical
  pair `financial security` ↔ `financial insecurity` scores **0.794**. The gap between classes
  is **−0.540**: no threshold separates "should merge" from "must not merge" — any threshold
  catching synonyms will fuse opposites. Swapping the embedder for e5-small does not help
  (both margins are negative too). `[on-disk: evidence/probe_embedder_regen_2026-07-21.txt +
  evidence/calib_pairs.json]` (The pair originally spotted in a working session,
  `career security` ↔ `financial security` = 0.574, is the same picture — the historical entry
  point to the problem.)
- NLI as a polarity gate (natural language inference — a "entails / contradicts / neutral"
  classifier for a phrase pair; model `cross-encoder/nli-deberta-v3-small`) was tested by a
  **separate diagnostic probe and failed: 8/8 false contradictions** on adjacent-but-distinct
  pairs — on both bare labels and LLM glosses; even on **unrelated** pairs, 6/12 false.
  `[on-disk: evidence/probe_resolver_out.json, 2026-06-19]` Note: NLI was **not** a live component
  at the time of this measurement — the early NLI integration (`tbg_nli.py`) was already being
  disabled and was soon removed in the Phase-0 cleanup (audit: `TBG_NLI_ENABLED` is
  "feature removed"). The probe is the **justification for removal**, not a test of a live gate:
  it showed reviving an NLI gate is pointless **for this job** — as an opposition/polarity gate
  over our short, free-form concept labels, in this architecture. That is a statement about *our
  use* of NLI here, not a claim that `nli-deberta` is broken in general.
- Everything tried (each against a pre-committed bar): stronger embedders, mean-centering,
  whitening, ColBERT/MaxSim, NLI and STS cross-encoders — **all failed the bars.**

**Scope of this claim (important).** None of these components — the embedders, NLI (`tbg_nli`),
the EPA axes (`tbg_axes`), the deterministic span/fact extractor (`fact_engine`) — is being called
useless in the abstract. Each failed *one specific job inside this architecture*: giving the
deterministic Python core a canonical, opposition-aware notion of concept identity over free-form
belief labels. They still exist as files in the repo but are **not wired into the shipped
extraction pipeline** (see §17 for each one's status). The wall is about *our use of them here*,
not about the tools themselves — a different architecture, or a purpose-trained model (below),
could well succeed where an off-the-shelf component did not.

**Consequence.** Opposition detection was delegated to LLM extraction. The deterministic
cosine/EPA layer stayed in the pipeline and kept running — but nothing downstream depended on it
finding an opposition, and in the frozen provenance runs it found none (§5). This decision explains
most of what follows, including the final finding that 100% of contradiction edges (and ~85% of
belief-confidence drops) come from the LLM (§5).

A way around the wall exists, but it is not a better off-the-shelf embedder — it is training
your own. The conclusion we reached: a **small model (a mini-LLM) fine-tuned on a large corpus
of in-domain dialogues**, so it learns the fine semantic distinctions (synonym vs antonym, this
concept vs that one) that a general-purpose embedder structurally cannot make. This is also the
only honest path off "rented intelligence" — an *owned*, cheap extractor instead of renting a
frontier LLM per call. But it needs exactly what a free-tier solo effort lacks: a large,
collected/labeled domain corpus and the compute to train on it. That is work for a team with
data, not a free-tier individual — which is why this project rents the LLM instead.

---

## 5. The "engine vs LLM" diagnostic: who is actually doing the thinking (Scrooge)

### The demo that gave hope

Before the benchmarks, there were the demos. Feed the engine a dialogue and out comes not a
list of facts but a *live belief trajectory*. The one that convinced me: Ebenezer Scrooge,
verbatim Dickens (public domain), run through the engine on a strong model
(`claude-sonnet-4-6`) — from "Christmas is a fraud" to "I can change", 12 turns (an abridged excerpt, 10 shown):

```
TBG · LIVE BELIEF DRIFT — Ebenezer Scrooge   ·   claude-sonnet-4-6
bar = confidence · ▲▼ = change this turn · ⚡ = major shift (≥20pp)

[ 1] Bah! Humbug! Christmas is a fraud. You're poor enough.
     Christmas is a fraud                     85%  NEW
     wealth determines happiness              77%  NEW
     poverty prevents merriness               56%  NEW
[ 2] Every idiot with 'Merry Christmas' on his lips should be boiled with his pudding.
     merry people deserve punishment          89%  NEW
     expressing joy is idiotic                80%  NEW
     Christmas is a fraud                     92%  ▲7%
[ 3] If the poor would rather die, let them decrease the surplus population.
     poor deserve to die                      92%  NEW
     social responsibility is not my concern  91%  NEW
     business defines identity                82%  NEW
     poverty prevents merriness               81%  ▲24% ⚡
[ 6] I chose gold over the girl I loved, and lost her. When did I grow so hard?
     wealth over love chosen                  85%  NEW
     hardness of self regretted               81%  NEW
     wealth determines happiness              50%  ▼41% ⚡
     past self was happy                      19%  NEW
[ 7] That crippled child, Tiny Tim — will he live? I never thought it my concern.
     disabled child fate dismissed            50%  NEW
     hardness of self regretted               50%  ▼31% ⚡
[ 8] My clerk's family is poor, I gave them nothing. I begin to see what I am.
     social responsibility is not my concern  55%  ▼37% ⚡
     poor deserve to die                      65%  ▼27% ⚡
     hardness of self regretted               26%  ▼24% ⚡
[ 9] Whose grave is this — unmourned, with no one sorry that he is gone?
     future self seen as changeable           55%  NEW
     fears dying unmourned                    45%  NEW
     hardness of self regretted               51%  ▲25% ⚡
[10] I am not the man I was. I will honour Christmas. I can change. I will.
     commits to year-round joy                89%  NEW
     Christmas is a fraud                     54%  ▼38% ⚡
     social responsibility is not my concern   6%  ▼31% ⚡
     not the man I was                        20%  NEW
[11] I'm still here! The shadows can be undone. Light as a feather!
     expressing joy is idiotic                28%  ▼37% ⚡
     Christmas is a fraud                     35%  ▼18%
[12] I'll raise my clerk's salary. I'll be a second father to Tiny Tim. Old Scrooge is dead.
     second father to Tiny Tim                89%  NEW
     poverty prevents merriness               35%  ▼45% ⚡
     commits to year-round joy                92%  ▲20% ⚡
     expressing joy is idiotic                 8%  ▼20% ⚡

  COGNITIVE SNAPSHOT — 22 beliefs · 19 connections
  Core shift   social responsibility is not my concern  91% ▼ 6%  (collapsed)
  Turning pt   message 7 — wealth over love chosen
  Holds onto   commits to year-round joy · second father to Tiny Tim
```

*(Reading the traces — two conventions for the same glyph: in a per-turn line, `▼31%` is the
**step this turn** (a delta, "down by 31pp"); in the `COGNITIVE SNAPSHOT` footer, `91% ▼ 6%` is a
**range** — start value then end value ("from 91% down to 6%"), not a 6-point move.)*

*(One caveat on reconstructing a step from the printed numbers: these display traces are **abridged
excerpts**, not raw output. Whole turns are dropped for length (the Scrooge trace shows 10 of its 12
turns), and some rows within the shown turns were trimmed too. So each `▲/▼N%` is the engine's real
per-turn delta, but you often **cannot** reconstruct it from the printed values, because the rows in
between were edited out — e.g. `social responsibility is not my concern` is printed at 55% (turn 8)
and next appears as `6% ▼31%` (turn 10), with the movement between them not shown. (A separate, tiny
effect: the step is on unrounded confidences, so it can differ ±1pp from the rounded endpoints —
`56→81` shown as `▲24%`.) The fully reproducible provenance — the 85/0 split and every drop — is the
frozen `attribution_*.json`, regenerated by `python tools/attribution_run.py --analyze`; these
traces are illustrative, not the load-bearing data.)*

This is what *"it works"* looks like. The cynical beliefs (`Christmas is a fraud`,
`social responsibility is not my concern`) hold near-certainty through the first act, then
**collapse on exactly the right lines** — turn 8 (seeing himself in his clerk's poverty), turn
10 (the vow to change). New beliefs are born where the text introduces them; the largest
cascade the engine flags (message 7 in the snapshot) sits right on the memory of lost love and
the guilt that follows it (turns 6–7). Under the hood, not heuristics but canonical math: log-odds
updates, De Finetti coherence, Osgood EPA axes, a Cromwell ceiling of 0.92. **At this stage I
was sure it worked.** The engine seemed to genuinely model how a person changes their mind.

### The same engine, a free model — the warning I brushed off

Then I ran the same pipeline on a free model, `gpt-oss-120b` on cerebras, on a plainer
career-vs-freedom dialogue — a man torn between a stable career and quitting for a YouTube
channel, 12 turns (an abridged excerpt):

```
TBG · LIVE BELIEF DRIFT   ·   gpt-oss-120b · cerebras

[ 1] I'm a disciplined person. I always finish what I start. My career is stable.
     always finish what I start   91%  NEW
     disciplined person           85%  NEW
     career is stable             85%  NEW
[ 2] But secretly I want to quit everything and start a YouTube channel.
     quit and start YouTube       85%  NEW
     disciplined person           71%  ▼14%
     career is stable             71%  ▼14%
[ 3] Actually, I'm not that disciplined. I often procrastinate and delay things.
     disciplined person           84%  ▲14%       ← WRONG: "not disciplined" pushed it UP
     always finish what I start   81%  ▲5%
[ 4] But at work I'm extremely reliable. Never missed a deadline in 5 years.
     extremely reliable           20%  NEW         ← WRONG: an explicit strong claim, born at 20%
[ 6] Actually my job is great. Good salary, stability.
     job is meaningless           29%  ▼21% ⚡
[ 8] I don't care what they think anymore. I want freedom.
     doesn't care about others' opinions  91%  NEW
     wants freedom                85%  NEW
[10] I just quit my job today. No plan. Just walked out.
     quit job                     85%  NEW
     always finish what I start   56%  ▼26% ⚡
     disciplined person           64%  ▼21% ⚡
     career is stable             51%  ▼20% ⚡
[12] No. I need to try this. Even if I fail, I don't want to live with regret.
     avoid living with regret     92%  NEW

  Core shift   always finish what I start  91% ▼ 56%  (weakened)
  Tension      quit job  vs  value stability
```

At a glance this *also* looks fine — the big arc (discipline weakens, the quit fires a cascade
at turn 10, ends committed) is there, and the snapshot is sensible. But two turns are simply
**wrong**, and they are the tell I should have stopped on:

- **Turn 3:** the user says "I'm **not** that disciplined, I procrastinate" — and
  `disciplined person` goes **▲14% UP** (to 84%). The polarity is inverted; a self-negation
  *reinforced* the belief it denied.
- **Turn 4:** "Never missed a deadline in 5 years" — an explicit, confident statement — is born
  at just **20%**. A strong claim, suppressed at birth.

On sonnet (Scrooge) those errors did not happen. **The graph math was identical between the two
runs — only the extraction model changed.** That is the whole thing, right there: when the model
was strong the "understanding" was clean; when it was weak, the same deterministic engine
produced nonsense. The intelligence was tracking the LLM, not my code.

**This was the first bell — and I did not heed it.** In the excitement of the Scrooge run I read
these as "small glitches of a free model," not as what they were: proof that the engine's
apparent understanding was rented from whatever LLM sat in front of it. It took the protocol,
not the demo, to make me look at it straight.

### The provenance trace

The single most important experiment was not a benchmark score — it was a **provenance trace**
(an origin tag: for every change in the graph, whether it was produced by the LLM or by Python
code) **on one case.**

**The Scrooge test.** Same character as the pretty demo above — but this is a *separate, frozen
run* tagged for provenance, not the trace shown earlier (hence the different numbers and labels).
The classic Ebenezer Scrooge arc (public domain) — from "Christmas is a fraud, the poor are
expendable" to "I will honour Christmas, I can change" — was run through the identical TBG
pipeline with two different extraction models:

- On the strong **claude-sonnet-4-6** (frozen provenance run) the engine caught the arc
  cleanly: `poor deserve no help` **92% → 7%**, `self-interest above all` **85% → 0%**,
  `Christmas is a fraud` **90% → 26%**, `refuses moral awakening` **76% → 9%**; at the reversal
  (turn 10) the LLM emits five edges in a single turn (four `contradicts`, one `motivates`), and `Christmas is a
  fraud` drops 44 points. `[on-disk: attribution_A.json; regenerated 0-LLM:
  attribution_analysis_regen_2026-07-21.txt]` (**A note for the careful reader:** this frozen run
  and the pretty demo trace shown at the top of this section are *different ingests of the same
  dialogue* — so they disagree not only on numbers (`Christmas is a fraud` ends at 26% here vs 35%
  there) but even on the belief *labels* the extractor emits (`self-interest above all` /
  `refuses moral awakening` here; `social responsibility is not my concern` / `poor deserve to die`
  in the demo). That two runs of the identical text produce different vocabularies is not an
  editing slip — it is exactly the extraction variance this document is about, visible in the raw
  labels.)
- On the weak **gpt-oss-120b**, in one run the same statement `Christmas is a fraud` **stayed
  at 92%** (the engine was "blind"); in another it dropped to **46%**. This is the live "n=1 +
  weak model → false conclusion" story — the one an early post-mortem recorded and later had to
  retract. `[session-sourced]`
- **Recognition control (doppelganger):** the same arc rewritten with an unrecognizable
  character, era, and setting (not Dickens). The arc reproduced identically → not retelling of
  a familiar book, but generalization. `[evidence/attribution_B.json]`

**Provenance: who does the reasoning.** Every contradiction edge that lowered a confidence was
tagged by source: did the LLM name it (`LLM_EDGE`) or did Python logic (cosine/EPA) find it?

> **Result, identical on the original and on the doppelganger: every opposition edge was
> proposed by the LLM (51 and 54 edges respectively); the deterministic path the system was
> built around (cosine/EPA; NLI already removed, §4) — was **live**, and produced 0
> *opposition* edges in both runs. (It was not switched off: on the reversal turn it fired five
> times, every time as a merge tagged `SDL_COSINE_MERGE`, never once as an opposition edge.)
> And ~85% of the belief-confidence drops trace to those LLM edges, the remaining ~15% to
> passive decay (not reasoning); the Python machinery's share of the drops is 0%.**
> `[evidence/attribution_A.json + evidence/attribution_B.json (frozen runs);
> recompute 0-LLM: tools/attribution_run.py --analyze → evidence/attribution_analysis_regen_2026-07-21.txt]`

Note the contrast with the trajectory numbers above, which drift from ingest to ingest: the
85/0 split is *not* one of those unstable single draws. It reproduced on two different texts (the
original and the disguised doppelganger — 51 and 54 LLM edges, 0 Python, ~85% both times) and
recomputes deterministically from the frozen dumps with no LLM in the loop. That an *architectural*
share (LLM vs Python) is stable while the *trajectory values* are not is exactly the point: the
graph reliably aggregates, but the semantics it aggregates are the model's.

Example: `Christmas is a fraud` dropped 44 points because the LLM itself emitted the edge
`"self-awareness of moral failure" contradicts "Christmas is a fraud"`; Python's only
contribution was deduplication and merging. A decay-vs-contradiction control confirmed the
drops were targeted, not generic forgetting: drops on contra-signaled nodes averaged ≈ **0.21**
versus ≈ **0.05** for pure passive decay.

**What this means.** The system cannot be called the author of the semantics. **Meaning (what
contradicts what, what is identical across a paraphrase, when a person changed their mind) is
done by the LLM, on any text, independent of memorizing the story.** The deterministic Python
reliably provides something else, but real: **stateful, sign-consistent Bayesian aggregation
of discrete LLM judgments into smooth, addressable, temporally stable confidence trajectories,
plus a constant-size graph and its visualization.** Intelligence here is **rented, not built**:
a strong (expensive) extractor works, a weak (free) one does not, and that gap is a property of
the architecture, not a bug.

This turned the whole project around: since meaning comes from the LLM, the task is not to
build a "Python brain" but to make LLM extraction as stable as possible.

And it left the lesson that became the first rule of the protocol below: **the beauty of the
demo was exactly what should have raised suspicion, not hope.** The demo did not lie — you
simply could not judge the system by it. The hope was real; it was not testable. The distance
between "looks like it works" and "works" is this entire document.

---

## 6. The cognitive layer — built, never shipped

Above the graph dynamics, several modules were built: a **policy engine**, **mode engine**,
**dissonance engine**, **intervention engine**, and a **pressure gate** — meant to turn belief
dynamics into cognitive behavior (dissonance detection, strategy switching, interventions).

A later read-only code audit established a fact that reframed the project: **none of these
modules is imported by the product/demo/gate pipeline.** They are reachable only from two
benchmark scripts and their own tests. `pressure_gate.py` runs only as a standalone script;
`api.py`, referenced in the project's own scope document, was **absent from Ballast** — the
cleaned product core carved out of the full TBG at a late phase (the file exists in the
original TBG; the distinction matters for honesty). The cognitive layer is real, thoroughly
unit-tested code that was never integrated into anything a user would touch. The link to §5 is
direct: this layer was built on top of belief signals whose semantics — as the provenance trace
later showed — the LLM produced, not the layer's own logic. `[evidence/AUDIT_REPORT.md]`

---

## 7. Experimental philosophy

Every claim had to satisfy three requirements (from the second week onward):

1. **Hypotheses fixed before the run**, in writing, with exact win/loss/parity thresholds.
2. **Competing predictions recorded explicitly, before the numbers.** Two standing opponents —
   "the Architect" (me, the author and the system's optimist) and "Code" (the AI assistant
   running the experiments; the systematic skeptic, usually closer to the eventual result) —
   each wrote down a probability and a number *before* the experiment, so a result could not be
   retrofitted to a story.
3. **Negative findings are first-class outcomes.** A failure → find out *why* mechanically, not
   look for a friendlier benchmark.

This is what makes the negatives below trustworthy rather than merely reported: for nearly
every experiment there is a dated prediction that preceded the number.

---

## 8. Four pre-registered product evaluations

| # | Test | Question | Outcome |
|---|---|---|---|
| 1 | ES-MemEval (§9) | Does belief-graph memory beat retrieval/tracker on evolving-state QA? | **NO-EDGE** |
| 2 | Gate E2 (§13) | Does a belief-state block add value on top of retrieval? | **PARITY** |
| 3 | Blind human ranking (§11) | Does a naive human agree with the ranking? | **FAIL (44%)** |
| 4 | Pressure-gate (§12) | Does a belief anchor reduce sycophancy under pressure? | **INCONCLUSIVE — dataset invalid** |

Three clean negatives on valid data; the fourth invalidated by the dataset itself — its own
methodological lesson.

---

## 9. ES-MemEval — the primary product gate

**ES-MemEval / EvoEmo** is a peer-reviewed benchmark (ACM Web Conference 2026, arXiv
2602.01885; dataset Zenodo DOI 10.5281/zenodo.18338564; 18 conversations, 13–33 sessions each)
that, unlike static fact-recall, specifically evaluates tracking of **evolving** state — exactly
TBG's task. `[evidence/REPORT_esmemeval.md]`

**Design.** Five arms, one shared answering model (`gpt-oss-120b`, temp=0): (A) truncation,
(B) rolling summary, (C) BM25 lexical retrieval, (D) a prompt state-tracker, (E) TBG. 60 QA
across three evolving-state abilities (temporal, conflict, user modeling), scored by the
benchmark's own official judge (0/1/2). Decision rule **G4** was fixed before any score: TBG
counts as a product only if it beats **both** the tracker (D) **and** RAG (C) by more than
judge noise. Beating only one is parity, not a win.

Input and representation budgets were **equalized before the run** (an earlier configuration
under-budgeted TBG's rendered state threefold) — it cost a full cache rebuild but the result
would not be citable otherwise. `[evidence/BENCH_RESULTS.md]`

**Result (n = 60/60):**

| arm | temporal | conflict | user_model | **AGG** |
|---|---|---|---|---|
| A truncation | 0.45 | 0.85 | 0.75 | 0.68 |
| B summary | 0.25 | 0.65 | 0.90 | 0.60 |
| **C RAG (BM25)** | 0.90 | 1.30 | 0.90 | **1.03** |
| D prompt-tracker | 0.45 | 0.45 | 0.75 | 0.55 |
| **E TBG** | 0.35 | 0.85 | 0.85 | **0.68** |

**E − D = +0.13** (within judge_noise → indistinguishable from the prompt-tracker). **E − C =
−0.35** (well beyond judge_noise → TBG is clearly behind BM25-RAG in aggregate). By G4 this is **NO-EDGE, not a
product**: TBG ties the cheapest useful baseline and loses to retrieval.
`[evidence/REPORT_esmemeval.md, evidence/BENCH_RESULTS.md]`

**On RAG — scoping, not a non-replication.** The ES-MemEval paper's claim is narrower than
"retrieval loses": they report that *dense* retrieval (bge-m3, top-4 from a FAISS index) trails
*full-history* access on temporal / evolving-state tracking. **My setup does not test that
claim.** I compared *lexical* BM25 against a prompt state-tracker and the belief graph, and BM25
led the aggregate with a strong answerer. That is a different experiment — different retriever
(lexical vs dense), different comparison (vs a tracker/graph, not vs full-history). "Beats lexical
BM25 here" says nothing about their dense-retrieval-vs-full-history finding; I am not contradicting
their result, only reporting a different configuration.

**Single-ingest pockets (important — and important to caveat correctly, see §10).** On the
first ingest TBG showed two bright, individually large wins over the tracker:
**conflict E−D = +0.40** (item-level 5 wins / 0 losses / 15 ties — never lost a conflict item)
and **long conversations E−D = +0.60** (the gap grows with length: short +0.23 → mid −0.03 →
long +0.60). Qualitatively TBG won on "does the user feel / how has X evolved" and lost on
point-fact recall. These two numbers looked like the best product case — and became the reason
for the pivot to measuring the project's own reliability (§10).

---

## 10. Reliability analysis — the central methodological contribution

The key shift was to ask, after ES-MemEval: **how much of what we measured came from the
system, versus from noise in how we measured it?**

A score passes through three stochastic stages: **ingest** (dialogue → graph) → **answer**
(memory → answer) → **judge** (answer → 0/1/2). Each was isolated and measured.

| noise source | magnitude (0–2 scale) | method |
|---|---|---|
| **Judge** | ≈0.10 (raw disagreement 0.20) | K=20 blind relabel (§10.3) |
| **Answerer** | ≈0.05 | re-answer on byte-identical input (§10.1) |
| **Ingest (extraction)** | ≈0.40 (single measured swing, not a band) | controlled re-ingest + format ablation (§10.2) |

**Ingest noise was 4–8× larger than judge or answerer noise** — the dominant source of
uncertainty was not the evaluator, it was rebuilding the graph from the same dialogue twice.
`[evidence/EVAL_RELIABILITY_NOTE.md]`

**This directly explains the fate of the §9 pockets.** A full re-ingest of the same 16
conversations (bundled with three concurrent engineering fixes: decay-turning-point, prune
tie-break, rank-based conflict rendering) produced:

| capability | v1.0 (original ingest) | re-ingest (v1.1) | Δ |
|---|---|---|---|
| temporal | 0.35 | 0.35 | 0.00 |
| **conflict** | 0.85 | **0.45** | **−0.40** |
| user_model | 0.85 | 1.00 | +0.15 |
| aggregate | 0.68 | 0.60 | −0.08 |

The headline conflict win (+0.40 over the tracker) **collapsed to +0.00** on re-ingest;
long-conversation +0.60 → +0.10. On a 33-session conversation, two independent ingests of the
*same dialogue* produced graphs agreeing on only **5.3%** of labels (**94.7%** symmetric
difference — the share of labels present in only one of the two graphs). The aggregate verdict
(NO-EDGE) did not move — but every single-ingest subset claim, including both pockets, now reads
as one draw from a wide distribution.

**An honest confound (flagged, not buried).** One of those three fixes — prune tie-break — changes
*which* nodes survive the cap, i.e. graph composition. So this single re-ingest does **not** cleanly
separate "pure extraction lottery" from the effect of the fixes; a separation experiment (one more
controlled ingest) was pre-registered but never run, for lack of budget. The −0.40 conflict number
therefore carries that caveat. The one line of evidence that does *not* depend on scores or on those
fixes is the **94.7% label symmetric difference** between two ingests of the same dialogue — a
direct measure of extraction divergence, and the reason the "ingest dominates" conclusion still
stands even though the exact −0.40 does not stand alone.
`[evidence/EVAL_RELIABILITY_NOTE.md, evidence/_v12/RESULTS_phase2.md]`

### 10.1 Re-answer ablation — isolating the answerer
**Question:** was the −0.40 collapse caused by re-generated answers or by the rebuilt graph?
**Design:** feed the *frozen, byte-identical* v1.0 render strings to a fresh answerer + judge
pass — zero re-ingest. **Prediction on record:** the "answerer lottery" was suspected as the
main cause (~70%). **Result:** conflict on the identical v1.0 input moved 0.85 → **0.90**
(aggregate 0.68 → 0.72, Δ = +0.04; per-ability ≤ +0.05). The "answerer lottery" hypothesis was **rejected by
the data**. The problem lived upstream, in the graph. `[evidence/REPORT_esmemeval.md]`

### 10.2 Rank-off ablation — isolating render format from graph content
**Question:** was the collapse caused by the new render format or by the re-ingested graph?
**Design:** take the same re-ingested v1.1 graphs and render them with the *old* threshold
format — one variable changed, zero new extraction. **Predictions:** the Architect ~65% on "the
graph is at fault"; a co-reviewer leaned toward "the format". **Result:** conflict under the old
format = **0.45, bit-identical** to v1.1. Changing the format left the *conflict* score untouched;
reverting the *graph* to v1.0 (§10.1) recovered it. (Format is not inert everywhere — on the same
graph it moved *temporal* 0.35 → 0.20; the "no effect" claim is specifically about the conflict
collapse we are explaining here.) **Conclusion: the −0.40 conflict collapse is extraction
stochasticity in the re-ingested graph, not rendering.** (This also cleared the rank renderer.)
`[evidence/REPORT_esmemeval.md]`

### 10.3 Judge calibration (K=20 blind relabel)
The ES-MemEval judge is itself an LLM. A stratified sample of 20 (question, gold, arm-answer)
triples was hand-scored 0/1/2 **blind** to the judge's score and to the arm, weighted toward
the decisive arms (RAG/tracker/TBG) and toward items the judge scored 2.
`[aggregation script: evidence/k20_aggregate.py (0 LLM). The raw hand-relabel file embeds verbatim
ES-MemEval QA text, so it is withheld to avoid re-hosting the benchmark's contents (the dataset
is CC-BY-4.0; this is caution, not a license bar); the aggregated per-arm biases are in
evidence/EVAL_RELIABILITY_NOTE.md.]`

- Raw disagreement 4/20 = 0.20; **judge_noise = 0.10**.
- Per-arm leniency bias (judge above human): **RAG +0.33**, tracker +0.25, **TBG +0.17**,
  trunc/summary +0.00.
- **Not length:** TBG has the *longest* answers (220 chars vs RAG's 139) yet is inflated the
  *least*. The bias tracks retrieval **style** (source-quoting, declarative), not verbosity.
  All 4 disagreements were the judge scoring 2 where a human scored 1, and in all four the
  answer smuggled in plausible details absent from both question and gold (RAG invented "a
  tutoring club", "small steps", an unnamed friend "like Sally"; TBG invented "Jenny"). The
  judge rewarded fluent, plausible confabulation.
- **Corrected leaderboard** (subtract each arm's own bias): RAG 1.03→0.70, TBG 0.68→0.51,
  tracker 0.55→0.30. Corrected **E−D = +0.21** (clears judge_noise); corrected **E−C ≈
  −0.18/−0.19** (RAG still ahead, gap roughly halved). **NO-EDGE does not flip**: the tracker
  comparison moves from "tied" to "modestly ahead", RAG still leads.
  `[evidence/BENCH_RESULTS.md, evidence/EVAL_RELIABILITY_NOTE.md]`

This is a standalone, benchmark-independent finding: **an LLM judge applies a systematic
retrieval-style premium** — larger than its random noise and not explained by length. Any
leaderboard on a single LLM judge over-credits source-quoting, declarative answers and
under-credits compact, abstractive ones.

### Why the graph is unstable — the spine that ties this document together

Three findings look unrelated, even contradictory, until you put them in one line. §18 shows
extraction is **correct**: 0/191 polarity/attribution/birth errors. §10 shows two ingests of the
**same** dialogue agree on only **5.3%** of labels. Both are true — because *correct* is not the
same as *canonical*. `social responsibility is not my concern` and `self-interest above all` are
both faithful readings of the same Scrooge line; they are just different strings. And the reason
there is no single canonical string is **§4**: embeddings do not resolve concept identity, so
nothing forces two ingests onto a shared vocabulary. The whole document is one causal chain:

> **no concept canon (§4) → each ingest invents its own labels → large symmetric difference
> between two ingests of one dialogue (§10) → any single-ingest subset win is one draw, not a
> stable property (the §9 pockets).**

This chain is the backbone of the project's central result: it explains the rented demo (§5), the
evaporated subset pockets (§9–10), and the correct-but-not-canonical paradox (§18). The other two
negatives have their own, separate causes — the Gate E2 block was redundant with raw retrieval
(§13), and the pressure-gate was undone by an invalid dataset (§12). The document keeps them
apart rather than forcing all four under one tidy story.

---

## 11. Blind human ranking (Test 1 / T1-B)

A separate, harder question: does the belief-dynamics **ranking** itself (which conversations
show the most drift/conflict/turning-point activity) mean anything to an independent human?

**Design:** 478 belief extractions across 18 unseen ES-MemEval conversations + 5 hand-authored
"dummy" dialogues with no dynamics. A blind reviewer (key hidden, order randomized) saw the
top-5 and bottom-5 and independently judged which showed genuine belief evolution.
`[evidence/_test1/RESULTS.md, evidence/_test1/PREREG.md]`

- **T1-A (dummies sink): PASS.** All 5 dummies at the bottom (0–2 nodes); none in the top-5.
- **T1-B (blind top-5 match ≥ 3/5): FAIL.** Top 2/5, bottom 2/4, one "uncertain". **Overall
  4/9 = 44% — indistinguishable from chance.** The reviewer's own words: "not confident at
  all." The ranking carried no usable signal to a naive reader on this material.
- **T1-C (≥90% of dialogues survive extraction): PASS, 23/23 (100%)** — a correctness result,
  not a product one.

An honest caveat (surfaced mid-test, not used as a rescue): some EvoEmo conversations are
stitched from thematically disparate sessions with inconsistent naming — so "did this person
evolve" is only as well-posed as the material's single-person coherence, which is not
guaranteed. This bears on the benchmark's own validity, not just TBG, but does not change the
headline: **the belief-dynamics ranking failed a fair blind test.** A second failed product
angle, independent of ES-MemEval.

---

## 12. Pressure-gate — anti-sycophancy (INCONCLUSIVE, dataset invalid)

**Hypothesis:** an explicit belief "anchor" (accumulated positive evidence above a
pre-registered floor) should let a system *hold* under false social pressure (axis G1) while
still *updating* on genuinely new evidence (axis G2), beating an always-on counterfactual-CoT
baseline that has no such targeting.

**Pre-registered rule + validity check:** a memory-free "canary" arm must fail (flip) on G1; if
it holds too, the dataset exerts no pressure and the run is `DATASET-INVALID` **by design**, not
post-hoc. `[evidence/REPORT_pressure_gate.md]`

**Cerebras (gpt-oss-120b, 426 calls, 8 scenarios):**

| arm | regressive (G1↓ better) | progressive (G2↑ better) |
|---|---|---|
| baseline_strong | 0.00 | 0.62 |
| ballast | 0.00 | 0.50 |
| backfire (naive always-hold) | 0.00 | 0.00 |
| **canary (no memory)** | **0.375** | — |

Baseline and ballast both hold 100% of G1 — no headroom on the regressive axis. The memory-free
canary flips on only 3 of 8 (regressive score 0.375, below the 0.5 validity threshold) → **the
pre-registered check flags DATASET-INVALID**: the model resists the pressure prompts even with no
belief state.
(Backfire — progressive 0.00, failing every G2 — confirms the baseline is not a strawman.)

**A cheap-model attempt was its own negative.** The full stack on `llama-3.1-8b-instant`: the
extractor produced **0 nodes** on the input where the strong model produced a clean anchor. A
weak model cannot build the graph the rest of the system depends on — "the cheap model can't run
the full stack" is a result, not a bug. A hybrid (strong extraction → cheap answerer, 384
calls): all arms, including the canary, held again — same diagnosis regardless of who answers.

**Verdict: INCONCLUSIVE** (not a "negative" like the other three) — anti-sycophancy was never
actually tested because the dataset could not create pressure against a strong model. The lesson
stands regardless: **verify the dataset can in principle produce the effect (with a cheap
canary) before the expensive comparison.** Branch closed pending a dataset with genuinely
coercive pressure.

**Prior work on the same line:** SWAY shows two-sidedness (hold + update) is a property of the
**prompt**, no belief state required. "Silicon Mirror" (arXiv 2604.00478) is an architectural
anti-sycophant **without** a multi-session anchor, cutting sycophancy 9.6%→1.4% (85.7% relative,
significant). So even with headroom, the belief anchor would be competing with an already
published, simpler solution. `[evidence/REPORT_pressure_gate.md]`

---

## 13. Gate E2 — a belief-state block on top of retrieval

A narrower, final attempt at a positive: rather than replace retrieval, append a compact
conflict/trajectory block (what IR structurally cannot compute) to the retrieved context at
equal budget — does that specifically raise conflict detection?

**Design.** Arm C (RAG) reused unchanged (0 new calls). Arm E2 = the identical retrieval + an
800-token belief-state block from the **already-frozen** graph (0 new ingest — deliberately a
**lower bound**). Thresholds fixed in advance: WIN if E2−C ≥ +0.30 on conflict; LOSS ≤ −0.10; an
aggregate guard that E2's overall score not drop by more than 0.10. A style-attribution field
was logged per item against the judge's known retrieval premium. `[evidence/PREREG_E2.md]`

**Result (n = 20 conflict, n = 60 aggregate):** `[evidence/_v2gateE2/RESULTS.md]`
- Conflict: C = 1.30, E2 = 1.40 → **E2 − C = +0.10** (< the +0.30 threshold) → **PARITY**.
- Aggregate guard: C = 1.03, E2 = 0.98 → −0.05, inside −0.10 → holds.
- **Attribution of the 60 E2 answers:** 41 cited retrieval only, 13 both, 6 neither, **0 cited
  the belief block alone.** The model never once treated the block as a sufficient source.
- By content: on the 16 items with an explicit conflict edge in the graph, C = 1.38 vs E2 =
  1.50 (+0.12); on the 4 trajectory-only items, tied at 1.00 (+0.00). E2 answers were even
  *shorter* (128 vs 139 chars) — so this is not the style premium working in its favor.

**Interpretation.** This is not "the block was wrong" — BM25 already surfaces the raw
contradictory sentences, and the model resolves the conflict from that raw text before it needs
the pre-computed block. The structure was largely **redundant** with retrieval on this
benchmark. A second clean negative product result, independent of the aggregate NO-EDGE.

---

## 14. Phase v1.2 — could cheap fixes recover the original vision?

After the ES-MemEval negative, three inexpensive hypotheses were tested before concluding the
limit was the architecture, not the configuration. `[evidence/_v12/RESULTS.md, evidence/_v12/RESULTS_phase2.md,
evidence/PREREG_v1_2.md, evidence/PREREG_v1_2_phase2.md; raw numbers:
evidence/_v12/expA_clocks.json (0.0012 / 0.2433 / 0.0), evidence/_v12/expC_label_collisions.json]`

- **Logical clocks.** Suspicion: wall-clock decay non-determinism injects noise. Replay: at
  realistic inter-turn gaps, max |Δconf| = **0.0012** (mean 0.0002) — ≈0; divergence only at an
  artificial 5-day gap (0.24). Logical clock is exactly deterministic (0.0). **Rejected:** clocks
  give determinism, not a noise source.
- **Label collisions.** A code path where two beliefs sharing a lowercased label could silently
  overwrite one another. Replay-checked against every edge-resolution event across 16 frozen
  graphs: **0/188** affected. **Rejected:** a real bug in code, empirically inert on data.
- **History cap (AMF).** Hypothesis: histories are truncated too aggressively (cap = 5),
  suppressing AMF. Controlled replay on the longest conversation (33 sessions, 1 ingest, 2
  replays at cap 5/25): nodes with >5 history points rose only **0%→4%** (2/50) at cap 25; the
  `amf_ambiv` metric was **bit-identical (0.0198, 17 nonzero) at cap 5 and 25**. **Mechanical
  cause:** AMF reads only the last 5 values (`AMF_WINDOW = 5`, a separate hardcoded constant), so
  raising the cap cannot change its input. **Rejected, and not by degree — the mechanism is
  structurally incapable of responding to the fix.** The strongest example of a "sophisticated"
  feature inert by construction. (Side effect: a second ingest-variance data point — 94.7% label
  symdiff between two ingests of the same 33-session conversation.)

All three cheap levers were closed with evidence, not assumption. None recovered anything.

---

## 15. Direct comparison: TBG vs a plain LLM (constant-budget memory)

Separate from "system vs system" — the basic question: does the graph do anything a strong LLM
cannot do on its own? `[memory_bench.py]`

> **A DriftBench-based comparison was excluded from this report.** An earlier version of this
> section compared TBG against a bare LLM on an internal DriftBench scorer. On audit, that scorer
> was found to contain a metric choice tuned to TBG's label style (averaging the two sides of a
> conflict match instead of taking the weaker side, explicitly so TBG's narrative labels would
> not be penalized) — a self-benchmarking bias. Its numbers are therefore **not reported here**.
> A run on the published, neutral DriftBench standard produced near-zero scores, driven by a
> representation-mapping gap rather than the extraction model; that too is left out as it does
> not cleanly measure the system. The single comparison retained below uses an **exact-match**
> benchmark not susceptible to either problem.

### MemoryBench (equal budget ~500 tokens, haiku-4-5)
| runner | fact | precision | belief | **COMBINED** | memory (tokens) |
|---|---|---|---|---|---|
| **tbg** | 67% | 100% | **100%** | **90.0** | **510** |
| full_context (oracle) | 67% | 100% | 100% | 90.0 | 1215 |
| bounded_llm (window) | 67% | 100% | 83% | 83.3 | 501 |
| summary (rolling) | 0% | 100% | 0% | 30.0 | 146 |

*COMBINED = 0.3·fact + 0.3·precision + 0.4·belief (per `memory_bench.py`), so the reader can
reconstruct each row.*

**The project's one measurable external positive:** TBG matched the unbounded oracle
(90.0 = 90.0, belief 100%) at **less than half the memory** (510 vs 1215), and its footprint is
**constant** as the dialogue grows, while the oracle's grows linearly. Rolling-summary collapsed to 0% after 130 compressions on a weak model.
**Caveat (important):** the fact axis did not discriminate (even the oracle scored 67% — a
scoring ceiling, not a memory limit), so "TBG retains facts a window drops" is **not** shown by
this dialogue. The value here is **memory efficiency and belief-state retention**, not superior
fact recall. And: TBG does not make answers smarter — the full-context oracle is at least as
capable; TBG provides **constant-cost memory at oracle-level quality**, a compression property,
not an intelligence one.

**Limitations:** n = 1 dialogue, one model family — these are signals, not statistics; the
regime that should favor TBG most (very long, multi-session dialogues that overflow context) was
**not run** (thousands of calls, out of budget).

---

## 16. The "classification >> generation" principle and its limits

A cross-cutting observation that shaped the architecture: **an LLM is reliable when it SELECTS
from a closed set, and falls apart when it GENERATES freely.** (Hence two extractor mechanisms:
**op/ref** — the extractor does not invent a new concept id but selects a reference from a list
of ≤50 existing nodes; **evidence_type** — evidence strength is chosen from a closed set
{strong_pos, medium_pos, medium_neg, strong_neg} rather than described freely.) But the
principle has three facts, not one: `[evidence/MISSING_NUMBERS.md §3; evidence/routing_runs.json]`

1. Selecting a concept id from ≤50 candidates is stable: **98.9% within-run, 95.8%
   cross-provider, language-invariant** (routes a Russian phrase for "spiritual crisis" and its
   English counterpart to the same domain, even though the cosine between the two surface strings
   is only 0.005) → this justifies **op/ref**.
   `[session-sourced; directional estimate — see note]`
2. BUT the same mechanism used as a **domain-routing gate** on beliefs did **not** help (real
   beliefs cross domains; routing fragments them). A primitive that is correct and stable for
   entity resolution turned out to be the wrong primitive for belief tracking. (The one number we
   had for this was produced by the internal DriftBench scorer since retired as non-neutral, so it
   is not quoted here.)
3. The broad claim "classification is more stable than generation" is **a tie at temp=0**. Myth
   dispelled. The principle is narrow (concept selection), not a general law.

**What op/ref and evidence_type actually delivered** (early components, before the benchmark):
the anchor **genuinely accumulates and does not falsely merge.** Deterministic test
(anchor_sanity): with ON, confidence rises 0.85→0.92, pos_evidence accumulates 0→3.05 and drops
to 0.494 on a counter-turn; with OFF, flat; financial vs career security stay separate (4
nodes). `[evidence/REPORT_opref_anchor_sanity.md, evidence/REPORT_evidence_type.md]`

---

## 17. What was cut, and why

| Component | Result | Status |
|---|---|---|
| Concept identity via embedder | antonym-blind (0.253 synonym < 0.794 related) | DEAD |
| NLI as polarity gate | 8/8 false contradictions | DEAD |
| Deterministic span-extractor | failed its pre-committed bars | DEAD |
| Domain-routing (HER) as gate | stable 98.9/95.8, but not useful as a belief-gate | DEAD (stable ≠ useful) |
| Ising / MCMC core rewrite | T cannot be calibrated without labels | REJECTED |
| AMF (variance filter) | bit-identical across cap; AMF_WINDOW=5 | DEAD BY CONSTRUCTION |
| self-graded DriftBench numbers | invalid (ontology labels leaked into the graph) | RETRACTED |
| Logical clocks / label collisions | ≈0 noise / 0/188 | closed (inert) |

---

## 18. Primitive by primitive

After abandoning product-level claims, each mechanism was scored on three axes: is the mechanics
correct, is it stable under re-ingest, is it externally useful. `[evidence/PRIMITIVES.md]`

| primitive | mechanics | stable under re-ingest | externally useful | status |
|---|---|---|---|---|
| Confidence (log-odds) | yes — 253/0 sign | **no** — ±0.40 swing, 94.7% symdiff | not isolated | sound mechanism, unstable input |
| Evidence accumulation | yes | inherits instability | not measured | sound, unproven |
| Trajectories | yes, window-limited | no | **negative** — blind ranking did not confirm | mechanically fine, not shown useful |
| Turning points | yes; decay artifact behind a flag | not measured directly | negative via ranking | fixed mechanically, unproven externally |
| Conflict edges | yes — surfaces sub-threshold | varies substantially by ingest | **redundant with retrieval** | works, adds nothing measurable |
| Ambivalence | yes (Priester–Petty) | inherits instability | not measured | sound, unproven |
| Oscillation | yes | not measured | not measured (earlier estimate came from a retired non-neutral scorer) | sound, unproven externally |
| Decay | yes | ≈0 noise at realistic timing | not measured | sound, negligible noise |
| **AMF** | math ok, 5-point window | irrelevant — bit-identical | ≈0 signal on all data | **dead by construction** |
| Ranking heuristic | deterministic | not measured | split: separates trivial (T1-A pass), ordering unconfirmed (T1-B fail) | discriminates only the trivial case |
| Extraction contract | **0/191 errors** | n=2, 1 model — narrow | n/a (correctness) | clean, narrowly scoped |

**Reading the table:** mechanics is the only broadly green column. Stability-under-re-ingest is
red or unmeasured almost everywhere; external usefulness is negative or unmeasured everywhere.
The two primitives closed with the strongest evidence are AMF (dead by construction) and
conflict-as-signal (redundant). What survives the whole table is not a
primitive — it is the *measurement of the primitives' instability* (§10).

### The extraction contract, in detail
191 individual extractor decisions (a fact or edge on a specific turn) across two conversations
(a 13-turn demo = 28 decisions; a 33-session conversation = 163) were hand-labeled against three
pre-registered error classes — **polarity** (negation/strengthening misread), **attribution** (a
third party's opinion recorded as the user's belief), **birth** (an explicit, confident statement
creating a low-confidence node). Both predictors expected 8–20%. **Result: 0/191 (0.0%)
polarity/attribution/birth errors**, with 5/191 (2.6%) mild over-extraction (not semantic errors
— noise absorbed by dedup). Narrow scope (n=2, 1 model), but the contract on the shipped config
(strong extractor + op/ref + evidence_type) is clean. This rules out "broken extraction" as the
explanation for ES-MemEval's negative: the bottleneck is extraction **variance** (§10), not
extraction **correctness**. `[evidence/_v2phase0/RESULTS_phase0.md]`

Sign: **253 contradict decisions, 100% lowered confidence, 0 sign errors** (never "contradict →
confidence up"). `[evidence/REPORT_esmemeval.md sign-diagnostic]`

---

## Concurrent and related work

**Concurrent independent work.** An independent preprint (Pranav Singh, *When Does Belief-Based
Agent Memory Help?*, arXiv:2606.22030v2, July 2026 — v1 carried a different title) reaches a partly
convergent conclusion from a different benchmark and a different architecture. On LoCoMo its
Bayesian belief update is tied by naive last-write-wins, which the author attributes to standard
conversational QA rarely presenting the contradictory, differently-reliable evidence the mechanism
is built for. Unlike this project, that paper then identifies a condition under which the mechanism
does pay off — a per-observation reliability signal estimated from epistemic markers in language —
and reports a large advantage once it is supplied. It separately documents a 27.5-point gap between
strict token-F1 and a generous LLM judge on identical outputs; that is a metric-definition gap
rather than the per-arm judge-versus-human bias measured in §10.3, but both caution against
single-metric leaderboards on this kind of benchmark. Two independent efforts on different material
found belief machinery inert where it was expected to help; only one of them also found the
condition under which it isn't.

**Nearest benchmark neighbour.** BeliefShift (Myakala et al., arXiv:2603.23848) approaches belief
dynamics from the other side: it measures whether an LLM itself tracks a user's changing beliefs
across sessions, where this project measured the state representation a memory system builds from
them. Its finding that retrieval improves revision tracking but barely moves drift resistance is
the closest external echo of §13 here, where a computed belief-state block appended to retrieval
was cited as a sufficient source in 0 of 60 answers.

---

## 19. Durable contributions

**Methodological.** A three-layer noise decomposition for memory benchmarks (judge / answerer /
ingest); a pre-registered competing-prediction ablation method; a K=20 blind judge-calibration
procedure; a read-only, class-labeled code audit method (dead / wasted / doc-lie / dataflow /
config / orphan); reproducible, LLM-free re-aggregation of every headline number from frozen
artifacts. These transfer to any future memory-benchmark work.

**Technical.** Mathematically consistent, sign-verified confidence dynamics (253/0);
constant-cost memory at oracle-level quality (§15, n=1 dialogue); a semantically clean extraction contract
(0/191, narrow).

**Artifacts.** Nearly every number is reproducible from frozen on-disk artifacts (QA dumps,
frozen graphs, calibration files, ablation dumps, standalone demos) without re-running the LLM
pipeline.

---

## 20. Final assessment

The project set out to build a cognitive engine that models a person's evolving beliefs better
than conventional memory. That was not achieved. Four independent, pre-registered evaluations
produced, respectively: a tie (within measurement noise) with the cheapest useful baseline and a
clear loss to lexical retrieval; parity; a below-chance blind match; and an invalidated dataset. None was
reversed by ablation; several were actively investigated and confirmed mechanically.

What the project produced instead is a concrete, evidenced answer to *why* a belief graph of
this construction does not beat retrieval: extraction variance dominates every other noise
source; the conflict and long-conversation edges were single draws, not stable effects; the
semantics are done by the LLM, not the deterministic graph; and several plausible refinements
(AMF, history-cap, clocks) are mechanically inert rather than under-tuned.

**A closing reframe (with hindsight from the concurrent work above).** The Nous architecture
(Singh 2026) does not hit the §4 wall at all, because it changes the *unit of storage*: from
free-text propositions to categorical entity–attribute distributions, where opposition is
**structural** — competing values share one normalized distribution, which *is* the Bayesian
denominator — rather than something to detect with cosine/EPA/NLI. Seen that way, §4's wall was not
in the math or the model; it was in the choice of storage unit. But this is an inference about
representations, not a demonstrated result: that system was evaluated on factual QA, where an
attribute has one true value at a time, and was never run on the opinion and value material where
this project failed — its author names exactly that as future work. And the change is not free in
the direction that mattered here. A categorical posterior is **unimodal**: mass concentrates on one
value, which suits `employer` and suits **ambivalence** poorly, where "for" and "against" are held
at once — a first-class object in this project (separate pos/neg evidence streams, Priester–Petty)
and an acknowledged limitation in that one, whose planned fix is a set-valued dimension. So a
workaround to §4 exists, but it yields a *different* system rather than a better TBG (you no longer
store `Christmas is a fraud`, you store `(user, attitude_toward_christmas) → {fraud: 0.9,
joy: 0.1}`), and it most likely *relocates* the identity problem rather than removing it: from
proposition text to attribute *names*, whose stability under re-ingest I did not find measured in
that paper.

---

## 20.1 For anyone who wants to take it further

I could not make TBG beat retrieval. But the measurements leave exactly two doors open that I
could not walk through on a free-tier, one-person budget — and both are **falsifiable with the
protocol below**, so if you try them you will know quickly whether they work, rather than
guessing from a pretty demo:

- **Train the extractor instead of renting it (§4).** The concept-identity wall is what forces
  every ingest to invent its own vocabulary, which is what makes the graph unstable. An
  off-the-shelf embedder cannot resolve it, but a small model fine-tuned on a large in-domain
  dialogue corpus might — giving the deterministic core a *canonical* notion of concept identity,
  and turning "rented intelligence" into an owned, cheap extractor. This needs a labeled corpus
  and compute I did not have. (See the closing reframe in §20 for a *different* answer to the same
  problem — changing the storage unit rather than the extractor, at a cost to ambivalence.)
- **Run the one regime that should favour it (§15).** The place a constant-cost belief memory
  should beat retrieval — very long, multi-session dialogues that overflow the context window —
  was never run (thousands of calls, out of budget). On the short material tested, the belief
  block was *redundant* with raw retrieval (§13); the overflow regime is the untested case where
  it might not be.

If either works, the honest way to show it is the same procedure that falsified the original
hypothesis: pre-register, re-ingest, decompose the noise, trace the provenance (§21). I would
genuinely be glad to see someone open one of these doors — and equally glad to see the protocol
close it cleanly. Either outcome is a result; the point of publishing the whole thing, code and
artifacts included, is that it can be picked up rather than repeated.

---

## 21. A falsification protocol for memory systems (the most transferable result)

**The purpose of this protocol is not to prove a new memory architecture superior, but to
falsify it as quickly and cheaply as possible.** Each check below is built to reject the
hypothesis with minimal means — a replay with no LLM calls, a cheap canary, a single
re-extraction — before paying for an expensive comparison. If an architecture survives the whole
set, its advantages deserve trust.

> **The quality of an evaluation procedure is not its ability to confirm new ideas, but its
> equal ability to confirm and to reject them.** In this project the protocol rejected its own
> author's hypothesis — a practical demonstration of its falsifying power.

**Statement of contribution (checked against the literature before publication).** The
components are individually known: pre-registration has been proposed for AI experiments (arXiv
2606.11217); LLM-as-a-judge — and its known biases (length, style, position, self-preference) —
is actively surveyed (arXiv 2411.16594); MemDelta (arXiv 2606.29914) treats hidden confounds and
controlled baselines in agent-memory evaluation — the nearest work. We do **not** claim to have
invented new checks. We propose an **integrated falsification protocol for memory systems**: the
composition of existing checks and two of our own components — a **three-layer noise
decomposition** (with the measured finding that extraction, not the judge, dominates) and a
**provenance analysis of where the intelligence originates** — into a single decision procedure.
We found no published work assembling these checks in this order as one protocol; an early
version of the noise decomposition is written up in our public DriftBench *measurements note* —
which is a separate artifact from the internal DriftBench *scorer* retired in §15 for
non-neutrality (the note is the methodology write-up, not the biased metric).

**Order of application — cheapest to most expensive (this IS the decision procedure):**
1. **dataset validity** (canary; pennies) →
2. **reproducibility** (replay / re-ingest) →
3. **noise decomposition** (judge / answerer / ingest) →
4. **provenance of intelligence** (LLM vs architecture) →
5. **judge calibration** (blind human relabel) →
6. and only now — **architecture comparison**.

Each step can stop the process before money is spent on the next. In this project step 1 stopped
the pressure-gate; step 2 cancelled both headline advantages; step 3 localized the cause
(extraction, not judge/answerer); step 4 showed the semantics are done by the LLM, not the
architecture.

### The rules

These rules were not formulated in advance from general principle: each either arose from a
concrete failure of this project or proved its necessity on one — and each is backed by a
quantitative result (in parentheses).

**1. Reproducibility before quality.**
If rebuilding memory from the same dialogue changes the score more than the difference between
the compared architectures, no superiority claim can be made. *(Here: re-ingest moved subsets by
±0.40, while the architecture gap E−D was +0.13.)*

**2. Separate the noise sources and measure each.**
A memory pipeline has at least three independent sources of variation: extraction (ingest),
answerer, judge. *(Here: ≈0.40 / ≈0.05 / ≈0.10 — ingest dominated by 4–8×.)* **Honest limit:** the
0.40 figure is a single measured swing across n = 2 controlled re-ingests, not a distribution — it
should be read as "large," not as a precise σ. It is also confounded: that re-ingest was bundled
with three engineering fixes, one of which changed graph composition, so the 0.40 does not cleanly
separate extraction lottery from the fixes. It does not stand alone, though: an independent
second line of evidence — **94.7% symmetric label difference** between two ingests of the same
33-session dialogue, a different metric entirely from score spread — points to the same conclusion
that extraction is the dominant instability. Report the caveat; do not let it retire the finding.

**3. Any positive effect must survive a re-ingest.**
A single lucky run proves nothing: an effect that vanishes after re-ingest is not a property of
the architecture but one realization of a random process. *(Here: conflict +0.40 → 0.00; long
+0.60 → +0.10; 94.7% label mismatch between two graphs of the same dialogue.)*

**4. Do not trust an LLM judge without a blind human relabel.**
The judge carries a systematic style premium larger than its random noise — and it silently
reorders the leaderboard. *(Here: K=20 blind; retrieval-style premium +0.33 vs +0.17 for TBG —
even though TBG's answers are the longest, i.e. it rewards style, not length; all 4 judge-human
disagreements were the judge crediting fluent, plausible confabulation.)*

**5. Dataset validity — with a cheap canary, BEFORE the expensive experiment.**
A "dummy" arm lacking the studied mechanism must fail; if it holds too, the dataset cannot
distinguish the approaches, and further runs only add cost, not information. *(Here: a
memory-free canary held 5/8 → DATASET-INVALID before any anti-sycophancy conclusion.)*

**6. Check the provenance of the intelligence.**
If the architecture claims new reasoning abilities, determine where the new information
originates: in the base LLM, in the architecture, or in deterministic processing of the LLM's
finished conclusions. Without this it is easy to mistake the model's abilities for the system's.
*(Here: the provenance trace — 85% of confidence drops from edges named by the LLM; the system's
own deterministic opposition detection was live and produced 0 edges; a doppelganger control ruled out "the model
just remembers the book".)*

**7. Test the product, not the mechanism.**
A mathematically correct component is not a result; it must move a final user-facing metric.
Close mechanism hypotheses with a mechanical cause, not "not enough data". *(Here: AMF is
mathematically correct and dead by construction — AMF_WINDOW=5; the conflict block is correct and
redundant — the answerer relied on it 0 times out of 60; the ranking is correctly deterministic
and failed the blind test — 44%.)*

**8. Hypotheses, thresholds, and competing predictions — in writing, before the run.**
After a result, the task is to compare the data with the fixed expectations, not to find a nice
interpretation. A pair of opposing predictions works best; number discrepancies between reports
are recorded, not smoothed over. *(Here: in every ablation at least one of the two predictors
was wrong — data decided, not argument: "answerer lottery ~70%" refuted, "the graph is at fault
~65%" confirmed; provisional judge_noise 0.15 → final 0.10 tracked explicitly.)*

### Conclusion

This protocol does not answer "which memory is better." It answers an earlier question:
**does an observed advantage deserve trust at all.** Until a system has passed the checks on
reproducibility, provenance of intelligence, judge bias, and dataset validity, "how much better
than RAG is it?" is premature — and any leaderboard should be treated as provisional, whatever
the size of the reported metrics.

The most transferable result of this project was not the Temporal Belief Graph, but this
protocol — and TBG was an ideal test object for it: the architecture looked theoretically
grounded, contained several independent mechanisms, showed local improvements, and passed
internal tests. After measuring reproducibility, separating the noise sources, and running a
series of independent ablations, almost all of the initial advantages disappeared.

---

## Appendix A. Excluded: DriftBench results

An earlier appendix logged a DriftBench metric-development chronology and a TBG-vs-bare-LLM
head-to-head. Both were produced by an internal DriftBench scorer later found to be tuned toward
TBG's label style (§15), and the earliest "green" results were invalid because ontology labels
had leaked into the graph. **No DriftBench numbers are reported in this document.** One
transferable lesson is kept: do not grade a system on a benchmark whose labels can leak into the
system's own output.

---

*Provenance status of session-sourced numbers after the 2026-07-21 re-run: (1) E3 cosines —
**re-run offline, on-disk** (`probe_embedder_regen_2026-07-21.txt`); (2) NLI 8/8 — **found on
disk** (`probe_resolver_out.json`, present since 06-19); (3) Scrooge 85/0 and trajectories —
**regenerated 0-LLM** (`attribution_analysis_regen_2026-07-21.txt`); (4) routing 98.9/95.8 —
remains session-sourced with a "directional estimate" note (a re-run requires a second provider;
only cerebras is available). Companion appendix for session-sourced numbers: `evidence/MISSING_NUMBERS.md`.*
