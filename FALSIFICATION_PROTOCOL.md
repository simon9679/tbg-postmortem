# A falsification protocol for memory systems

*This is the transferable result of the Temporal Belief Graph (TBG) project, extracted so it can
be read and cited on its own. It is not theory: every rule below arose from a concrete failure of
a real system, and the full case study — how a belief-graph memory layer was built, measured, and
falsified by this exact procedure — is in [`FULL_HISTORY.md`](FULL_HISTORY.md). Section references
like “§10” point into that document, where each number is backed by an on-disk artifact in
[`evidence/`](evidence/).*

*If you only take one thing from the project, take this.*

---

**The purpose of this protocol is not to prove a new memory architecture superior, but to
falsify it as quickly and cheaply as possible.** Each check below is built to reject the
hypothesis with minimal means — a replay with no LLM calls, a cheap canary, a single
re-extraction — before paying for an expensive comparison. If an architecture survives the whole
set, its advantages deserve trust.

> **The quality of an evaluation procedure is not its ability to confirm new ideas, but its
> equal ability to confirm and to reject them.** In the source project this protocol rejected its
> own author's hypothesis — a practical demonstration of its falsifying power.

**Statement of contribution (checked against the literature before publication).** The
components are individually known: pre-registration has been proposed for AI experiments (arXiv
2606.11217); LLM-as-a-judge — and its known biases (length, style, position, self-preference) —
is actively surveyed (arXiv 2411.16594); MemDelta (arXiv 2606.29914) treats hidden confounds and
controlled baselines in agent-memory evaluation — the nearest work. This is **not** a claim to
have invented new checks. The contribution is an **integrated falsification protocol for memory
systems**: the composition of existing checks and two original components — a **three-layer noise
decomposition** (with the measured finding that extraction, not the judge, dominates) and a
**provenance analysis of where the intelligence originates** — into a single decision procedure.
No published work was found assembling these checks in this order as one protocol.

## Order of application — cheapest to most expensive (this IS the decision procedure)

1. **dataset validity** (canary; pennies) →
2. **reproducibility** (replay / re-ingest) →
3. **noise decomposition** (judge / answerer / ingest) →
4. **provenance of intelligence** (LLM vs architecture) →
5. **judge calibration** (blind human relabel) →
6. and only now — **architecture comparison**.

Each step can stop the process before money is spent on the next. In the source project step 1
stopped the pressure-gate; step 2 cancelled both headline advantages; step 3 localized the cause
(extraction, not judge/answerer); step 4 showed the semantics are done by the LLM, not the
architecture.

## The rules

These rules were not formulated in advance from general principle: each either arose from a
concrete failure of the source project or proved its necessity on one — and each is backed by a
quantitative result (in parentheses; details in `FULL_HISTORY.md`).

**1. Reproducibility before quality.**
If rebuilding memory from the same dialogue changes the score more than the difference between
the compared architectures, no superiority claim can be made. *(There: re-ingest moved subsets by
±0.40, while the architecture gap E−D was +0.13.)*

**2. Separate the noise sources and measure each.**
A memory pipeline has at least three independent sources of variation: extraction (ingest),
answerer, judge. *(There: ≈0.40 / ≈0.05 / ≈0.10 — ingest dominated by 4–8×.)* **Honest limit:**
the 0.40 figure is a single measured swing across n = 2 controlled re-ingests, not a distribution
— read it as "large," not as a precise σ. It does not stand alone: an independent second line of
evidence — **94.7% symmetric label difference** between two ingests of the same 33-session
dialogue, a different metric entirely from score spread — points to the same conclusion. Report
the caveat; do not let it retire the finding.

**3. Any positive effect must survive a re-ingest.**
A single lucky run proves nothing: an effect that vanishes after re-ingest is not a property of
the architecture but one realization of a random process. *(There: conflict +0.40 → 0.00; long
+0.60 → +0.10; 94.7% label mismatch between two graphs of the same dialogue.)*

**4. Do not trust an LLM judge without a blind human relabel.**
The judge carries a systematic style premium larger than its random noise — and it silently
reorders the leaderboard. *(There: K=20 blind; retrieval-style premium +0.33 vs +0.17 for the
tested system — even though its answers are the longest, i.e. it rewards style, not length; all 4
judge–human disagreements were the judge crediting fluent, plausible confabulation.)*

**5. Dataset validity — with a cheap canary, BEFORE the expensive experiment.**
A "dummy" arm lacking the studied mechanism must fail; if it holds too, the dataset cannot
distinguish the approaches, and further runs only add cost, not information. *(There: a
memory-free canary held 5/8 → DATASET-INVALID before any anti-sycophancy conclusion.)*

**6. Check the provenance of the intelligence.**
If the architecture claims new reasoning abilities, determine where the new information
originates: in the base LLM, in the architecture, or in deterministic processing of the LLM's
finished conclusions. Without this it is easy to mistake the model's abilities for the system's.
*(There: the provenance trace — 85% of confidence drops from edges named by the LLM; the system's
own Python opposition detection produced 0 edges; a doppelganger control ruled out "the model
just remembers the book".)*

**7. Test the product, not the mechanism.**
A mathematically correct component is not a result; it must move a final user-facing metric.
Close mechanism hypotheses with a mechanical cause, not "not enough data". *(There: a variance
filter mathematically correct and dead by construction — its window constant was 5; a
conflict block correct and redundant — the answerer relied on it 0 times out of 60; a ranking
correctly deterministic and failed the blind test — 44%.)*

**8. Hypotheses, thresholds, and competing predictions — in writing, before the run.**
After a result, the task is to compare the data with the fixed expectations, not to find a nice
interpretation. A pair of opposing predictions works best; number discrepancies between reports
are recorded, not smoothed over. *(There: in every ablation at least one of the two predictors
was wrong — data decided, not argument.)*

## Conclusion

This protocol does not answer "which memory is better." It answers an earlier question:
**does an observed advantage deserve trust at all.** Until a system has passed the checks on
reproducibility, provenance of intelligence, judge bias, and dataset validity, "how much better
than RAG is it?" is premature — and any leaderboard should be treated as provisional, whatever
the size of the reported metrics.

In the source project, TBG was an ideal test object: the architecture looked theoretically
grounded, contained several independent mechanisms, showed local improvements, and passed
internal tests. After measuring reproducibility, separating the noise sources, and running a
series of independent ablations, almost all of the initial advantages disappeared. The full
story — including the demo that fooled its own author, and the retractions — is in
[`FULL_HISTORY.md`](FULL_HISTORY.md).
