# v2 Phase-0 — Extractor error-baseline (P0-B) + gold conflicts (P0-A)

*0 code edits, 0 LLM. Labeler = Code. Scheme + pre-registration written before labeling
(mtime-checked). Full labels: `labels.jsonl` (191 decisions); gold: `gold_conflicts.md`.*

## Material actually labeled
2 captures, both **cerebras gpt-oss + op/ref + evidence_type** (the current contract):
demo (13 turns, 28 decisions) + p8 (33 turns, 163 decisions) = **191 decisions**.
**TBG3 and KLOD (sonnet) are absent from disk** → cross-model comparison (Part C.1) is
**not possible this phase**; deferred until a sonnet capture exists.

## Headline
> **0% of extractor decisions are semantically wrong on POLARITY / ATTRIBUTION / BIRTH
> (0 / 191), with 2.6% (5 / 191) flagged as doubtful over-extraction (OTHER, `?`), on the
> current contract (gpt-oss + op/ref + evidence_type) with clean engine sign purity (253/0,
> prior diagnostic).**

## Error rate breakdown
| cut | decisions | P/A/B errors | doubt `?` |
|---|---|---|---|
| total | 191 | **0 (0.0%)** | 5 (2.6%) |
| demo | 28 | 0 | 1 |
| p8 | 163 | 0 | 4 |
| by class (errors) | — | POLARITY 0, ATTRIBUTION 0, BIRTH 0 | OTHER 5 |
| by model | cerebras gpt-oss only | — | cross-model deferred (no sonnet) |

- **POLARITY:** no negate-then-reinforce cases. Contradictions were captured via `contradicts`
  edges (demo T2/T3/T6; p8 T8/T32) — all correct. No missed-contradict, no false-contradict →
  the Part-C.2 asymmetry is **n/a (0 polarity errors)**.
- **ATTRIBUTION:** third parties (demo's parents; p8's George/Jennifer/phone-friend) were
  **not** attributed to the user — the extractor produced no user-facts for their opinions.
- **BIRTH:** **zero** facts below 0.4 confidence; explicit statements consistently got 0.62–0.85.
  The "explicit statement → <0.4" failure mode did not occur once.
- The 5 `?` are all **over-extraction** (a goal possibly inferred but tagged explicit; a
  redundant 3rd support-value; a negation-as-node quirk) — NOT the P/A/B classes.

## The three reference failure modes did NOT reproduce
The architect's reference errors came from an earlier **TBG3** run (not on disk):
- msg4 "I'm not that disciplined → disciplined ▲14%" → here handled by `contradicts` edges (correct).
- msg5 "never missed a deadline → 0.20" → here `has discipline` **0.84** (no birth suppression).
- msg7 "my parents think…" → here **skipped** (no fact; correct non-attribution).
Most plausible cause: **op/ref + evidence_type (the flags we built to address exactly polarity/
attribution) work** — the current contract is materially cleaner than the reference run.

## Verdict against the pre-registered thresholds
Architect thresholds: <10% viable · 10–25% survives on engine forgiveness · >25% pivot mandatory.
Result **0.0% P/A/B (2.6% doubt) → contract VIABLE**, well below the "viable" line.

## Prediction scorecard (both over-predicted)
| | prediction | outcome |
|---|---|---|
| **Architect** | 10–20% P+A+B, POLARITY largest | **missed low** (0%); no class populated |
| **Code** | 8–18%, POLARITY largest, **BIRTH rarer** | **BIRTH prediction correct** (0 birth); overall also missed low |
Both expected a broken-ish contract; the data says the current contract is clean. Honest miss.

## Decision-relevant caveats (do not skip)
1. **n = 2 dialogues, 1 model, 1 config.** Not a broad baseline — "this contract on this
   material." The clean result should not be over-generalized.
2. **The reference (TBG3) run is unavailable**, so we could not verify the claimed errors nor
   confirm they were fixed vs never-present.
3. **This undercuts the "contract is broken → structured extraction is mandatory" rationale.**
   On this evidence, structured extraction must beat a contract that is already ~0% on P/A/B —
   a high bar. The pivot case now rests on *other* grounds (ingest variance ±0.40; the
   reliability layer), NOT on extractor semantic errors.
4. Over-extraction (2.6% doubt) is real but mild and is what op/ref-merge + prune absorb; it is
   an efficiency/noise issue, not a correctness one.

## Acceptance
- 0 code edits (mtime), 0 LLM (read-only). Scheme + PREREG before labels (mtime). `labels.jsonl`
  parses, all rows schema-valid, verdict ∈ enum. OTHER = 100% of doubt but only 2.6% of all
  decisions (« 20% threshold). `spot_check.md` = 10 human-readable rows.
