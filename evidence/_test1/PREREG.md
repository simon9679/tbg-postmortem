# Test 1 — Pre-registration (blind batch-audit on unseen data + dummies)

*Registered BEFORE the run (mtime). This test may fail — failure is a result.*

## Material (Step 0)
- **Truly-unseen ES-MemEval:** only **2** (p10 18 sessions, p16 25) — the other 16 were
  gate-ingested + entered QA. ESConv is **not on disk** and not downloadable in this environment.
- **All 18 EvoEmo via the runner:** the **runner (`ballast_audit`) never processed any of the
  18** (it only ran demo+en1+en2), so from the PRODUCT's view all 18 are unseen; the belief-graph
  saw 16 (caveat). ~**401 calls** (session-granularity; 1 call/session).
- Converter groups one EvoEmo session → one runner turn (matches system granularity; flattening
  per utterance would explode calls). 5 hand-authored dummies (no state dynamics), 15–16 turns each.
- **Corpus decision is pending the architect** (2 truly-unseen [underpowered] vs all-18-via-runner
  [powered, 16 gate-seen caveat] vs supply ESConv). Predictions below are corpus-agnostic.

## Predictions (before numbers)
### T1-A — dummies land in the bottom third of the ranking
- **Architect:** all dummies bottom third, ~75%. If any dummy reaches top-5 → the product as-is
  is not sellable.
- **Code:** **~90% (stronger).** Dummies have 0 conflicts / 0 down-trajectories / 0 oscillating
  / no turning points → `review_score` ≈ 0, firmly bottom. Named risk: extractor over-extraction
  (Phase-0's "appreciates support" template) could inject spurious dynamics into chit-chat and
  lift a dummy off 0 — the one way this fails.

### T1-B — blind human match: ≥3 of the runner's top-5 flagged "worth reviewer attention"
- **Architect:** ~55%.
- **Code:** **~45% (more skeptical).** On an all-emotional-support corpus, *everything* looks
  review-worthy to a human, so discrimination is weak: the ranking correlates with "emotional
  heaviness" and so does the human, but the specific top-5 overlap is noisy. The dummies make the
  bottom obvious (T1-A); telling apart the *heavy* dialogues is the hard, noisy part.

### T1-C — technical survival: ≥90% of dialogues processed without `failed`
- **Architect:** ~70%.
- **Code:** **~85% (more optimistic).** The extractor is robust (Phase-0 clean), EvoEmo is its
  native format, dummies are clean English. Empty extraction on a dummy is a valid empty graph,
  not a `failed`. Risk: a transient rate-limit mid-run (retried) or an odd long session.

## Outcome reading (fixed now)
- Any dummy in top-5 → product not sellable as-is (hard stop signal).
- T1-B ≥3/5 → ranking carries real signal to a naive reader; <3/5 → ranking not yet useful,
  honest negative before any client letter.
- Health report (repair rate, warnings) on unfamiliar data is itself a deliverable regardless.
