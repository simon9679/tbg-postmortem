# Test 1 — Results (blind batch-audit on unseen data + dummies)

Corpus (B): 18 EvoEmo (runner never processed any) + 5 hand-authored dummies. 478 extractions.
Blind reviewer (Alexey) marked +/-/? on a 10-dialogue pack (top-5 by review_score + 5 bottom
non-dummy), random order, key hidden. Predictions in `PREREG.md`.

## Verdicts
| test | result | vs prediction |
|---|---|---|
| **T1-A** dummies in bottom third | **PASS** | both right (arch 75% / Code 90%) |
| **T1-B** blind top-5 match ≥3/5 | **FAIL (2/5)** | Code closer (arch 55% / Code 45%) |
| **T1-C** ≥90% dialogues survive | **PASS (100%, 23/23)** | both beaten (arch 70% / Code 85%) |

## T1-A — dummies (PASS)
All 5 dummies at the bottom (0–2 nodes, review_score 0–1); none in top-5. The ranking cleanly
separated mundane chit-chat from emotional dialogues. (One dummy, thread_w08, scored 1 — the
over-extraction leak I flagged in PREREG — still bottom.)

## T1-B — blind human match (FAIL)
- **TOP accuracy: 2/5** (evo_p8 and evo_p16 marked "+"; the two highest-scored, evo_p13=22 and
  evo_p2=18, plus evo_p12, marked "−"). Threshold was ≥3/5 → **fail**.
- **BOTTOM accuracy: 2/4** (evo_p5, evo_p17 marked "−"; evo_p4, evo_p3 marked "+").
- **doubt: 1** (evo_p1 marked "?" — correctly, see caveat).
- **Overall 4/9 = 44% (below chance ~50%).** The reviewer's own words: "не уверен вообще" —
  consistent with a genuinely hard/noisy discrimination. **The ranking did not carry usable
  signal to a naive reader on this material.**

**Confound — material coherence (surfaced mid-test, honestly).** Diagnosing the "?" item
(evo_p1) showed EvoEmo convs can be **stitched from disparate topics** (insomnia → boss drama →
Sarah/Mark relationship → sabbatical over 32 sessions; seeker unnamed early, "Sarah" late). So
"did the person change?" is only as well-posed as single-person coherence, which EvoEmo does not
guarantee. This confounds T1-B — but is NOT used as a rescue: the headline stands (fail). A clean
re-test needs verified-coherent material and a fresh pre-registration.

**Confound — quota (DISMISSED).** 62/478 (13%) turn-extractions failed_other (token-quota tail),
but every real conv reached a full graph (17×50 nodes, p9=31), so the ranking is not corrupted.

**Thin positive inside the negative:** the one clearly-coherent conv (evo_p8, Jimmy's recovery
arc) was a TOP hit — where the material is coherent, the human agreed with the ranking. n=1;
not overstated.

## T1-C — technical survival (PASS)
23/23 dialogues completed, 0 failed at the dialogue level. **0 format-driven failures**
(repair_rate 0.0, warnings {} — foreign format/length/English did not break extraction). Health
on foreign data: 478 extractions (ok 383 / no_update 33 / failed_other 62 = quota tail),
mean extraction_confidence 0.80, De Finetti interventions 25.

## p10 / p16 (truly-unseen) — reported separately
- **evo_p16** (truly-unseen, score 11) landed in **top-5** and was a **T1-B hit** ("+").
- **evo_p10** (truly-unseen, score 10, rank #7) fell between top-5 and bottom-half → not in the
  blind pack, so no T1-B verdict. Both survived extraction (T1-C).

## Prediction scorecard
- **T1-A:** both correct (dummies bottom). **T1-C:** both beaten (100% vs 70/85%).
- **T1-B:** Code's ~45% (skeptic, "all-emotional corpus → noisy discrimination") landed on the
  nose (44%, fail); Architect's ~55% (≥3/5 top) missed. Code's market skepticism confirmed again.

## Bottom line
**The "belief-dynamics ranking as a product" is not validated by a fair blind test** — a
domain-naive reader could not confirm the ranking's ordering (44%, below chance). This is the
**second** product-angle to miss a pre-registered bar (E2 conflict = PARITY; T1-B ranking =
FAIL). The stitched-material finding partly confounds T1-B but also independently weakens the
premise (the belief graph was measured on non-single-user data). The durable asset remains the
**reliability / measurement layer**, not the ranking or the conflict signal.
