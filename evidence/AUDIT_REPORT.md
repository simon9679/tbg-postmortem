# Ballast — Full Audit (read-only, zero edits)

**Mode:** read-only. Nothing fixed, no code touched (this report is the only new file).
Every finding carries line numbers + grep evidence. This is an INPUT for future TZs,
not a set of fixes. Classes: `dead` / `wasted` / `doc-lie` / `dataflow` / `config` / `orphan`.
Deletion safety: `safe` / `needs-flag` / `frozen-v1.0-touch` (core file on the gated E-path —
no edit without pre-registration).

Legend for verdicts: *confirms external audit* = matches the already-known list; *new* =
not previously flagged; *refutes* = contradicts a prior claim (code shown).

---

## TOP-10 findings (by importance)

1. **Whole snapshot/drift DB path is dead in the carve-out.** `update_tbg_background`
   (engine:1082) has **no caller repo-wide** → its callees `save_snapshot` (323),
   `ConfidenceSnapshot.from_tbg` (schema:94), and `get_drift_labels` (970) are unreachable
   in Ballast; `load_last_snapshots` (353) and `ConfidenceSnapshot.diff` (schema:100) have
   no caller at all. `get_insight` is called once (`tbg_realtest:232`) **without** snapshots.
   `class` dead · `frozen-v1.0-touch` (all in `tbg_engine`/`tbg_schema`).
2. **`node_type` docstring lies** ("used for conflict gating", schema:46). Set by extractor
   (283/380/414/484), read only by `ballast_signals` type_profile (217). **No `node_type`
   comparison anywhere in engine/SDL** (grep for `node_type ==`/`.node_type` in gating → none).
   `doc-lie` · confirms external audit + supplies proof.
3. **`strong_contradict_ids` is fully dead.** SDL never appends it (only `contradict_ids.append`
   at extractor:429); no external caller constructs it either. Engine reads it (401) but it is
   always `[]`. `dead` · confirms external audit.
4. **`reinforce_ids` dead in the extraction pipeline** (SDL never appends; init [] at 339, read
   at 428, passed at 453, never grown). **BUT alive via manual bench deltas**
   (`ballast_live_benchmark:125`, `ballast_pressure_test:89` build `TBGDelta(reinforce_ids=[...])`)
   and consumed by engine:389. `dead`(pipeline)/`alive`(bench) · refines external audit.
5. **`_amf_state` = wasted compute.** `compute_graph_amf(tbg.nodes)` runs every `apply_delta`
   (engine:283 and 465), stores `tbg._amf_state`, **never read anywhere** (no reader in grep).
   AMF is also empirically ≈0 at history-cap 5. `wasted` · confirms external audit + adds "runs
   twice per apply_delta".
6. **Silent label collision in edge resolution.** `existing_by_label = {n.label.lower(): nid …}`
   (extractor:496) — two nodes with the same lowercased label → **last one silently wins**;
   used at 505-506 to resolve edge src/tgt. `dataflow` · confirms external audit's pattern.
7. **`ConfidenceSnapshot.diff` (schema:100) is an orphan** — `get_drift_labels` compares via
   `latest.state.get(...)` (986), not `.diff()`. No caller of `.diff` exists. `dead` · new.
8. **`TBG_NLI_ENABLED` is doc-only** — appears solely in the extractor header comment
   (tbg_extractor:13); never read as env (grep `TBG_NLI` → only line 13). Feature removed,
   docstring survived. `doc-lie`/`dead` · confirms external audit.
9. **Eight config flags are read but never set by anyone** → always default (see Config table).
   Notably `TBG_FIX_DOUBLE_EDGES` (engine:79, read at 189) and `TBG_CLOSED_VOCAB` /
   `TBG_LABEL_CANONICALIZATION` / `TBG_DISABLE_EXTRACTION_CONF_PENALTY` (import-time constants).
   `config` · new inventory.
10. **`llm_client` default-config docstring is stale/incomplete.** Header (11-15) says Groq
    max_tokens=1024, but code (192) uses `3000` for reasoning models; **Cerebras is omitted**
    from the "Default configs" block though it's the workhorse, and its default model
    `zai-glm-4.7` (265) is never used (always overridden to gpt-oss-120b). `doc-lie` · new.

---

## `tbg_schema.py`

| line | finding | class | evidence | delete-safety |
|---|---|---|---|---|
| 46 | `node_type` doc "used for conflict gating" — never read for gating | doc-lie | set by extractor; read only by ballast_signals:217; no `node_type ==` in engine/SDL | frozen-v1.0-touch (field kept; doc only) |
| 31 | `signal_span` — legacy, "no longer populated" (own comment), grep: def only | dead field | no writer/reader | frozen-v1.0-touch |
| 43 | `axis_projection` — legacy, EPA removed (own comment), grep: def only | dead field | no writer/reader | frozen-v1.0-touch |
| 78 | `reinforce_ids` — see engine/extractor; empty from SDL | dead(pipeline) | extractor never appends | frozen-v1.0-touch |
| 80 | `strong_contradict_ids` — never populated by anyone | dead | no append anywhere | frozen-v1.0-touch |
| 94 | `ConfidenceSnapshot.from_tbg` — only caller is dead `save_snapshot` | dead | engine:324 only | frozen-v1.0-touch |
| 100 | `ConfidenceSnapshot.diff` — no caller (drift uses `.state`) | dead/orphan | grep `.diff(` → def only | frozen-v1.0-touch |
| 238 | `upsert_edge` — **alive** (engine:190, 217) | alive | 2 engine callers | — |

## `tbg_engine.py`

| line | finding | class | evidence | delete-safety |
|---|---|---|---|---|
| 1082 | `update_tbg_background` — **no caller repo-wide** (DB-backed prod path) | dead/orphan | grep `update_tbg_background(` → def only | frozen-v1.0-touch |
| 323/353 | `save_snapshot`/`load_last_snapshots` — dead (only reachable via 1082 / no caller) | dead | callers = update_tbg_background (1098,1126) / none | frozen-v1.0-touch |
| 970 | `get_drift_labels` — reachable only via `get_insight(snapshots≠None)`; realtest calls with None | dead(carve-out) | tbg_realtest:232 `get_insight(tbg)` | frozen-v1.0-touch |
| 283,465 | `_amf_state = compute_graph_amf(...)` computed twice/turn, never read | wasted | no reader of `_amf_state` | frozen-v1.0-touch |
| 79/189 | `_FIX_DOUBLE_EDGES` flag read (189) but never set → always default OFF | config | grep: no setter | needs-flag |
| 389/401 | engine consumes `reinforce_ids`/`strong_contradict_ids` — always empty from SDL | dead(input) | extractor never appends | frozen-v1.0-touch |
| 997 | `get_insight` — alive (realtest); `snapshots` branch never exercised in carve-out | partial-dead | — | frozen-v1.0-touch |

## `tbg_extractor.py`

| line | finding | class | evidence | delete-safety |
|---|---|---|---|---|
| 496 | `existing_by_label` dict-comp — duplicate lowercased labels: last wins **silently** | dataflow | used 505-506 for edge resolve | frozen-v1.0-touch |
| 13 | header lists `TBG_NLI_ENABLED=1` — no env read anywhere | doc-lie | grep `TBG_NLI` → line 13 only | safe (comment) |
| 11 | header "oscillation pair linking (auto conflicts_with)" — no implementation (removed SDL) | doc-lie | prior session confirmed; grep `oscillation` → line 11 only | safe (comment) |
| 1019-1020 | `existing_tbg_summary`, `existing_label_to_uuid` params — "kept for API compat, unused" | dead(param) | own comment; not read in body | frozen-v1.0-touch |
| 339/428/453 | `reinforce_ids` init/read/pass but never appended | dead(pipeline) | grep: no `reinforce_ids.append` | frozen-v1.0-touch |
| 525/529/535 | `TBG_LABEL_CANONICALIZATION`/`TBG_DISABLE_EXTRACTION_CONF_PENALTY`/`TBG_CLOSED_VOCAB` — never set | config | grep: no setter | needs-flag |
| 968 | `TBG_EXTRACT_DEBUG` — never set | config | grep: no setter | needs-flag |

## `llm_client.py`

| line | finding | class | evidence | delete-safety |
|---|---|---|---|---|
| 11-15 | "Default configs" block: Groq says max_tokens=1024, code (192) = 3000 for reasoning | doc-lie | line 192 `"3000" if _reasoning else "1024"` | safe (doc) |
| 11-15 | Cerebras absent from Default-configs block though it's the workhorse | doc-lie | block lists only 4 providers | safe (doc) |
| 265 | Cerebras default model `zai-glm-4.7` never used (always overridden to gpt-oss-120b) | dead(default) | every runner sets LLM_MODEL | safe |
| 9 | "free 1M tok/day" — real wall is 5 rpm + 30K tpm + 1M/day (rate-bucket) | doc-lie(minor) | EVAL_RELIABILITY_NOTE / ratelimit log | safe (doc) |

## `amf_filter.py`

| line | finding | class | evidence | delete-safety |
|---|---|---|---|---|
| — | `compute_graph_amf` alive (engine imports it) but its output (`_amf_state`) is never read | wasted | see engine 283/465 | frozen-v1.0-touch (engine side) |

## Wrapper: `ballast_signals.py` / `tbg_telemetry.py` / `esmem_adapter.py` / `demo_ballast.py` / `ballast_audit.py`

| file | finding | class | evidence |
|---|---|---|---|
| ballast_signals | consumers: `_test_signals.py`, `ballast_audit.py` — alive | — | grep |
| ballast_signals | reads `node_type`/`domain`/`stance` via `_profile` (getattr) — sole readers of these fields | — | 215-217 |
| tbg_telemetry | alive (imported by tbg_engine:26, tbg_extractor:27); no module state | — | grep |
| esmem_adapter | `_DUMP_SIGNS` (63) import-time const; sign-dump path hardcoded to CWD `tbg_signs_<user_id>.txt` (125) | config/dataflow | ballast_audit relocates it (known) |
| demo_ballast | `LANG` module global; `_build_state` writes `demo_state_ru.json`, `render_demo` writes fixed HTML (demo-bound) | dataflow | audit runner composes sub-helpers instead (known) |
| ballast_audit | alive (script); imports demo_ballast + ballast_signals | — | grep |

## Cognitive layer (inventory only)

| file | imported by | verdict |
|---|---|---|
| `policy.py` | `pressure_gate.py` (312,374), `test_policy_logic.py`, `test_policy_live.py` | alive only in pressure-gate + tests; not in shipped pipeline |
| `mode_engine.py` | `ballast_pressure_test.py`, `ballast_live_benchmark.py` | alive only in 2 bench scripts |
| `dissonance_engine.py` | `ballast_pressure_test.py`, `ballast_live_benchmark.py` | alive only in 2 bench scripts |
| `intervention_engine.py` | `ballast_pressure_test.py`, `ballast_live_benchmark.py` | alive only in 2 bench scripts |
| `pressure_gate.py` | (nobody) — run as script | standalone script, not orphan |
| `api.py` | — | **absent from tree** (in TZ scope but does not exist) |

**None of the cognitive-layer modules is imported by the core pipeline, demo, gate, or audit
runner** — they are reachable only from `pressure_gate.py` (policy) and two bench scripts. This
matches "not positively measured, out of product."

---

## Config-hygiene: full env-flag inventory

| flag | default | read where (time) | ever set? |
|---|---|---|---|
| `TBG_OPREF` | 0 | extractor `_opref_enabled()` (call) | yes (demo/gate/adapter/audit) |
| `TBG_EVIDENCE_TYPE` | 0 | extractor `_evidence_type_enabled()` (call) | yes |
| `TBG_RANK_RENDER` | 1 | esmem_adapter:162 (call) | yes (rankoff ablation) |
| `TBG_DUMP_SIGNS` | 0 | esmem_adapter:63 (**import**) | yes |
| `TBG_TELEMETRY` / `_PATH` | 0 / file | tbg_telemetry (call) | yes (audit/realtest) |
| `TBG_FIX_DECAY_TP` | 0 | engine:86 (import) + signals:125 (call) | yes (v1.1) |
| `TBG_PROV_ON` | 0 | extractor:39 (import) | yes (edge experiment) |
| `TBG_FIX_DOUBLE_EDGES` | 0 | engine:79 (import), used 189 | **never** → always OFF |
| `TBG_DECAY_USE_LOGICAL_CLOCK` | 0 | engine:65 (import) | **never** |
| `TBG_DECAY_MIN_INTERVAL_TURNS` | 14 | engine:68 (import) | **never** |
| `TBG_DECAY_TURNS_PER_DAY` | 0.001 | engine:71 (import) | **never** |
| `TBG_LABEL_CANONICALIZATION` | 0 | extractor:525 (import) | **never** |
| `TBG_DISABLE_EXTRACTION_CONF_PENALTY` | 0 | extractor:529 (import) | **never** |
| `TBG_CLOSED_VOCAB` | 0 | extractor:535 (import) | **never** |
| `TBG_EXTRACT_DEBUG` | 0 | extractor:968 (import) | **never** |
| `TBG_TURN_DELAY_SECONDS` | 20 / 13 | pressure_gate:68, audit:269 (call) | yes (audit/pressure) |
| `CEREBRAS_RPS_DELAY` | 0 / 13 | gate/ablations (call) | yes |
| `CEREBRAS_LOG_HEADERS` | — | llm_client:299 (call) | yes (once) |

**Import-time gotcha (already bit us):** every flag marked "(import)" is frozen at module load —
a caller must set it BEFORE importing the module (the audit runner does this deliberately for
`TBG_DUMP_SIGNS`). Call-time flags can be flipped mid-process.

---

## Import graph / orphans

- No truly orphan files: all scope files are either imported or run as `__main__`.
- **Only consumer of `ballast_signals.snapshot()`**: `_test_signals.py` + `ballast_audit.py`.
- **`amf_filter`** imported only lazily by `tbg_engine` (282,464); its result is unread (wasted).
- Core pipeline import spine: `esmem_adapter`/`demo_ballast`/`ballast_audit` → `tbg_extractor`
  + `tbg_engine` + `tbg_schema` + `tbg_telemetry`; `ballast_signals` → `tbg_engine` (constants).

---

## Everything that touches the frozen v1.0 contour (NOT edit candidates without pre-registration)

All of these live in `tbg_engine.py` / `tbg_extractor.py` / `tbg_schema.py` — the gated E-path.
Even a "dead code" removal here edits a frozen-contour file and changes the golden byte image:

- snapshot/drift DB path (engine 323/353/970/1082; schema 94/100) — dead but frozen-contour.
- `_amf_state` wasted compute (engine 283/465) — removing it changes apply_delta bytes.
- dead channels `reinforce_ids`/`strong_contradict_ids` (schema 78/80; engine 389/401) — frozen.
- legacy fields `signal_span`/`axis_projection` (schema 31/43) — frozen (serialization compat).
- `existing_by_label` collision (extractor:496) — a real dataflow bug, but a fix mutates the
  frozen extraction path → needs pre-registration + re-ingest measurement (ingest variance ±0.40).
- `node_type` doc-lie (schema:46) — even a comment fix edits a frozen-contour file.

**Safe (non-frozen) doc/config candidates:** `llm_client` docstring (10), `TBG_NLI`/oscillation
header comments in extractor (8 — but they sit inside a frozen file, so still a byte change),
and the never-set config flags (documentation, not removal).

---

## Cross-check vs external audit

- **Confirms:** reinforce_ids/strong_contradict_ids dead, TBG_NLI dead, node_type/domain/stance
  unread by engine, `_amf_state` unread + AMF≈0, `_PROV` cross-user (comment already added),
  existing_tbg_summary/label_to_uuid unused, label-collision pattern.
- **Refines:** reinforce_ids is *not* globally dead — bench scripts feed it manually; `_amf_state`
  is recomputed **twice** per apply_delta.
- **New:** whole snapshot/drift DB path dead (update_tbg_background orphan), `ConfidenceSnapshot.diff`
  orphan, `llm_client` default-config docstring stale (Groq 1024→3000, Cerebras omitted), eight
  never-set config flags, `api.py` absent.
- **Refutes:** none — no external claim contradicted by the code.

*Read-only audit. No `.py` modified. This file is documentation, not code.*
