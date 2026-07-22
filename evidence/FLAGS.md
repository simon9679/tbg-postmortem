# Ballast — Environment Flag Reference

*Generated from AUDIT_REPORT.md 2026-07-10; update on any flag change.*

Every environment flag read anywhere in the codebase, with its default, where it is read,
whether it is read at **import time** (frozen at module load) or **call time** (flippable
mid-process), and who sets it.

| flag | default | read where (time) | ever set? |
|---|---|---|---|
| `TBG_OPREF` | 0 | tbg_extractor `_opref_enabled()` (call) | yes (demo/gate/adapter/audit) |
| `TBG_EVIDENCE_TYPE` | 0 | tbg_extractor `_evidence_type_enabled()` (call) | yes |
| `TBG_RANK_RENDER` | 1 | esmem_adapter:162 (call) | yes (rank-OFF ablation) |
| `TBG_DUMP_SIGNS` | 0 | esmem_adapter:63 (**import**) | yes |
| `TBG_TELEMETRY` | 0 | tbg_telemetry:24 (call) | yes (audit/realtest) |
| `TBG_TELEMETRY_PATH` | tbg_telemetry.jsonl | tbg_telemetry:35 (call) | yes (audit sets per-run) |
| `TBG_FIX_DECAY_TP` | 0 | tbg_engine:86 (**import**) + ballast_signals:125 (call) | yes (v1.1) |
| `TBG_PROV_ON` | 0 | tbg_extractor:39 (**import**) | yes (edge experiment) |
| `TBG_FIX_DOUBLE_EDGES` | 0 | tbg_engine:79 (**import**), used at 189 | **never** → always OFF |
| `TBG_DECAY_USE_LOGICAL_CLOCK` | 0 | tbg_engine:65 (**import**) | **never** |
| `TBG_DECAY_MIN_INTERVAL_TURNS` | 14 | tbg_engine:68 (**import**) | **never** |
| `TBG_DECAY_TURNS_PER_DAY` | 0.001 | tbg_engine:71 (**import**) | **never** |
| `TBG_LABEL_CANONICALIZATION` | 0 | tbg_extractor:525 (**import**) | **never** |
| `TBG_DISABLE_EXTRACTION_CONF_PENALTY` | 0 | tbg_extractor:529 (**import**) | **never** |
| `TBG_CLOSED_VOCAB` | 0 | tbg_extractor:535 (**import**) | **never** |
| `TBG_EXTRACT_DEBUG` | 0 | tbg_extractor:968 (**import**) | **never** |
| `TBG_TURN_DELAY_SECONDS` | 20 / 13 | pressure_gate:68, ballast_audit:269 (call) | yes (audit/pressure) |
| `CEREBRAS_RPS_DELAY` | 0 / 13 | esmem_gate:35, ablations (call) | yes |
| `CEREBRAS_LOG_HEADERS` | — | llm_client:299 (call) | yes (once) |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | see llm_client docstring | llm_client (call) | yes |
| `ESMEM_MAX_CALLS` / `GATE_MAX_CALLS` | 3000 / 2000 | esmem_gate:31, pressure_gate:72 (call) | yes |
| `AUDIT_SALT` | — (no default) | ballast_audit:247 (call) | required (or `--salt`) |

## Import-time gotcha (this has already bitten us)

Every flag marked **(import)** is frozen at module load — a caller MUST set it in `env`
**before** importing the module, or the default is baked in for the process. Call-time flags
can be flipped any time.

Example (from the audit runner, `ballast_audit.py`): `TBG_DUMP_SIGNS`/`TBG_OPREF`/
`TBG_EVIDENCE_TYPE` are set in `os.environ` at the top of `_run()` **before** `import
esmem_adapter`, precisely because `_DUMP_SIGNS` is an import-time constant.

## Never-set flags (read but never set by anyone → always default)

`TBG_FIX_DOUBLE_EDGES`, `TBG_DECAY_USE_LOGICAL_CLOCK`, `TBG_DECAY_MIN_INTERVAL_TURNS`,
`TBG_DECAY_TURNS_PER_DAY`, `TBG_LABEL_CANONICALIZATION`, `TBG_DISABLE_EXTRACTION_CONF_PENALTY`,
`TBG_CLOSED_VOCAB`, `TBG_EXTRACT_DEBUG`.

These are inert today. **Changing any of these defaults is a behavioral change to the frozen
v1.0 contour → it requires v1.2 pre-registration, not a silent edit.** (`TBG_DECAY_USE_LOGICAL_CLOCK`
and the history cap are two candidates already earmarked for v1.2.)
