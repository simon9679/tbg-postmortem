"""
TBG telemetry sink — append-only JSONL, env-gated.
TBG_TELEMETRY=1        -> enabled (default OFF: zero behavior, zero IO)
TBG_TELEMETRY_PATH=... -> output path (default: tbg_telemetry.jsonl)
Rationale: ingest is the dominant measured noise source on the memory
benchmark (±0.40 vs judge 0.10 / answerer 0.05, see EVAL_RELIABILITY_NOTE).
This module makes the ingest layer observable. Same pattern as
TBG_DUMP_SIGNS: a free side-product of the normal pipeline, no extra LLM calls.

No module-level state (lesson from _PROV): the flag is read dynamically and the
sink appends per event, so a single process can flip it and never accumulates
cross-user state in memory.
"""
import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def telemetry_enabled() -> bool:
    """Dynamic env read (like _opref_enabled), not a module constant."""
    return os.getenv("TBG_TELEMETRY", "0") == "1"


def emit(event: dict) -> None:
    """No-op when OFF. When ON: stamp UTC ts and append one JSON line.
    Telemetry must never crash the pipeline — any write error is logged, not raised."""
    if not telemetry_enabled():
        return
    try:
        record = dict(event)
        record["ts"] = datetime.now(timezone.utc).isoformat()
        path = os.getenv("TBG_TELEMETRY_PATH", "tbg_telemetry.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001 — telemetry is best-effort, never fatal
        logger.warning(f"TBG telemetry emit failed: {e}")
