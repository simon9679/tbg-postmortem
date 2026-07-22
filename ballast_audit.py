"""
Ballast Batch-Audit Runner v0 — pure orchestration over the existing pipeline.
Core (tbg_engine/tbg_extractor/tbg_schema) is NOT touched; demo_ballast is NOT
modified (we compose its pure render sub-helpers). No policy / no alerts / no
thresholds — this only ingests, snapshots, renders, and ORDERS for human review.

Input : a directory of dialogue files (JSON or JSONL), one user/dialogue per file:
    {"dialogue_id": "case_001", "turns": [
       {"role": "user", "text": "...", "ts": "2026-01-15T10:00:00Z"},
       {"role": "assistant", "text": "..."}]}
Output: <out>/<pseudonym>/{snapshot.json, report.html, signs.txt}
        <out>/{batch_summary.html, batch_health.json, _telemetry.jsonl,
               _mapping.json, _audit_state.json}

CLI: py -3 ballast_audit.py --input <dir> --output <dir> [--limit N] [--resume]
     privacy: --salt <s> or env AUDIT_SALT (NO default — missing salt => hard stop).
"""
import os
import sys
import json
import glob
import shutil
import hashlib
import asyncio
import argparse
import datetime
from collections import Counter, defaultdict

# Flag config is FIXED by the runner (printed into every report for reproducibility).
# These MUST be set before importing esmem_adapter (its _DUMP_SIGNS is a module
# constant read at import time — same for op/ref/evidence_type resolution paths).
FLAG_CONFIG = {
    "TBG_OPREF": "1",
    "TBG_EVIDENCE_TYPE": "1",
    "TBG_DUMP_SIGNS": "1",
    "TBG_TELEMETRY": "1",
}


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _pseudonym(salt: str, dialogue_id: str) -> str:
    return hashlib.sha256((salt + str(dialogue_id)).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def _load_dialogue(path: str) -> dict:
    """Load one dialogue file (JSON object, or JSONL whose first non-empty line is
    the object). Raises ValueError with a human reason on malformed input."""
    raw = open(path, encoding="utf-8").read().strip()
    if not raw:
        raise ValueError("empty file")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # JSONL: take the first parseable line
        obj = None
        for line in raw.splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                break
        if obj is None:
            raise ValueError("no parseable JSON")
    if not isinstance(obj, dict):
        raise ValueError("top-level is not an object")
    if "dialogue_id" not in obj:
        raise ValueError("missing dialogue_id")
    if not isinstance(obj.get("turns"), list) or not obj["turns"]:
        raise ValueError("missing/empty turns")
    return obj


def _to_adapter_conv(dialogue: dict, pseudonym: str) -> dict:
    """Convert the audit input format to the adapter conversation format. Only
    user turns become the belief source (seeker); the following assistant turn is
    supporter context. emotion/topic are absent (not invented)."""
    turns = dialogue["turns"]
    sessions = []
    i = 0
    while i < len(turns):
        t = turns[i]
        if t.get("role") == "user":
            dlg = [{"idx": 1, "role": "seeker", "content": (t.get("text") or "")}]
            ts = t.get("ts", "")
            if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
                dlg.append({"idx": 2, "role": "supporter",
                            "content": (turns[i + 1].get("text") or "")})
                i += 2
            else:
                i += 1
            sessions.append({"session": len(sessions) + 1, "timestamp": ts,
                             "emotion": "", "topic": "", "dialogue": dlg})
        else:
            i += 1  # skip leading/orphan assistant turns
    return {"id": pseudonym, "dialog_history": sessions}


def _user_messages(dialogue: dict):
    msgs, ts = [], []
    for t in dialogue["turns"]:
        if t.get("role") == "user":
            msgs.append(t.get("text") or "")
            ts.append(t.get("ts", ""))
    return msgs, ts


# ---------------------------------------------------------------------------
# Review ranking (deterministic, from core signals only)
# ---------------------------------------------------------------------------
# review_score = (#down-trajectory beliefs) + (#active conflicts)
#              + (turning point in last third of turns ? 1 : 0) + (#oscillating)
# Simplest defensible formula; ORDERING ONLY, not a risk score.
REVIEW_FORMULA = ("(#down-trajectory beliefs) + (#active conflicts) "
                  "+ (turning-point in last third ? 1 : 0) + (#oscillating)")
REVIEW_BANNER = ("Sorting heuristic for human review prioritization — NOT detection, "
                 "NOT a risk score. Signals are one draw from a noisy extraction "
                 "process (see ingest-variance note).")


def _review_score(snap: dict, n_turns: int):
    core = snap["core"]
    a = sum(1 for t in core["trajectories"] if t.get("direction") == "down")
    b = len(core["conflicts"])
    tps = core["turning_points"]["points"]
    thr = (2 * n_turns) // 3 if n_turns else 0
    c = 1 if (n_turns and any(p["message_count"] >= thr for p in tps)) else 0
    d = len(core["oscillating"])
    return {"score": a + b + c + d, "down": a, "conflicts": b,
            "tp_last_third": c, "oscillating": d}


# ---------------------------------------------------------------------------
# HTML (compose demo_ballast's pure sub-helpers; demo file untouched)
# ---------------------------------------------------------------------------
def _dialogue_html(D, state: dict, pseudonym: str, cfg_line: str) -> str:
    L = D.L
    h = [f"<!doctype html><meta charset='utf-8'><title>Ballast audit — {pseudonym}</title>"
         f"<style>{D._CSS}</style>"]
    h.append(f"<h1>Ballast audit — <code>{pseudonym}</code></h1>")
    h.append(f"<div class='sub'>{cfg_line}</div>")
    h.append(f"<div class='sub'>{L('xling')}</div>")
    h.append(f"<h2>{L('s1_h')}</h2><div class='sub'>{L('s1_sub')}</div>")
    h.append(D._drift(state))
    h.append(f"<h2>{L('s2_h')}</h2><div class='sub'>{L('s2_sub')}</div>")
    h.append(D._svg_traj(state["trajectories"]))
    h.append(f"<h2>{L('s3_h')}</h2><div class='sub'>{L('s3_sub')}</div>")
    h.append(D._conflicts(state))
    h.append(f"<div class='note' style='margin-top:8px'>{L('nondet')}</div>")
    h.append(f"<h2>{L('s4_h')}</h2><div class='sub'>{L('s4_sub')}</div>")
    h.append(D._turning(state))
    h.append(f"<h2>{L('s5_h')}</h2><div class='sub'>{L('s5_sub')}</div>")
    h.append(D._ambiv(state))
    h.append(f"<h2>{L('s6_h')}</h2><div class='sub'>{L('s6_sub')}</div>")
    h.append(D._contrast(state))
    return "".join(h)


def _batch_summary_html(D, rows, health, cfg_line, schema_version) -> str:
    import html as H
    css = D._CSS
    h = [f"<!doctype html><meta charset='utf-8'><title>Ballast audit — batch summary</title>"
         f"<style>{css}</style>"]
    h.append("<h1>Ballast audit — batch summary</h1>")
    h.append(f"<div class='sub'>{cfg_line}</div>")
    h.append(f"<div class='sub'>snapshot schema {schema_version} · pseudonymized; "
             f"mapping stored separately (_mapping.json)</div>")

    # Review ranking
    h.append("<h2>Review order</h2>")
    h.append(f"<div class='note' style='margin:6px 0'><b>⚠ {H.escape(REVIEW_BANNER)}</b></div>")
    h.append(f"<div class='sub'>score = {H.escape(REVIEW_FORMULA)} · descending</div>")
    h.append("<table style='border-collapse:collapse;width:100%;font-size:13px'>")
    h.append("<tr style='text-align:left;color:#8a93a3'><th>#</th><th>pseudonym</th>"
             "<th>score</th><th>down</th><th>conflicts</th><th>tp⅓</th><th>oscillating</th></tr>")
    for i, r in enumerate(sorted(rows, key=lambda r: (-r["score"], -r["down"],
                                                      -r["conflicts"], -r["tp_last_third"])), 1):
        h.append(f"<tr style='border-top:1px solid #232937'><td>{i}</td>"
                 f"<td><a href='{r['pseudonym']}/report.html'><code>{r['pseudonym']}</code></a></td>"
                 f"<td><b>{r['score']}</b></td><td>{r['down']}</td><td>{r['conflicts']}</td>"
                 f"<td>{r['tp_last_third']}</td><td>{r['oscillating']}</td></tr>")
    h.append("</table>")

    # Telemetry health
    h.append("<h2>Ingest health (this run's telemetry)</h2>")
    h.append("<div class='sub'>the dominant measured noise layer, made inspectable "
             "(see EVAL_RELIABILITY_NOTE)</div>")
    h.append("<div class='amb'>")
    h.append(f"extractions: <b>{health['extractions']}</b> · "
             f"status: {H.escape(json.dumps(health['status'], ensure_ascii=False))}<br>")
    h.append(f"repair rate: <b>{health['repair_rate']}</b> · "
             f"mean extraction_confidence (ok): <b>{health['mean_extraction_confidence']}</b><br>")
    h.append(f"warnings by type: {H.escape(json.dumps(health['warnings_by_type'], ensure_ascii=False))}<br>")
    h.append(f"De Finetti interventions (turns): <b>{health['definetti_applied_total']}</b>")
    h.append("</div>")
    return "".join(h)


# ---------------------------------------------------------------------------
# Telemetry health aggregation (from this run's _telemetry.jsonl)
# ---------------------------------------------------------------------------
def _health(telemetry_path: str) -> dict:
    status = Counter()
    warn_types = Counter()
    confs = []
    repair = 0
    n_ex = 0
    definetti = 0
    if os.path.exists(telemetry_path):
        for line in open(telemetry_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "extraction":
                n_ex += 1
                status[r.get("status")] += 1
                if r.get("repair_used"):
                    repair += 1
                if r.get("status") == "ok" and "extraction_confidence" in r:
                    confs.append(r["extraction_confidence"])
                for w in r.get("warnings", []) or []:
                    warn_types[str(w).split(":", 1)[0]] += 1
            elif r.get("event") == "turn_dynamics":
                definetti += r.get("definetti_applied", 0) or 0
    return {
        "extractions": n_ex,
        "status": dict(status),
        "repair_rate": round(repair / n_ex, 3) if n_ex else 0.0,
        "mean_extraction_confidence": round(sum(confs) / len(confs), 3) if confs else None,
        "warnings_by_type": dict(warn_types),
        "definetti_applied_total": definetti,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def _run(args):
    salt = args.salt or os.getenv("AUDIT_SALT")
    if not salt:
        print("ERROR: no salt. Pass --salt <string> or set AUDIT_SALT. "
              "Pseudonymization is mandatory; refusing to run without it.", file=sys.stderr)
        return 2

    # WARNING: set flag env BEFORE importing esmem_adapter (module-level _DUMP_SIGNS
    # is read at import time). Runner is single-process sequential, so a per-run
    # telemetry path via env is race-free.
    for k, v in FLAG_CONFIG.items():
        os.environ[k] = v
    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)
    tele_path = os.path.join(out, "_telemetry.jsonl")
    os.environ["TBG_TELEMETRY_PATH"] = tele_path

    import esmem_adapter as AD
    import demo_ballast as D
    from ballast_signals import snapshot, SCHEMA_VERSION
    from llm_client import gemini_call
    D.LANG = "en"  # module-level LANG set once per run; runner is single-process sequential

    delay = float(os.getenv("TBG_TURN_DELAY_SECONDS",
                            os.getenv("CEREBRAS_RPS_DELAY", "13")))
    cfg_line = ("flags: " + " ".join(f"{k}={v}" for k, v in FLAG_CONFIG.items())
                + f" · provider={os.getenv('LLM_PROVIDER','cerebras')}"
                + f" · turn_delay={delay}s · snapshot_schema={SCHEMA_VERSION} · pseudonymized")

    async def llm_fn(prompt):
        r = await gemini_call(prompt, timeout=90.0)
        if delay:
            await asyncio.sleep(delay)
        return r

    # State: resume-aware checkpoint
    state_path = os.path.join(out, "_audit_state.json")
    st = {"completed": [], "failed": [], "skipped": [], "config": FLAG_CONFIG,
          "started": _utc_now()}
    if args.resume and os.path.exists(state_path):
        st = json.load(open(state_path, encoding="utf-8"))
    done = set(st["completed"])
    mapping_path = os.path.join(out, "_mapping.json")
    mapping = json.load(open(mapping_path, encoding="utf-8")) if os.path.exists(mapping_path) else {}

    files = sorted(glob.glob(os.path.join(args.input, "*.json"))
                   + glob.glob(os.path.join(args.input, "*.jsonl")))
    if args.limit:
        files = files[:args.limit]

    rows = []
    # rebuild rows for already-completed dialogues (so summary is complete on resume)
    for pn in done:
        sp = os.path.join(out, pn, "snapshot.json")
        if os.path.exists(sp):
            snap = json.load(open(sp, encoding="utf-8"))
            n_turns = snap.get("message_count") or 0
            rows.append({"pseudonym": pn, **_review_score(snap, n_turns)})

    for path in files:
        try:
            dlg = _load_dialogue(path)
        except Exception as e:
            st["skipped"].append({"file": os.path.basename(path), "reason": str(e)[:120]})
            print(f"SKIP {os.path.basename(path)}: {e}")
            continue

        pn = _pseudonym(salt, dlg["dialogue_id"])
        mapping[pn] = dlg["dialogue_id"]
        if pn in done:
            print(f"RESUME-SKIP {pn} (already done)")
            continue

        ddir = os.path.join(out, pn)
        os.makedirs(ddir, exist_ok=True)
        cwd_sign = f"tbg_signs_{pn}.txt"   # adapter writes here (user_id = pseudonym)
        try:
            conv = _to_adapter_conv(dlg, pn)
            tbg = await AD.ingest(conv, llm_fn)   # same path as demo/bench
            # relocate the CWD sign dump into the dialogue output folder immediately
            dst_sign = os.path.join(ddir, "signs.txt")
            if os.path.exists(cwd_sign):
                shutil.move(cwd_sign, dst_sign)

            snap = snapshot(tbg)
            json.dump(snap, open(os.path.join(ddir, "snapshot.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)

            umsgs, uts = _user_messages(dlg)
            per_turn = D.parse_sign_dump(dst_sign) if os.path.exists(dst_sign) else []
            state = {**D.extract_final(tbg), "user_messages": umsgs,
                     "timestamps": uts, "per_turn": per_turn}
            open(os.path.join(ddir, "report.html"), "w", encoding="utf-8").write(
                _dialogue_html(D, state, pn, cfg_line))

            rows.append({"pseudonym": pn, **_review_score(snap, len(umsgs))})
            st["completed"].append(pn)
            print(f"OK {pn}  turns={len(umsgs)} nodes={snap['core'] and len(snap['core']['beliefs'])}")
        except Exception as e:
            st["failed"].append({"pseudonym": pn, "reason": str(e)[:160]})
            print(f"FAIL {pn}: {str(e)[:160]}")
        finally:
            # cleanup: no tbg_signs_*.txt may survive in CWD, even on failure
            if os.path.exists(cwd_sign):
                try:
                    os.remove(cwd_sign)
                except OSError:
                    pass
            # checkpoint after every dialogue
            json.dump(st, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            json.dump(mapping, open(mapping_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            try:
                os.chmod(mapping_path, 0o600)  # best-effort owner-only (limited on Windows)
            except OSError:
                pass

    # Batch outputs
    health = _health(tele_path)
    json.dump(health, open(os.path.join(out, "batch_health.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    open(os.path.join(out, "batch_summary.html"), "w", encoding="utf-8").write(
        _batch_summary_html(D, rows, health, cfg_line, SCHEMA_VERSION))

    print(f"\nDONE: completed={len(st['completed'])} failed={len(st['failed'])} "
          f"skipped={len(st['skipped'])} -> {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Ballast batch-audit runner v0")
    ap.add_argument("--input", required=True, help="directory of dialogue files")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--salt", default=None, help="pseudonymization salt (or env AUDIT_SALT)")
    a = ap.parse_args()
    sys.exit(asyncio.run(_run(a)))


if __name__ == "__main__":
    main()
