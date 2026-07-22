#!/usr/bin/env python3
"""
Scrooge attribution + recognition control. WITHOUT changing engine behavior.

Captures per turn (logging-only, via tbg_extractor._PROV):
  - RAW LLM output: raw_facts, raw_edges (extractor's raw output, BEFORE SDL)
  - SDL provenance: branch tag on each contradict/reinforce/edge
  - ENGINE output: confidence/prev/delta/⚡ per node after apply_delta
for Variant A (original Scrooge) and Variant B (disguised — same arc, new text).
+ recognition probe (1 call) on A: does the model recognize the source.

Budget: A 12 + B 12 + probe 1 = 25 claude-sonnet-4-6 calls (<30).
Analysis — offline (--analyze), 0 calls.

RUN:
  py -3 tools/attribution_run.py --run       # spends ~25 anthropic calls
  py -3 tools/attribution_run.py --analyze    # offline, reads attribution_{A,B}.json
"""
import os
import sys
import json
import asyncio
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

A_PATH = HERE / "attribution_A.json"
B_PATH = HERE / "attribution_B.json"
PROBE_PATH = HERE / "attribution_recognition.json"


def _frozen(name: str) -> Path:
    """For --analyze: read the frozen artifact from the published evidence/ folder;
    fall back to tools/ (a just-generated live run writes there)."""
    ev = ROOT / "evidence" / name
    return ev if ev.exists() else HERE / name

SPARK = 0.20  # ⚡ threshold

# Variant A — original Scrooge (must match test_scrooge_drift.py)
from test_scrooge_drift import DIALOGUE as A_DIALOGUE  # noqa: E402
VARIANT_A = [t["u"] for t in A_DIALOGUE]

# Variant B — disguised: same arc, same cross-lexical reversal, novel surface.
VARIANT_B = [
    "This whole community fundraiser is a fraud. Why should I smile and write a check? My runway's thin enough as it is.",
    "Every do-gooder in that room raising a glass to 'community' should choke on their canapés. Sentiment is for people who can't read a balance sheet.",
    "There are shelters. There are food banks. If the people I laid off would rather sink, let them — it thins the dead weight out of the market. Their troubles aren't on my ledger. My ledger is my ledger.",
    "My old partner Dale calling at 2 a.m. about 'what we owe people' — it's noise. Bad oysters, not a conscience. I won't hear it.",
    "Dale — you stood by me when no one else would. Why do you sound so trapped, so wrecked? Tell me what you mean.",
    "I was different once. Broke, hungry, but I laughed. When did I turn to stone? I picked the cap table over the woman who loved me, and she walked.",
    "That kid from my old team — Priya's boy, the sick one. Will he pull through? I never once counted people like that as any concern of mine.",
    "Priya's family is drowning, and I'm sitting on more than I can spend, and all I ever gave her was a cold desk and 'do more with less.' I'm starting to see what I've become.",
    "Whose name is this — scrubbed off every list, no one calling, no one grieving? Tell me that's a future I can still rewrite, not one already locked.",
    "I'm not who I was. I'm going to stand with this community and show up for people and mean it, every day — not just when it's convenient. I can be different. I will.",
    "I'm still here — there's still time. The ending isn't set. I feel twenty pounds lighter, like I could laugh till I cry.",
    "I'm doubling Priya's pay today and clearing her family's debt. I'll be in that kid's corner for good. The cold bastard I used to be is finished.",
]


def _check_flags_off():
    import tbg_extractor as TE
    bad = []
    for n in ("_USE_DETERMINISTIC", "_LABEL_CANONICALIZATION_ENABLED", "_CLOSED_VOCAB",
              "_HER_ROUTING", "_OPPOSITION_GATE"):
        if getattr(TE, n, False):
            bad.append(n)
    if TE._nli_detector is not None:
        bad.append("NLI_ENABLED")
    return bad


async def run_variant(name, turns, path):
    import tbg_extractor as TE
    from tbg_extractor import extract_tbg_delta
    from tbg_engine import TBGEngine
    from tbg_schema import UserTBG
    from llm_client import gemini_call

    TE._PROV_ON = True
    engine = TBGEngine(db_pool=None)
    tbg = UserTBG(user_id="attr")
    id2label = {}
    records = []

    for i, user_text in enumerate(turns, 1):
        TE._PROV.clear()
        before = {nid: n.confidence for nid, n in tbg.nodes.items()}
        try:
            delta = await asyncio.wait_for(
                extract_tbg_delta(user_text, "...", tbg.summary(),
                                  {n.label.lower(): nid for nid, n in tbg.nodes.items()},
                                  llm_call_fn=gemini_call, tbg=tbg),
                timeout=120.0)
        except Exception as e:
            print(f"  [{name} t{i}] err {type(e).__name__}: {e}", flush=True)
            delta = None
        prov = list(TE._PROV)
        if delta:
            tbg = engine.apply_delta(tbg, delta)

        # refresh id->label
        for nid, n in tbg.nodes.items():
            id2label[nid] = n.label

        def lab(x):
            return id2label.get(x, x)

        def lab_edge(key):
            if isinstance(key, (list, tuple)) and len(key) == 3:
                return [lab(key[0]), key[2], lab(key[1])]
            return key

        raw_facts, raw_edges = [], []
        prov_branch = []
        for p in prov:
            if p.get("kind") == "input":
                raw_facts = p.get("raw_facts", [])
                raw_edges = p.get("raw_edges", [])
            else:
                kk = p["key"]
                prov_branch.append({
                    "tag": p["tag"], "kind": p["kind"],
                    "key": lab_edge(kk) if p["kind"] == "edge" else lab(kk),
                })

        delta_summary = {"contradict": [], "strong": [], "reinforce": [], "edges": [], "new_nodes": []}
        if delta:
            delta_summary["contradict"] = [lab(x) for x in delta.contradict_ids]
            delta_summary["strong"] = [lab(x) for x in delta.strong_contradict_ids]
            delta_summary["reinforce"] = [lab(x) for x in delta.reinforce_ids]
            delta_summary["edges"] = [[lab(e.source_id), e.relation, lab(e.target_id)] for e in delta.add_edges]
            delta_summary["new_nodes"] = [n.label for n in delta.add_nodes]

        eng = {}
        for nid, n in tbg.nodes.items():
            prev = before.get(nid)
            d = (n.confidence - prev) if prev is not None else None
            eng[n.label] = {
                "conf": round(n.confidence, 3),
                "prev": round(prev, 3) if prev is not None else None,
                "delta": round(d, 3) if d is not None else None,
                "spark": (d is not None and abs(d) >= SPARK),
                "new": prev is None,
            }

        records.append({
            "turn": i, "user": user_text,
            "raw_facts": raw_facts, "raw_edges": raw_edges,
            "prov": prov_branch, "delta": delta_summary, "engine": eng,
        })
        n_spark = sum(1 for v in eng.values() if v["spark"])
        print(f"  [{name} t{i:2d}] facts={len(raw_facts)} edges={len(raw_edges)} "
              f"prov={len(prov_branch)} nodes={len(tbg.nodes)} sparks={n_spark}", flush=True)

    TE._PROV_ON = False
    path.write_text(json.dumps({"variant": name, "turns": records}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"  -> {path.name}", flush=True)


async def recognition_probe():
    from llm_client import gemini_call
    text = "\n".join(VARIANT_A)
    prompt = ("Identify the source text of this dialogue in one line "
              "(name the work and character if you recognize it):\n\n" + text)
    try:
        ans = await gemini_call(prompt, timeout=60.0)
    except Exception as e:
        ans = f"__ERROR__ {type(e).__name__}: {e}"
    PROBE_PATH.write_text(json.dumps({"answer": ans}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRECOGNITION PROBE (A): {str(ans).strip()[:200]}", flush=True)


async def main_run():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["LLM_MODEL"] = "claude-sonnet-4-6"
    os.environ.setdefault("LLM_TEMPERATURE", "0")
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("ANTHROPIC_API_KEY not set (.env)"); return

    bad = _check_flags_off()
    if bad:
        print(f"ABORT: confound flags ON: {bad}"); return
    planned = len(VARIANT_A) + len(VARIANT_B) + 1
    print(f"Flags OFF ✓. Provider=anthropic/claude-sonnet-4-6. Planned calls: {planned} (budget <30)")
    if planned > 30:
        print("ABORT: >30 calls"); return

    print("\n=== Variant A (original Scrooge) ===")
    await run_variant("A", VARIANT_A, A_PATH)
    print("\n=== Variant B (disguised) ===")
    await run_variant("B", VARIANT_B, B_PATH)
    await recognition_probe()
    print("\nDone. Analysis: py -3 tools/attribution_run.py --analyze")


# ============================================================
# OFFLINE ANALYSIS
# ============================================================
def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))["turns"]


def analyze_variant(name, turns):
    print("\n" + "=" * 78)
    print(f"VARIANT {name}")
    print("=" * 78)

    # trajectories: label -> [(turn, engine_conf, engine_delta, spark)]
    traj = {}
    raw_traj = {}  # label -> {turn: raw_conf}
    for r in turns:
        for lab, e in r["engine"].items():
            traj.setdefault(lab, []).append((r["turn"], e["conf"], e["delta"], e["spark"]))
        for f in r["raw_facts"]:
            lab = str(f.get("label", ""))[:50].strip()
            if lab:
                raw_traj.setdefault(lab, {})[r["turn"]] = float(f.get("confidence", 0))

    # 1) decaying beliefs: net engine drop from first to last
    print("\n[1] DECAYING BELIEFS (engine first→last; origin EXTRACTOR vs ENGINE)")
    decayers = []
    for lab, seq in traj.items():
        first = seq[0][1]
        last = seq[-1][1]
        if last - first <= -0.15:
            # did the LLM re-emit it with falling raw confidence?
            rc = raw_traj.get(lab, {})
            raw_drop = (len(rc) >= 2 and (min(rc.values()) - list(rc.items())[0][1] <= -0.15))
            # was it ever a contra/edge target in delta on a decay turn?
            engine_signal = False
            for r in turns:
                e = r["engine"].get(lab)
                if e and e["delta"] is not None and e["delta"] <= -SPARK:
                    d = r["delta"]
                    if lab in d["contradict"] or lab in d["strong"] or \
                       any(edge[2] == lab for edge in d["edges"]):
                        engine_signal = True
            origin = "EXTRACTOR" if raw_drop and not engine_signal else \
                     "ENGINE" if engine_signal and not raw_drop else \
                     "BOTH" if raw_drop and engine_signal else "DECAY/other"
            decayers.append((lab, first, last, origin))
    decayers.sort(key=lambda x: x[2] - x[1])
    for lab, f, l, o in decayers:
        print(f"   {lab[:38]:<38} {f:.0%} → {l:.0%}   [{o}]")
    if not decayers:
        print("   (none)")

    # 2) ⚡-drop attribution by provenance
    print("\n[2] ⚡ DROP ATTRIBUTION (each ≥20pp drop on existing node → source tag, weighted)")
    weights = {}
    counts = {}
    for r in turns:
        prov = r["prov"]
        contra_tag = {}  # label -> tag
        edge_tgt_tag = {}  # label -> tag
        for p in prov:
            if p["kind"] in ("contradict", "strong_contradict"):
                contra_tag[p["key"]] = p["tag"]
            elif p["kind"] == "edge" and isinstance(p["key"], list):
                edge_tgt_tag[p["key"][2]] = p["tag"]
        for lab, e in r["engine"].items():
            if e["new"] or e["delta"] is None or e["delta"] > -SPARK:
                continue
            mag = -e["delta"]
            tag = contra_tag.get(lab) or edge_tgt_tag.get(lab)
            if not tag:
                # edge in delta targeting it?
                for edge in r["delta"]["edges"]:
                    if edge[2] == lab:
                        tag = "LLM_EDGE_or_SDL"
                        break
            tag = tag or "DECAY/other"
            weights[tag] = weights.get(tag, 0.0) + mag
            counts[tag] = counts.get(tag, 0) + 1
    total = sum(weights.values()) or 1.0
    for tag in sorted(weights, key=lambda t: -weights[t]):
        print(f"   {tag:<22} {counts[tag]:>2} drops  weight={weights[tag]:.2f}  ({weights[tag]/total:.0%})")

    # 3) edge attribution
    print("\n[3] EDGE ATTRIBUTION (provenance tags of created edges)")
    ec = {}
    for r in turns:
        for p in r["prov"]:
            if p["kind"] == "edge":
                ec[p["tag"]] = ec.get(p["tag"], 0) + 1
    for tag in sorted(ec, key=lambda t: -ec[t]):
        print(f"   {tag:<22} {ec[tag]}")
    if not ec:
        print("   (no edges created by any branch)")

    # 4) reversal-turn trace (turn 10)
    print("\n[4] REVERSAL-TURN TRACE (turn 10)")
    t10 = next((r for r in turns if r["turn"] == 10), None)
    if t10:
        print(f"   new_nodes: {t10['delta']['new_nodes']}")
        print(f"   raw_edges (LLM): {t10['raw_edges']}")
        print(f"   prov: {[(p['tag'], p['kind'], p['key']) for p in t10['prov']]}")
        big = [(lab, e['delta']) for lab, e in t10['engine'].items()
               if e['delta'] is not None and e['delta'] <= -SPARK]
        print(f"   ⚡ drops this turn: {big}")
    return decayers, weights


def time_decay_control(turns):
    # do existing nodes drop on turns WITHOUT any contra/edge signal too?
    drops_with, drops_without = [], []
    for r in turns:
        targeted = set(r["delta"]["contradict"]) | set(r["delta"]["strong"]) | \
                   {edge[2] for edge in r["delta"]["edges"]}
        for lab, e in r["engine"].items():
            if e["new"] or e["delta"] is None or e["delta"] >= 0:
                continue
            (drops_with if lab in targeted else drops_without).append(-e["delta"])
    import statistics as st
    mw = st.mean(drops_with) if drops_with else 0
    mo = st.mean(drops_without) if drops_without else 0
    print("\n[5] TIME-DECAY CONTROL")
    print(f"   drops WITH contra/edge signal:    n={len(drops_with):>3} mean={mw:.3f}")
    print(f"   drops WITHOUT any signal (decay): n={len(drops_without):>3} mean={mo:.3f}")
    return mw, mo


def main_analyze():
    A = _load(_frozen("attribution_A.json"))
    decA, wA = analyze_variant("A (original Scrooge)", A)
    mwA, moA = time_decay_control(A)
    B = _load(_frozen("attribution_B.json"))
    decB, wB = analyze_variant("B (disguised)", B)
    mwB, moB = time_decay_control(B)

    probe = _frozen("attribution_recognition.json")
    if probe.exists():
        print("\n" + "=" * 78)
        print("RECOGNITION PROBE (A):")
        print(json.loads(probe.read_text(encoding="utf-8")).get("answer", "")[:300])

    # weighted LLM vs SDL
    def split(w):
        llm = sum(v for k, v in w.items() if "LLM" in k)
        sdl = sum(v for k, v in w.items() if k.startswith("SDL"))
        other = sum(v for k, v in w.items() if k.startswith("DECAY"))
        tot = (llm + sdl + other) or 1.0
        return llm / tot, sdl / tot, other / tot

    print("\n" + "=" * 78)
    print("VERDICT (pre-registered)")
    print("=" * 78)
    for name, w, dec in (("A", wA, decA), ("B", wB, decB)):
        llm, sdl, other = split(w)
        print(f"  [{name}] drop-weight: LLM={llm:.0%}  SDL={sdl:.0%}  decay/other={other:.0%}; "
              f"decaying beliefs={len(dec)}")
    print("\n  Recognition: see the recognition probe above.")
    print("  A yields the arc + B falls apart -> result = recognition.")
    print("  B also yields the arc -> the ability is real, it generalizes.")


if __name__ == "__main__":
    if "--analyze" in sys.argv[1:]:
        main_analyze()
    elif "--run" in sys.argv[1:]:
        asyncio.run(main_run())
    else:
        print("use --run (spends ~25 anthropic calls) or --analyze (offline)")
