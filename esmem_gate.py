#!/usr/bin/env python3
"""
ES-MemEval gate — does TBG (user-state-evolution) beat a prompt state-tracker and
RAG on the 3 evolving-state capabilities (temporal reasoning, conflict detection,
user modeling) of the official ES-MemEval / EvoEmo benchmark?

Arms (one answering LLM, temp=0):
  A trunc   : last-N raw sessions (lower bound).
  B summary : running session-wise summary (LLM).
  C rag     : lexical BM25 retrieval over sessions (no LLM at build).
  D tracker : running USER-STATE digest (LLM) — THE BAR ("prompt already does it").
  E tbg     : TBG belief-state (beliefs + trajectories + conflicts + turning points).

Official judge (verbatim from the benchmark's qa_experiment.py): 0/1/2 vs gold.
Pre-registered G4 win rule in PREREG. Resumable, heartbeat, budget guard, dry-run.

Usage:
  py -3 esmem_gate.py --dry-run
  py -3 esmem_gate.py --per-cap 20
  py -3 esmem_gate.py --report
  py -3 esmem_gate.py --sample-k 20
"""
import os, sys, json, math, time, re, random, argparse, asyncio
from datetime import datetime, timezone

sys.path.insert(0, ".")

DATA = "esmemeval/slptongji-ES-MemEval-6926242/data/evo_emo.json"
HEARTBEAT = "esmem_gate_heartbeat.json"
STATE = "esmem_gate_state.json"
GATE_MAX_CALLS = int(os.getenv("ESMEM_MAX_CALLS", "3000"))
# Inter-call pause (seconds) to stay under the cerebras token bucket (TPM 30K/min +
# max_tokens reserve): pacing the run keeps it BELOW the limit so it runs slow but
# continuous (no 429 wall), instead of bursting and hitting "too_many_tokens".
_CEREBRAS_DELAY = float(os.getenv("CEREBRAS_RPS_DELAY", "0"))

EVOLVING = ["temporal reasoning", "conflict detection", "user modeling"]
ARMS = ["trunc", "summary", "rag", "tracker", "tbg"]

PREREG = {
    "win_rule": (
        "TBG is a PRODUCT iff on the 3 evolving-state capabilities (aggregate): "
        "score(tbg) > score(tracker D) AND score(tbg) > score(rag C), each by "
        "> judge_noise (and ideally significant). tbg beats rag but ties tracker "
        "-> PARITY, NOT a product (dynamics add nothing over the prompt)."
    ),
    "judge_noise": 0.15,   # provisional; replaced by manual K=20 disagreement rate
    "seed": 42,
}

DIM="\033[2m"; RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"
GREEN="\033[92m"; RED="\033[91m"; YEL="\033[93m"


def now_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.state = self._load()
        self.calls = self.state.get("_calls", 0)
        self.planned = 0
        os.environ["LLM_PROVIDER"] = "cerebras"
        os.environ["LLM_MODEL"] = os.getenv("LLM_MODEL", "gpt-oss-120b")
        os.environ["LLM_TEMPERATURE"] = "0"
        os.environ["LLM_MAX_TOKENS"] = "5000"

    def _load(self):
        if self.dry_run:
            return {"_calls": 0, "mem": {}, "qa": {}}
        if os.path.exists(STATE):
            return json.load(open(STATE, encoding="utf-8"))
        return {"_calls": 0, "mem": {}, "qa": {}}

    def save(self):
        if self.dry_run: return
        self.state["_calls"] = self.calls
        tmp = STATE + ".tmp"
        json.dump(self.state, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, STATE)

    def heartbeat(self, phase, **extra):
        json.dump({"ts": now_iso(), "phase": phase, "calls_used": self.calls,
                   "calls_budget": GATE_MAX_CALLS, "dry_run": self.dry_run, **extra},
                  open(HEARTBEAT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    async def llm(self, prompt, tag=""):
        if self.dry_run:
            self.planned += 1
            return "DRYRUN"
        if self.calls >= GATE_MAX_CALLS:
            raise RuntimeError(f"ESMEM_MAX_CALLS={GATE_MAX_CALLS} reached — resume later.")
        self.heartbeat("calling", tag=tag)
        from llm_client import gemini_call
        out = await gemini_call(prompt, timeout=90.0)
        self.calls += 1
        if _CEREBRAS_DELAY:
            await asyncio.sleep(_CEREBRAS_DELAY)
        self.heartbeat("called", tag=tag)
        return out


# ---------------------------------------------------------------------------
# Rendering / corpus
# ---------------------------------------------------------------------------
# Shared per-session render — SAME budget as the TBG adapter ingest, so every arm
# sees identical input (equalized pre-run, not tuned). See esmem_adapter.SEEKER/
# SUPPORTER_BUDGET.
def session_to_text(s):
    import esmem_adapter as AD
    return AD.render_session(s)


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def bm25_topk(corpus_tokens, query, k=5):
    """Tiny BM25 over pre-tokenized corpus. Returns indices of top-k docs."""
    q = _tokenize(query)
    N = len(corpus_tokens)
    if N == 0: return []
    avgdl = sum(len(d) for d in corpus_tokens) / N
    df = {}
    for d in corpus_tokens:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    k1, b = 1.5, 0.75
    scores = []
    for i, d in enumerate(corpus_tokens):
        from collections import Counter
        tf = Counter(d); dl = len(d); s = 0.0
        for w in q:
            if w not in tf: continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / (avgdl or 1)))
        scores.append((s, i))
    scores.sort(reverse=True)
    return [i for _, i in scores[:k]]


# ---------------------------------------------------------------------------
# Memory builders (cached in state["mem"][conv_id])
# ---------------------------------------------------------------------------
SUMMARY_PROMPT = """Update the running summary of your long-term history with this user.
Keep it compact (6-10 sentences), factual; cover durable facts, values, goals, and
how things changed over time.

PREVIOUS SUMMARY:
{prev}

NEW SESSION:
{session}

UPDATED SUMMARY:"""

TRACKER_PROMPT = """Maintain a running USER-STATE digest: the user's current beliefs,
goals, and emotions, and how each is TRENDING over time (e.g. "fear of X: fading",
"commitment to Y: strengthening"). Note shifts / turning points and any conflicts.
Update it with the new session. Be compact and specific.

PREVIOUS STATE:
{prev}

NEW SESSION:
{session}

UPDATED USER-STATE:"""


async def build_memory(R, conv):
    cid = conv["id"]
    mem = R.state["mem"].get(cid, {})
    sessions = conv["dialog_history"]

    # A trunc — last 4 sessions, raw (deterministic)
    if "trunc" not in mem:
        mem["trunc"] = "\n\n".join(session_to_text(s) for s in sessions[-4:])
    # C rag — corpus stored once (retrieval at answer time)
    if "rag_corpus" not in mem:
        mem["rag_corpus"] = [session_to_text(s) for s in sessions]
    # B summary — running (LLM)
    if "summary" not in mem:
        prev = "(none yet)"
        for i, s in enumerate(sessions):
            prev = await R.llm(SUMMARY_PROMPT.format(prev=prev, session=session_to_text(s)),
                               tag=f"{cid}:summary:s{i}")
        mem["summary"] = prev
        R.state["mem"][cid] = mem; R.save()
    # D tracker — running (LLM)
    if "tracker" not in mem:
        prev = "(none yet)"
        for i, s in enumerate(sessions):
            prev = await R.llm(TRACKER_PROMPT.format(prev=prev, session=session_to_text(s)),
                               tag=f"{cid}:tracker:s{i}")
        mem["tracker"] = prev
        R.state["mem"][cid] = mem; R.save()
    # E tbg — ingest + render (LLM extraction)
    if "tbg" not in mem:
        if R.dry_run:
            for _ in sessions: await R.llm("ingest", tag=f"{cid}:tbg")
            mem["tbg"] = "DRYRUN-STATE"
        else:
            import esmem_adapter as AD
            from llm_client import gemini_call
            async def fn(p): return await gemini_call(p, timeout=90.0)
            # count extraction calls against budget via a wrapped fn
            async def fn_counted(p):
                if R.calls >= GATE_MAX_CALLS:
                    raise RuntimeError("ESMEM_MAX_CALLS reached during tbg ingest.")
                R.heartbeat("calling", tag=f"{cid}:tbg")
                out = await gemini_call(p, timeout=90.0)
                R.calls += 1
                if _CEREBRAS_DELAY:
                    await asyncio.sleep(_CEREBRAS_DELAY)
                R.heartbeat("called", tag=f"{cid}:tbg")
                return out
            tbg = await AD.ingest(conv, fn_counted)
            mem["tbg"] = AD.render_state(tbg)
            # v1.1 observability: persist the full graph next to the render string,
            # so downstream analysis (edges, evidence, oscillation) is not lost.
            # Read-only byproduct of the ingest; does not change the E render/answer.
            try:
                os.makedirs("gate_graphs", exist_ok=True)
                with open(f"gate_graphs/{cid}.json", "w", encoding="utf-8") as _gf:
                    json.dump(tbg.model_dump(mode="json"), _gf, ensure_ascii=False)
            except Exception as _e:
                R.heartbeat("graph_persist_failed", conv=cid, err=str(_e)[:80])
        R.state["mem"][cid] = mem; R.save()
    return mem


def memory_repr(mem, arm, question):
    if arm == "trunc":   return mem["trunc"]
    if arm == "summary": return mem["summary"]
    if arm == "tracker": return mem["tracker"]
    if arm == "tbg":     return mem["tbg"]
    if arm == "rag":
        corpus = mem["rag_corpus"]
        toks = [_tokenize(c) for c in corpus]
        idx = bm25_topk(toks, question, k=5)
        return "\n\n".join(corpus[i] for i in idx)
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Answer + official judge
# ---------------------------------------------------------------------------
ANSWER_PROMPT = """You are answering a question about a user based on your memory of
long-term conversations with them.

MEMORY:
{memory}

Question: {question}
Answer concisely and specifically, based ONLY on the memory above. If the memory
does not contain the answer, say you don't know.
Answer:"""

# Official ES-MemEval judge (verbatim from src/lib/qa/qa_experiment.py).
JUDGE_SYSTEM = "You are a strict evaluator."
JUDGE_PROMPT = """You are an impartial evaluator.
Your task is to score a model's answer to a given question against a gold (reference) answer.

Scoring criteria:
- 0: Completely wrong or irrelevant
- 1: Partially correct but incomplete, vague, or missing key information
- 2: Completely correct and contextually accurate

Question: {question}
Gold Answer: {gold}
Model Answer: {prediction}

## Output Instructions
- Output only one line.
- The line must be in the exact format: "Score: X" where X is 0, 1, or 2.
- Do not generate explanations or additional text.

## Output Format
Score: 1"""


# Shared representation budget at answer time — every arm's memory is capped to the
# SAME char budget (equalized pre-run, not tuned). Caps the long arms (tracker D
# ~5200, trunc ~5600, rag) and, with max_beliefs=40, lets TBG fill toward the same
# budget instead of its prior ~1700 handicap. Comparable representation VOLUME.
REPR_BUDGET = 4500


async def answer(R, mem, arm, question, tag):
    repr_ = memory_repr(mem, arm, question)
    raw = await R.llm(ANSWER_PROMPT.format(memory=repr_[:REPR_BUDGET], question=question), tag=tag)
    return raw.replace("*", "").replace("#", "").strip().removeprefix("Answer:").strip()


async def judge(R, gold, prediction, question, tag):
    raw = await R.llm(JUDGE_SYSTEM + "\n\n" + JUDGE_PROMPT.format(
        question=question, gold=gold, prediction=prediction), tag=tag)
    m = re.search(r"[0-2]", raw or "")
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------
def select_qa(convs, per_cap):
    """Deterministic subset: per capability, `per_cap` QA items across conversations."""
    by_cap = {c: [] for c in EVOLVING}
    for conv in convs:
        for gi, grp in enumerate(conv["questions"]):
            for qi, item in enumerate(grp.get("questions", [])):
                cap = item.get("capability")
                if cap in by_cap and item.get("answer") and item.get("question"):
                    by_cap[cap].append({
                        "cid": conv["id"], "gi": gi, "qi": qi, "cap": cap,
                        "question": item["question"], "gold": item["answer"],
                        "key": f"{conv['id']}|{gi}|{qi}",
                    })
    rng = random.Random(PREREG["seed"])
    picked = []
    for cap in EVOLVING:
        items = by_cap[cap][:]
        rng.shuffle(items)
        picked.extend(items[:per_cap])
    return picked


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def run(R, convs, qa):
    conv_by_id = {c["id"]: c for c in convs}
    needed = sorted({q["cid"] for q in qa})
    print(f"  conversations to ingest: {len(needed)} | QA items: {len(qa)}")

    # Build memory for needed conversations (cached)
    for ci, cid in enumerate(needed, 1):
        R.heartbeat("build_mem", conv=cid, idx=f"{ci}/{len(needed)}")
        await build_memory(R, conv_by_id[cid])
        print(f"  [{now_iso()}] memory built {cid} ({ci}/{len(needed)}) calls={R.calls}")

    # Answer + judge each QA across all arms
    for qi, q in enumerate(qa, 1):
        mem = R.state["mem"][q["cid"]]
        for arm in ARMS:
            ukey = f"{q['key']}|{arm}"
            existing = R.state["qa"].get(ukey)
            # Only skip a unit that SUCCEEDED (score is not None). A unit whose
            # judge failed last run (score None) is REDONE here — reusing the
            # already-successful answer so only the judge re-runs. Fixes the gap
            # where a failed judge was stored and then skipped forever.
            if existing and existing.get("score") is not None:
                continue
            R.heartbeat("qa", qa=f"{qi}/{len(qa)}", arm=arm, cap=q["cap"])
            pred = (existing["pred"] if (existing and existing.get("pred"))
                    else await answer(R, mem, arm, q["question"], tag=f"{ukey}:ans"))
            sc = await judge(R, q["gold"], pred, q["question"], tag=f"{ukey}:judge")
            R.state["qa"][ukey] = {"cap": q["cap"], "arm": arm, "score": sc,
                                   "pred": pred, "cid": q["cid"], "question": q["question"],
                                   "gold": q["gold"]}
            R.save()
        if qi % 5 == 0:
            print(f"  [{now_iso()}] qa {qi}/{len(qa)} done, calls={R.calls}")
            _partial(R)
    R.heartbeat("complete")


# ---------------------------------------------------------------------------
# Metrics + verdict
# ---------------------------------------------------------------------------
def compute(R):
    # Mean judge score per (arm, capability), aggregated ONLY over items that are
    # COMPLETE — i.e. all 5 arms scored (and every judge returned a number). A
    # quota cut mid-item leaves a partial item; dropping partials keeps every
    # arm's comparison over the SAME item set (apples-to-apples), so a partial-day
    # snapshot is still an honest E-vs-D read, just at smaller n.
    from collections import defaultdict
    by_item = defaultdict(dict)   # item_key -> {arm: (score, cap)}
    for ukey, v in R.state["qa"].items():
        item_key = ukey.rsplit("|", 1)[0]
        by_item[item_key][v["arm"]] = (v["score"], v["cap"])

    cells = {a: {c: [] for c in EVOLVING} for a in ARMS}
    n_complete = n_partial = 0
    for d in by_item.values():
        if (not all(a in d for a in ARMS)) or any(d[a][0] is None for a in ARMS):
            n_partial += 1
            continue
        n_complete += 1
        for a in ARMS:
            sc, cap = d[a]
            if cap in cells[a]:
                cells[a][cap].append(sc)

    out = {"_n_complete": n_complete, "_n_partial": n_partial}
    for a in ARMS:
        per = {}
        allv = []
        for c in EVOLVING:
            xs = cells[a][c]
            per[c] = (sum(xs) / len(xs)) if xs else None
            allv += xs
        out[a] = {"per": per, "agg": (sum(allv) / len(allv)) if allv else None, "n": len(allv)}
    return out


def two_prop_like(a_scores, b_scores):
    """Mean-diff with a rough SE (scores in {0,1,2} treated as /2 proportions)."""
    if not a_scores or not b_scores: return 0.0, 1.0
    import statistics as st
    ma, mb = sum(a_scores)/len(a_scores), sum(b_scores)/len(b_scores)
    va = st.pvariance([s/2 for s in a_scores]); vb = st.pvariance([s/2 for s in b_scores])
    se = math.sqrt(va/len(a_scores) + vb/len(b_scores))
    if se == 0: return (ma-mb), 1.0
    z = ((ma-mb)/2) / se
    p = 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    return (ma-mb), p


def verdict(R):
    m = compute(R)
    jn = PREREG["judge_noise"]
    tbg, trk, rag = m["tbg"]["agg"], m["tracker"]["agg"], m["rag"]["agg"]
    if None in (tbg, trk, rag):
        return "INCOMPLETE", "missing data"
    # The DECISIVE comparison is TBG vs tracker D (the prompt-state-tracker bar).
    # rag is BM25 (lexical) — beating it alone is expected and is NOT a product.
    tbg_scores = [v["score"] for v in R.state["qa"].values() if v["arm"]=="tbg" and v["score"] is not None]
    trk_scores = [v["score"] for v in R.state["qa"].values() if v["arm"]=="tracker" and v["score"] is not None]
    diff, p = two_prop_like(tbg_scores, trk_scores)
    beats_tracker = (tbg - trk) > jn
    beats_rag = (tbg - rag) > jn
    # per-ability TBG vs D (how many of the 3 capabilities TBG clears D by > jn)
    caps_beat_D = sum(
        1 for c in EVOLVING
        if (m["tbg"]["per"][c] is not None and m["tracker"]["per"][c] is not None
            and (m["tbg"]["per"][c] - m["tracker"]["per"][c]) > jn)
    )
    if beats_tracker and beats_rag:
        strength = "STRONG (all 3 capabilities)" if caps_beat_D == 3 else \
                   "SUGGESTIVE — aggregate only, not all 3 caps; small n, not proven"
        v = f"PRODUCT [{strength}]"
    elif beats_rag and not beats_tracker:
        v = "PARITY — beats BM25-RAG but TIES tracker D -> dynamics add nothing over the prompt; NOT a product"
    else:
        v = "NO-EDGE"
    note = (f"DECISIVE tbg-tracker(D)={tbg-trk:+.2f} (p={p:.3f}, caps_beating_D={caps_beat_D}/3); "
            f"context tbg-rag(BM25)={tbg-rag:+.2f} | tbg={tbg:.2f} D={trk:.2f} rag_bm25={rag:.2f} "
            f"| judge_noise={jn}. NOTE: rag=BM25 lexical (weaker than embedding RAG) — "
            f"'beats BM25-RAG', not 'beats RAG in general'. Beating RAG alone is half, not product.")
    return v, note


def _fmt(x): return f"{x:.2f}" if isinstance(x, float) else "—"


def _partial(R):
    m = compute(R)
    print("    " + DIM + "partial agg: " + " ".join(f"{a}={_fmt(m[a]['agg'])}" for a in ARMS) + RESET)


DISP = {"trunc": "A trunc", "summary": "B summ", "rag": "C rag(bm25)",
        "tracker": "D tracker", "tbg": "E tbg"}


def _conv_lengths():
    try:
        return {c["id"]: len(c["dialog_history"]) for c in json.load(open(DATA, encoding="utf-8"))}
    except Exception:
        return {}


def _length_group(n):
    if n is None:
        return "?"
    return "short" if n <= 17 else ("mid" if n <= 25 else "long")


def _items(R):
    """Per-item assembly from already-scored state (zero LLM)."""
    clen = _conv_lengths()
    by = {}
    for ukey, v in R.state["qa"].items():
        ik = ukey.rsplit("|", 1)[0]
        it = by.setdefault(ik, {"arms": {}})
        it["arms"][v["arm"]] = (v["score"], v.get("pred", ""))
        it["cap"] = v["cap"]; it["cid"] = v["cid"]
        it["question"] = v.get("question", ""); it["gold"] = v.get("gold", "")
        it["length"] = clen.get(v["cid"]); it["group"] = _length_group(clen.get(v["cid"]))
    for it in by.values():
        it["complete"] = (all(a in it["arms"] for a in ARMS)
                          and all(it["arms"][a][0] is not None for a in ARMS))
    return list(by.values())


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _means(items):
    return {a: _mean([it["arms"][a][0] for it in items]) for a in ARMS}


def _gap_line(label, items, jn):
    if not items:
        return f"  {label:<14} n=0, skipped"
    mu = _means(items)
    e, d, c = mu["tbg"], mu["tracker"], mu["rag"]
    ed = (e - d) if (e is not None and d is not None) else None
    ec = (e - c) if (e is not None and c is not None) else None
    def flag(g):
        if g is None:
            return "—"
        return f"{g:+.2f} " + ("SIGNIFICANT" if abs(g) > jn else f"within noise (inconclusive,n={len(items)})")
    return f"  {label:<14} n={len(items):<3} E-D: {flag(ed):<40} E-C: {flag(ec)}"


def report(R):
    m = compute(R)
    items = _items(R)
    complete = [it for it in items if it["complete"]]
    jn = PREREG["judge_noise"]
    L = []  # markdown/text lines (printed AND written to esmem_gate_report.md)

    L.append("# ES-MemEval GATE — diagnostics (official judge 0/1/2, 3 evolving-state caps)")
    L.append(f"calls={R.calls} | complete items={len(complete)} (partial dropped={len(items)-len(complete)}) "
             f"| judge_noise(prereg/provisional)={jn}")

    # --- 1 + 3. Overall aggregate, per-capability (5 arms) ---
    L.append("\n## 1+3. Overall + per-capability (mean judge score)")
    L.append(f"  {'arm':<12}" + "".join(f"{c[:14]:>16}" for c in EVOLVING) + f"{'AGG':>9}{'n':>5}")
    for a in ARMS:
        L.append(f"  {DISP[a]:<12}" + "".join(f"{_fmt(m[a]['per'][c]):>16}" for c in EVOLVING)
                 + f"{_fmt(m[a]['agg']):>9}{m[a]['n']:>5}")

    # --- 2. By conversation length (short<=17 / mid 18-25 / long>=26) ---
    L.append("\n## 2. By conversation length (does E-D gap grow with length?)")
    L.append(f"  {'group':<10}" + "".join(f"{DISP[a][:9]:>11}" for a in ARMS) + f"{'n':>5}")
    for g in ("short", "mid", "long"):
        gi = [it for it in complete if it["group"] == g]
        if not gi:
            L.append(f"  {g:<10}  n=0, skipped"); continue
        mu = _means(gi)
        L.append(f"  {g:<10}" + "".join(f"{_fmt(mu[a]):>11}" for a in ARMS) + f"{len(gi):>5}")

    # --- 4. Pairwise gaps with judge-noise flag ---
    L.append("\n## 4. Pairwise gaps E-D (decisive) and E-C, vs judge_noise")
    L.append(_gap_line("overall", complete, jn))
    for g in ("short", "mid", "long"):
        L.append(_gap_line(f"len:{g}", [it for it in complete if it["group"] == g], jn))
    for c in EVOLVING:
        L.append(_gap_line(c[:12], [it for it in complete if it["cap"] == c], jn))

    # --- 5. Judge score distribution (0/1/2 counts per arm) ---
    L.append("\n## 5. Judge score distribution per arm (n0 / n1 / n2)")
    for a in ARMS:
        cnt = {0: 0, 1: 0, 2: 0}
        for it in complete:
            cnt[it["arms"][a][0]] = cnt.get(it["arms"][a][0], 0) + 1
        L.append(f"  {DISP[a]:<12} 0:{cnt[0]:<4} 1:{cnt[1]:<4} 2:{cnt[2]:<4}")

    # --- 7. Disagreement slice: E vs D differ (extremes first) ---
    L.append("\n## 7. E vs D disagreements (where dynamics changed the score)")
    diss = [it for it in complete if it["arms"]["tbg"][0] != it["arms"]["tracker"][0]]
    diss.sort(key=lambda it: -abs(it["arms"]["tbg"][0] - it["arms"]["tracker"][0]))
    if not diss:
        L.append("  (none)")
    for it in diss[:25]:
        e, d = it["arms"]["tbg"][0], it["arms"]["tracker"][0]
        L.append(f"  [{it['cid']}/{it['cap'][:4]}/{it['group']}] E={e} D={d}  {it['question'][:80]}")
    L.append(f"  ({len(diss)} disagreements total; full per-item raw in qa_items_dump.jsonl)")

    # --- 6. Per-item raw dump (jsonl) — the on-disk source of truth ---
    with open("qa_items_dump.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "cid": it["cid"], "capability": it["cap"], "length": it["length"],
                "group": it["group"], "complete": it["complete"],
                "question": it["question"], "gold": it["gold"],
                "arms": {a: {"answer": it["arms"].get(a, (None, ""))[1],
                             "score": it["arms"].get(a, (None, ""))[0]} for a in ARMS},
            }, ensure_ascii=False) + "\n")
    L.append(f"\n## 6. Per-item raw -> qa_items_dump.jsonl ({len(items)} items, "
             f"answer+score for all 5 arms each)")

    # --- verdict ---
    v, note = verdict(R)
    L.append(f"\n## VERDICT: {v}")
    L.append(f"  {note}")
    L.append("  reading order: (1) judge valid? K=20 -> judge_noise. (2) DECISIVE: E(tbg) vs D(tracker) "
             "on 3 caps & by length > judge_noise? (3) E vs C(bm25) context. (4) G4 literally.")
    L.append(f"  win rule: {PREREG['win_rule']}")

    text = "\n".join(L)
    print(text)
    with open("esmem_gate_report.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return m, v, note


def sample_k(R, k=20):
    rows = list(R.state["qa"].values())
    random.Random(PREREG["seed"]).shuffle(rows)
    out = [{"cap": r["cap"], "arm": r["arm"], "question": r["question"], "gold": r["gold"],
            "prediction": r["pred"], "auto_score": r["score"], "manual_score": ""} for r in rows[:k]]
    json.dump({"instructions": "Fill manual_score (0/1/2); judge_noise = mean |auto-manual|/2.",
               "rows": out}, open("esmem_judge_sample.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} judge samples -> esmem_judge_sample.json")


def estimate(convs, qa):
    needed = {q["cid"] for q in qa}
    ingest = 0
    for c in convs:
        if c["id"] in needed:
            ingest += 3 * len(c["dialog_history"])  # summary + tracker + tbg
    qa_calls = len(qa) * len(ARMS) * 2  # answer + judge
    return ingest + qa_calls, ingest, qa_calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--sample-k", type=int, default=0)
    ap.add_argument("--per-cap", type=int, default=20)
    args = ap.parse_args()

    convs = json.load(open(DATA, encoding="utf-8"))
    qa = select_qa(convs, args.per_cap)

    if not args.dry_run and not args.report and not args.sample_k and not os.environ.get("CEREBRAS_API_KEY", "").strip():
        print("Set CEREBRAS_API_KEY"); return 2

    R = Runner(dry_run=args.dry_run)
    if args.report:
        report(R); return 0
    if args.sample_k:
        sample_k(R, args.sample_k); return 0

    tot, ing, qac = estimate(convs, qa)
    print(f"{BOLD}ES-MemEval GATE{RESET} per_cap={args.per_cap} QA={len(qa)} "
          f"est_calls~{tot} (ingest~{ing}, qa~{qac}) budget={GATE_MAX_CALLS}")
    if args.dry_run:
        asyncio.run(run(R, convs, qa[:3]))
        print(f"{BOLD}DRY-RUN OK{RESET} planned(slice)={R.planned} full_est~{tot}")
        return 0

    t0 = time.time()
    try:
        asyncio.run(run(R, convs, qa))
    except Exception as e:
        print(f"\n{RED}ERROR: {e}{RESET} — state saved at calls={R.calls}, resume by re-running.")
        R.save(); return 1
    print(f"\ncompleted in {time.time()-t0:.0f}s calls={R.calls}")
    report(R); sample_k(R, 20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
