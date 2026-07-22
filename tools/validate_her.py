#!/usr/bin/env python3
"""Validation of the her_resolver integration (offline). Does not change prod, only reads."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import her_resolver
import tbg_extractor as TE
from tbg_schema import UserTBG, BeliefNode
from collections import Counter

DOMAINS = list(TE.VALID_DOMAINS)


def dom_of(runs, label):
    v = [r["domain"] for r in runs.get(label, []) if isinstance(r, dict) and r.get("domain") in DOMAINS]
    return Counter(v).most_common(1)[0][0] if v else None


def shim(d):
    return type("N", (), {"domain": d})()


print("=" * 72)
print("1) CALIB GATE LIST — her_resolver.gate (домены = мода routing_runs.json)")
print("=" * 72)
runs = json.load(open(os.path.join(os.path.dirname(__file__), "routing_runs.json"), encoding="utf-8"))
checks = [
    ("career security", "financial security", "BLOCK_MERGE"),   # catastrophe career/money
    ("mortgage payments", "doctor partner", "BLOCK_MERGE"),      # catastrophe money/relationships
    ("financial security", "financial insecurity", "ALLOW"),    # one domain (money)
    ("wants stability", "wants change", "ALLOW"),                # one domain (self) — this is EPA's job, not routing
]
ok = True
for a, b, expect in checks:
    da, db = dom_of(runs, a), dom_of(runs, b)
    got = her_resolver.gate(shim(da), shim(db))
    mark = "OK" if got == expect else "FAIL"
    ok = ok and got == expect
    print(f"  [{mark}] {a!r}[{da}] vs {b!r}[{db}] -> {got} (ждали {expect})")
print(f"  ИТОГ calib-гейт: {'ВСЕ ВЕРНО' if ok else 'ЕСТЬ РАСХОЖДЕНИЯ'}")


def run_resolve(facts, existing_nodes):
    tbg = UserTBG(user_id="v")
    for spec in existing_nodes:
        tbg.set_node(BeliefNode(**spec))
    return TE._sdl.resolve(facts, "Actually now I feel different.", tbg, [])


print("\n" + "=" * 72)
print("2) ФУНКЦИОНАЛЬНО — VETO понижает MERGE -> NEW (флаги тогглим в рантайме)")
print("=" * 72)

# (a) HER_ROUTING: burnout[health] vs existing job burnout[career], cos~0.84 -> MERGE
ex_her = [{"label": "job burnout", "category": "career", "confidence": 0.6, "domain": "career"}]
fact_her = [{"label": "burnout", "category": "career", "confidence": 0.7, "source": "explicit", "domain": "health"}]

TE._HER_ROUTING = False; TE._OPPOSITION_GATE = False
d_off = run_resolve(fact_her, ex_her)
TE._HER_ROUTING = True
d_on = run_resolve(fact_her, ex_her)
TE._HER_ROUTING = False
print("HER_ROUTING (burnout[health] vs job burnout[career], cos~0.84):")
print(f"  OFF -> reinforce={len(d_off.reinforce_ids)} add_nodes={len(d_off.add_nodes)} "
      f"({'MERGE/reinforce' if d_off.reinforce_ids else 'new'})")
print(f"  ON  -> reinforce={len(d_on.reinforce_ids)} add_nodes={len(d_on.add_nodes)} "
      f"({'NEW (VETO сработал)' if d_on.add_nodes and not d_on.reinforce_ids else '???'})")

# (b) OPPOSITION_GATE: career instability[-] vs existing career stability[+], cos~0.89 same domain
ex_opp = [{"label": "career stability", "category": "career", "confidence": 0.6, "domain": "career"}]
fact_opp = [{"label": "career instability", "category": "career", "confidence": 0.7, "source": "explicit", "domain": "career"}]
TE._HER_ROUTING = False; TE._OPPOSITION_GATE = False
o_off = run_resolve(fact_opp, ex_opp)
TE._OPPOSITION_GATE = True
o_on = run_resolve(fact_opp, ex_opp)
TE._OPPOSITION_GATE = False
ep = TE._eval_prod("career instability", "career stability")
print(f"\nOPPOSITION_GATE (career instability vs career stability, eval_prod={ep:+.4f}, same domain):")
print(f"  OFF -> reinforce={len(o_off.reinforce_ids)} contradict={len(o_off.contradict_ids)} add_nodes={len(o_off.add_nodes)}")
print(f"  ON  -> reinforce={len(o_on.reinforce_ids)} contradict={len(o_on.contradict_ids)} add_nodes={len(o_on.add_nodes)} "
      f"({'NEW (оппозиция -> не merge)' if len(o_on.add_nodes) > len(o_off.add_nodes) or (o_on.add_nodes and not o_on.reinforce_ids) else 'без изменения'})")

print("\nПрод не трогался (флаги тогглились только в памяти теста). run_drift не гонялся.")
