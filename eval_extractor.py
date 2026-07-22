"""
Quick extractor eval — no DB required.
Tests: LLM call + SemanticDecisionLayer only.
Run: GEMINI_API_KEY=... python eval_extractor.py
"""
import asyncio
import json
import os
import sys
# Ensure UTF-8 console output on Windows cp1251 terminals (idempotent).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

CONCEPT_MATCH_THRESHOLD      = 0.55   # node matches expected concept
HALLUCINATION_FLAG_THRESHOLD = 0.72   # node too close to hallucination concept


TEST_CASES = [
    {
        "id": "quit_job_fear",
        "user_text": "I want to quit my job but I am afraid of losing stability",
        "assistant_text": "",
        "expected_categories": {"career", "fears"},
        "expected_min_nodes": 2,
        # Concepts that should NOT appear — checked via embedding similarity, not strings
        "hallucination_concepts": ["childhood trauma", "romantic relationship problems"],
    },
    {
        "id": "explicit_decision",
        "user_text": "I decided. Leaving on Friday.",
        "assistant_text": "",
        "expected_categories": {"career"},
        "expected_source": "explicit",
        "expected_min_confidence": 0.70,
    },
    {
        "id": "short_noise",
        "user_text": "Tired.",
        "assistant_text": "",
        "expected_categories": {"mood"},
        "expected_max_nodes": 2,
    },
    {
        "id": "reversal",
        "user_text": "Actually I changed my mind. Staying at work.",
        "assistant_text": "That is a big shift.",
        "expected_categories": {"career"},
        "expected_has_reversal": True,
    },
]


def _embed(labels: List[str]) -> np.ndarray:
    from fact_engine import get_embed_model
    return get_embed_model().encode(labels, normalize_embeddings=True)


def _validate(case: dict, nodes: list, delta) -> dict:
    """
    Validate extracted nodes via semantic checks, not string matching.
    Returns {ok: bool, issues: list[str]}.
    """
    issues = []
    extracted_cats = {n["category"] for n in nodes}

    # Category presence
    for cat in case.get("expected_categories", set()):
        if cat not in extracted_cats:
            issues.append(f"missing category '{cat}'")

    # Node count
    if len(nodes) < case.get("expected_min_nodes", 0):
        issues.append(f"too few nodes: {len(nodes)} < {case['expected_min_nodes']}")
    if len(nodes) > case.get("expected_max_nodes", 9999):
        issues.append(f"too many nodes: {len(nodes)} > {case['expected_max_nodes']}")

    # Source / confidence
    if "expected_source" in case:
        if not any(n.get("source") == case["expected_source"] for n in nodes):
            issues.append(f"no node with source='{case['expected_source']}'")
    if "expected_min_confidence" in case:
        if not any(n.get("confidence", 0) >= case["expected_min_confidence"] for n in nodes):
            issues.append(f"no node confidence >= {case['expected_min_confidence']}")

    # Reversal detection
    if case.get("expected_has_reversal"):
        has = delta and bool(delta.strong_contradict_ids or delta.contradict_ids)
        if not has:
            issues.append("reversal not detected")

    # Hallucination check — embedding similarity, no hardcoded labels
    hall = case.get("hallucination_concepts", [])
    if hall and nodes:
        labels = [n["label"] for n in nodes]
        try:
            embs      = _embed(labels + hall)
            node_embs = embs[:len(labels)]
            hall_embs = embs[len(labels):]
            for hi, hc in enumerate(hall):
                sims     = node_embs @ hall_embs[hi]
                best_idx = int(np.argmax(sims))
                if sims[best_idx] >= HALLUCINATION_FLAG_THRESHOLD:
                    issues.append(
                        f"hallucination: '{labels[best_idx]}' ~ '{hc}' "
                        f"(sim={sims[best_idx]:.2f})"
                    )
        except Exception as e:
            issues.append(f"hallucination check error: {e}")

    return {"ok": len(issues) == 0, "issues": issues}


async def run():
    if not GEMINI_API_KEY:
        print("Set GEMINI_API_KEY")
        return

    from tbg_schema import UserTBG
    from tbg_extractor import extract_tbg_delta

    from llm_client import gemini_call as llm
    passed = 0

    for case in TEST_CASES:
        tbg   = UserTBG(user_id="eval")
        delta = await extract_tbg_delta(
            user_text=case["user_text"],
            assistant_text=case.get("assistant_text", ""),
            existing_tbg_summary="",
            existing_label_to_uuid={},
            llm_call_fn=llm,
            tbg=tbg,
        )

        nodes = []
        if delta:
            nodes = [
                {"label": n.label, "category": n.category,
                 "confidence": round(n.confidence, 2), "source": n.source}
                for n in delta.add_nodes
            ]

        result = _validate(case, nodes, delta)
        status = "PASS" if result["ok"] else "FAIL"
        if result["ok"]:
            passed += 1

        print(f"\n[{status}] {case['id']}")
        print(f"  nodes ({len(nodes)}): {[n['label'] for n in nodes]}")
        print(f"  cats:  {[n['category'] for n in nodes]}")
        for issue in result["issues"]:
            print(f"  ! {issue}")

    print(f"\n{passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    asyncio.run(run())
