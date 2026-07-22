"""
AMF Filter — Phase 2
Ambivalence Momentum Filter: per-node stability signals.

Inputs:  BeliefNode.confidence_history (already computed by engine)
Outputs: AMFState per node — amf_conf, amf_ambiv

Does NOT modify node confidence. Read-only signal layer.
"""
from dataclasses import dataclass
from typing import Dict, List
import statistics

AMF_WINDOW = 5  # last N confidence values


@dataclass
class AMFState:
    amf_conf: float   # stability: 1/(1+d+v), high = stable
    amf_ambiv: float  # volatility: v/(1+v), high = uncertain
    mu: float         # running mean
    sigma: float      # running std


def compute_node_amf(confidence_history: List[List[float]]) -> AMFState:
    """
    Compute AMF signals from node's confidence_history.
    history format: [[message_count, confidence], ...]
    """
    if len(confidence_history) < 2:
        return AMFState(amf_conf=0.5, amf_ambiv=0.0, mu=0.5, sigma=0.0)

    recent = [c for _, c in confidence_history[-AMF_WINDOW:]]
    mu = statistics.mean(recent)
    v  = statistics.variance(recent) if len(recent) >= 2 else 0.0
    current = recent[-1]
    d = abs(current - mu)

    amf_conf  = 1.0 / (1.0 + d + v)
    amf_ambiv = v / (1.0 + v)

    return AMFState(
        amf_conf=round(amf_conf, 4),
        amf_ambiv=round(amf_ambiv, 4),
        mu=round(mu, 4),
        sigma=round(v ** 0.5, 4),
    )


def compute_graph_amf(nodes: dict) -> Dict[str, AMFState]:
    """
    Compute AMF for all nodes. Returns {node_id: AMFState}.
    Called once per apply_delta cycle (after engine updates).
    """
    return {
        node_id: compute_node_amf(node.confidence_history)
        for node_id, node in nodes.items()
    }


if __name__ == "__main__":
    # Stable node: confidence converges
    stable = [[i, 0.7 + 0.01 * (i % 2)] for i in range(10)]
    s = compute_node_amf(stable)
    assert s.amf_conf > 0.8, f"stable node should have high conf: {s}"
    print(f"OK stable: conf={s.amf_conf}, ambiv={s.amf_ambiv}")

    # Volatile node: confidence oscillates.
    # With window=5 and 0.3↔0.8 swings: σ≈0.27, v≈0.075, amf_ambiv≈0.07.
    # Max achievable with {0,1} alternation is ~0.23 (sample variance cap).
    # Test checks relative signal: volatile >> stable, not an absolute threshold.
    volatile = [[i, 0.3 if i % 2 == 0 else 0.8] for i in range(10)]
    v = compute_node_amf(volatile)
    assert v.amf_ambiv > s.amf_ambiv * 10, (
        f"volatile node should have much higher ambiv than stable: "
        f"volatile={v.amf_ambiv} stable={s.amf_ambiv}"
    )
    assert v.amf_conf < s.amf_conf, f"volatile node should have lower conf: {v.amf_conf} vs {s.amf_conf}"
    print(f"OK volatile: conf={v.amf_conf}, ambiv={v.amf_ambiv}")

    print("AMF tests passed")
