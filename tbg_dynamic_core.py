import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class NodeState:
    id: str
    label: str
    confidence: float = 1.0  # s_i in Ising model [-1, 1]
    activation_count: int = 0
    last_seen_turn: int = 0

class DynamicBeliefGraph:
    """
    TBG Core v2: Dynamic weights, Hebbian learning, and Energy-based surprise detection.
    Designed to minimize LLM calls by handling dynamics through math.
    """
    def __init__(self, 
                 decay_rate: float = 0.995, 
                 learning_rate: float = 0.05,
                 surprise_threshold: float = 1.5):
        self.nodes: Dict[str, NodeState] = {}
        self.weights: Dict[Tuple[str, str], float] = {}  # W_ij matrix
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.surprise_threshold = surprise_threshold
        self.turn_count = 0

    def add_node(self, node_id: str, label: str, initial_conf: float = 0.5):
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeState(id=node_id, label=label, confidence=initial_conf)
            self.nodes[node_id].last_seen_turn = self.turn_count

    def update_weights_hebbian(self, active_nodes: List[str]):
        """
        Hebbian Learning: 'Cells that fire together, wire together'.
        Increases W_ij for nodes present in the same context window.
        """
        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                n1, n2 = sorted([active_nodes[i], active_nodes[j]])
                pair = (n1, n2)
                
                # Update existing or init with semantic prior (e.g. 0.1)
                current_w = self.weights.get(pair, 0.1)
                self.weights[pair] = current_w + self.learning_rate
                
                # Update metadata
                if n1 in self.nodes: self.nodes[n1].activation_count += 1
                if n2 in self.nodes: self.nodes[n2].activation_count += 1

    def apply_decay(self):
        """
        Exponential Decay: Weakens unused connections over time.
        """
        for pair in list(self.weights.keys()):
            self.weights[pair] *= self.decay_rate
            if self.weights[pair] < 0.01:
                del self.weights[pair]

    def calculate_system_energy(self) -> float:
        """
        Calculates the Ising Hamiltonian: H = - sum(W_ij * s_i * s_j)
        Lower energy = higher consistency/stability of the belief system.
        """
        energy = 0.0
        for (id1, id2), w in self.weights.items():
            if id1 in self.nodes and id2 in self.nodes:
                s1 = self.nodes[id1].confidence
                s2 = self.nodes[id2].confidence
                energy -= w * s1 * s2
        return energy

    def detect_surprise(self, new_nodes: List[str], new_confidences: Dict[str, float]) -> Tuple[bool, float]:
        """
        Surprise Detection (Heuristic for LLM Trigger):
        Measures the delta in energy if new beliefs are integrated.
        If Delta E > threshold, the system 'is confused' and needs LLM audit.
        """
        old_energy = self.calculate_system_energy()
        
        # Temporary projection
        temp_energy = old_energy
        for n_id in new_nodes:
            # Estimate impact based on existing neighbors
            s_new = new_confidences.get(n_id, 0.5)
            for existing_id, node in self.nodes.items():
                pair = tuple(sorted([n_id, existing_id]))
                if pair in self.weights:
                    temp_energy -= self.weights[pair] * s_new * node.confidence
        
        delta_e = abs(temp_energy - old_energy)
        needs_llm = delta_e > self.surprise_threshold
        
        return needs_llm, delta_e

    def step(self):
        self.turn_count += 1
        self.apply_decay()

# --- Example Usage ---
if __name__ == "__main__":
    graph = DynamicBeliefGraph()
    print("TBG v2 Dynamic Core Simulation Initialized.")
    
    # 1. Setup initial state
    graph.add_node("work", "Career and job stability", initial_conf=0.8)
    graph.add_node("money", "Financial security", initial_conf=0.9)
    
    # 2. Simulate co-occurrence (Hebbian)
    graph.update_weights_hebbian(["work", "money"])
    print(f"[Turn 1] Weight (work-money) after Hebb: {graph.weights[('money', 'work')]:.4f}")
    
    # 3. Calculate stability
    print(f"[Turn 1] System Energy: {graph.calculate_system_energy():.4f}")
    
    # 4. Detect surprise for a contradictory node
    new_nodes = ["burnout"]
    new_confs = {"burnout": -0.9} 
    
    # Inject a negative connection (learned from NLI or manual)
    graph.weights[tuple(sorted(["work", "burnout"]))] = -0.8 
    
    needs_llm, delta = graph.detect_surprise(new_nodes, new_confs)
    print(f"[Turn 2] Surprise Delta E: {delta:.4f}")
    print(f"[Turn 2] Trigger LLM audit? {'YES' if needs_llm else 'NO'}")
    
    graph.step()
    print(f"[Turn 3] Weight after decay: {graph.weights.get(('money', 'work'), 0):.4f}")
