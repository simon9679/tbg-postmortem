import numpy as np

class TrueIsingGraph:
    """
    A true Ising model (Energy-based Model).
    No heuristics. Includes:
    1. A full Hamiltonian E = -0.5 * s^T W s - h^T s
    2. Temperature T (thermodynamic noise / stochasticity)
    3. MCMC (Metropolis-Hastings) to model probabilistic dynamics.
    """
    def __init__(self, w_ij, temperature=0.1):
        """
        w_ij: The true coupling matrix. Must be either learned (Graphical Lasso on data),
              or hard-coded by the client's business logic. (NO cosine embeddings).
        temperature: T > 0. Determines how easily the system accepts "unfavorable" states.
                     If T is high the system is chaotic. If T is near 0 it freezes into a minimum.
        """
        self.w_ij = np.array(w_ij, dtype=float)
        np.fill_diagonal(self.w_ij, 0)  # A spin does not interact with itself
        self.n_nodes = self.w_ij.shape[0]
        self.T = temperature

    def energy(self, state, h):
        """
        Strict Hamiltonian.
        interaction: internal consistency of beliefs
        field: pressure of external facts (h)
        """
        interaction = -0.5 * np.dot(state.T, np.dot(self.w_ij, state))
        field = -np.dot(h, state)
        return interaction + field

    def mcmc_relax(self, initial_state, h, num_steps=5000):
        """
        The Metropolis-Hastings algorithm for continuous states [-1, 1].
        This is a true walk over the energy landscape, not gradient descent.
        """
        state = np.copy(initial_state)
        current_E = self.energy(state, h)
        
        history_state = [state.copy()]
        history_E = [current_E]
        
        for _ in range(num_steps):
            # 1. Proposal: randomly pick 1 node and try to shift it (Random Walk)
            idx = np.random.randint(self.n_nodes)
            proposal = np.copy(state)
            
            # Shift (normal distribution, bounded by the semantic field [-1, 1])
            proposal[idx] += np.random.normal(0, 0.5)
            proposal[idx] = np.clip(proposal[idx], -1, 1)
            
            # 2. Compute the energy delta
            new_E = self.energy(proposal, h)
            delta_E = new_E - current_E
            
            # 3. Metropolis Acceptance Rule
            if delta_E < 0:
                # If energy drops — accept (roll down into the well)
                state = proposal
                current_E = new_E
            else:
                # If energy rises — accept with Boltzmann probability!
                # Lets the system jump out of local minima.
                p_accept = np.exp(-delta_E / self.T)
                if np.random.rand() < p_accept:
                    state = proposal
                    current_E = new_E
                    
            history_state.append(state.copy())
            history_E.append(current_E)
            
        return state, history_E

# ==========================================
# DEMO: TRUE PHYSICS (Variant B)
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    
    print("=== TRUE ISING MCMC MODEL (Вариант B) ===")
    
    # 1. DEFINE THE TRUE MATRIX W (since there's no data, the B2B client sets it themselves)
    # Nodes:
    # 0: Product is clear
    # 1: Willingness to buy
    # 2: Irritation from bugs
    W = np.array([
        [ 0.0,  0.8, -0.2],
        [ 0.8,  0.0, -0.9],
        [-0.2, -0.9,  0.0]
    ])
    
    print("\nМатрица W (Декларативная топология домена):")
    print(W)
    
    # 2. Initialize the system
    # Low temperature (T=0.1) = the system tends toward a minimum and isn't very chaotic.
    engine = TrueIsingGraph(W, temperature=0.1)
    
    # Current state (Neutral)
    initial_state = np.array([0.0, 0.0, 0.0])
    
    # 3. EXTERNAL FIELD (h)
    # Incoming fact: "User hit a critical bug"
    # This strongly pushes node [2] (Irritation) up (toward +1).
    h = np.array([0.0, 0.0, 2.0]) 
    
    print(f"\nНачальное состояние: {initial_state}")
    print(f"Внешнее давление новых фактов (h): {h}")
    
    # 4. Run MCMC
    print(f"\nЗапуск Metropolis-Hastings (5000 итераций, T={engine.T})...")
    final_state, history_E = engine.mcmc_relax(initial_state, h=h, num_steps=5000)
    
    print(f"\nКонечное состояние системы: {np.round(final_state, 2)}")
    print(f"Итоговая Энергия: {history_E[-1]:.3f}")
    
    print("\nАнализ:")
    print("Узел [2] (Раздражение) улетел в +1.0 из-за внешнего давления h.")
    print("Узел [1] (Готовность купить) ушел в отрицательную зону из-за жесткой связи -0.9 с узлом [2].")
    print("Это честная вероятностная релаксация (MCMC), без косинусных суррогатов.")
