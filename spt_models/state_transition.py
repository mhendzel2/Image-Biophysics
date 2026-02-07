"""
State Transition Models for SPT

Hidden Markov models for bound/unbound state transitions.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings


class StateTransitionModel:
    """
    Simple two-state transition model.
    
    States:
    - 0: Unbound (free diffusion)
    - 1: Bound (confined/immobile)
    
    Transition rates:
    - k_on: 0 -> 1
    - k_off: 1 -> 0
    """
    
    def __init__(self, k_on: float, k_off: float):
        """
        Initialize transition model.
        
        Parameters
        ----------
        k_on : float
            Binding rate [1/s]
        k_off : float
            Unbinding rate [1/s]
        """
        self.k_on = k_on
        self.k_off = k_off
    
    def transition_matrix(self, dt: float) -> np.ndarray:
        """
        Compute transition probability matrix for time step dt.
        
        P(i->j, dt) for i,j in {0,1}
        
        Parameters
        ----------
        dt : float
            Time step [s]
        
        Returns
        -------
        P : np.ndarray
            2x2 transition matrix
        """
        # Analytical solution for two-state system
        k_total = self.k_on + self.k_off
        exp_term = np.exp(-k_total * dt)
        
        # Equilibrium probabilities
        p_unbound_eq = self.k_off / k_total
        p_bound_eq = self.k_on / k_total
        
        # Transition probabilities
        P = np.array([
            [p_unbound_eq + p_bound_eq * exp_term, p_bound_eq * (1 - exp_term)],
            [p_unbound_eq * (1 - exp_term), p_bound_eq + p_unbound_eq * exp_term]
        ])
        
        return P
    
    def steady_state(self) -> np.ndarray:
        """
        Compute steady-state probabilities.
        
        Returns
        -------
        pi : np.ndarray
            [P(unbound), P(bound)]
        """
        k_total = self.k_on + self.k_off
        p_unbound = self.k_off / k_total
        p_bound = self.k_on / k_total
        
        return np.array([p_unbound, p_bound])


class HiddenMarkovModel:
    """
    Hidden Markov Model for SPT state inference.
    
    Infers bound/unbound states from trajectory features
    (e.g., displacement, MSD).
    """
    
    def __init__(
        self,
        k_on: float,
        k_off: float,
        D_free: float,
        D_bound: float
    ):
        """
        Initialize HMM.
        
        Parameters
        ----------
        k_on : float
            Binding rate [1/s]
        k_off : float
            Unbinding rate [1/s]
        D_free : float
            Diffusion coefficient when unbound [μm²/s]
        D_bound : float
            Diffusion coefficient when bound [μm²/s]
        """
        self.k_on = k_on
        self.k_off = k_off
        self.D_free = D_free
        self.D_bound = D_bound
        
        self.transition_model = StateTransitionModel(k_on, k_off)
    
    def emission_probability(
        self,
        displacement: float,
        dt: float,
        state: int
    ) -> float:
        """
        Probability of observing displacement given state.
        
        Assumes 2D diffusion: r² ~ 4 D dt
        
        Parameters
        ----------
        displacement : float
            Observed displacement [μm]
        dt : float
            Time interval [s]
        state : int
            0 (unbound) or 1 (bound)
        
        Returns
        -------
        probability : float
            P(displacement | state)
        """
        D = self.D_free if state == 0 else self.D_bound
        
        # Rayleigh distribution for 2D displacement
        sigma_sq = 2 * D * dt
        
        if sigma_sq <= 0:
            return 0.0
        
        r = displacement
        prob = (r / sigma_sq) * np.exp(-r**2 / (2 * sigma_sq))
        
        return prob
    
    def viterbi(
        self,
        displacements: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Infer most likely state sequence (Viterbi algorithm).
        
        Parameters
        ----------
        displacements : np.ndarray
            Observed displacements [μm]
        dt : float
            Time step [s]
        
        Returns
        -------
        states : np.ndarray
            Most likely state sequence (0=unbound, 1=bound)
        """
        n_obs = len(displacements)
        
        # Transition matrix
        P_trans = self.transition_model.transition_matrix(dt)
        
        # Initialize
        pi = self.transition_model.steady_state()
        
        # Viterbi matrices
        viterbi_prob = np.zeros((2, n_obs))
        viterbi_path = np.zeros((2, n_obs), dtype=int)
        
        # Initial probabilities
        for s in range(2):
            viterbi_prob[s, 0] = (
                pi[s] * self.emission_probability(displacements[0], dt, s)
            )
        
        # Forward pass
        for t in range(1, n_obs):
            for s in range(2):
                # Max over previous states
                trans_probs = viterbi_prob[:, t-1] * P_trans[:, s]
                max_prev = np.argmax(trans_probs)
                
                viterbi_prob[s, t] = (
                    trans_probs[max_prev] *
                    self.emission_probability(displacements[t], dt, s)
                )
                viterbi_path[s, t] = max_prev
        
        # Backward pass
        states = np.zeros(n_obs, dtype=int)
        states[-1] = np.argmax(viterbi_prob[:, -1])
        
        for t in range(n_obs - 2, -1, -1):
            states[t] = viterbi_path[states[t+1], t+1]
        
        return states


def infer_states_from_trajectory(
    trajectory: np.ndarray,
    dt: float,
    k_on: float,
    k_off: float,
    D_free: float,
    D_bound: float
) -> Dict[str, np.ndarray]:
    """
    Infer bound/unbound states from SPT trajectory.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory positions, shape (N, 2) for 2D
    dt : float
        Time step [s]
    k_on, k_off : float
        Transition rates [1/s]
    D_free, D_bound : float
        Diffusion coefficients [μm²/s]
    
    Returns
    -------
    result : dict
        Inferred states and statistics
    """
    # Compute displacements
    displacements = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
    
    # Initialize HMM
    hmm = HiddenMarkovModel(k_on, k_off, D_free, D_bound)
    
    # Infer states
    states = hmm.viterbi(displacements, dt)
    
    # Statistics
    bound_fraction = np.mean(states == 1)
    
    return {
        'states': states,
        'bound_fraction': bound_fraction,
        'n_bound': np.sum(states == 1),
        'n_unbound': np.sum(states == 0)
    }
