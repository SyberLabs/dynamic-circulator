"""
Dynamic Topology Engine — Universal Routing Kernel
===================================================
A fully vectorized, application-agnostic routing kernel for non-stationary
Markov processes over weighted graphs.

Core equations:
    alignment_j = a_t · N_j + b_j
    W_ij(t)     = α · D_ij − β_ij · alignment_j − S_ij

Where:
    D_ij    = static base distance matrix (N×N)
    a_t     = agent telemetry vector at time t (F,)
    N_j     = feature vector of node j (F,)
    b_j     = baseline attractiveness bias of node j (scalar)
    B_ij    = morphing strength tensor, potentially conditioned on
              edge (i,j) and telemetry cluster (N×N)
    S_ij    = additive sponsor friction reduction tensor (N×N);
              reduces traversal cost independent of agent alignment
    α       = physical friction scalar

The transition probability matrix P is obtained by row-wise softmax
over −W/τ, with non-edges masked to −∞ so exp(−∞)=0.

Design principles:
    1. Zero Python loops in the hot path — everything is NumPy broadcast.
    2. β is a tensor (N×N), not a scalar. Sponsors bid per-edge.
    3. Environment writes back into telemetry via a feedback operator
       with entropy injection to prevent telemetry lock-in.
    4. The kernel is ontology-agnostic: nodes can be stores, assets,
       concepts, or anything representable as a feature vector.
    5. Temperature τ controls exploration vs. exploitation.
    6. Softplus weight floor preserves gradient information near zero.
    7. Two sponsor channels: β (alignment-coupled desire) and S
       (alignment-independent friction reduction). Different economic
       primitives with different efficiencies depending on agent state.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Topology:
    """Static structure of the environment."""
    node_features: np.ndarray       # (N, F) — feature matrix
    distance_matrix: np.ndarray     # (N, N) — base traversal costs
    labels: list[str]               # length N — human-readable names
    adjacency_mask: np.ndarray = field(init=False)  # (N, N) bool

    def __post_init__(self):
        D = self.distance_matrix
        # Valid edge: finite, positive, not self-loop
        self.adjacency_mask = (D > 0) & (D < np.inf)
        # Force diagonal to False
        np.fill_diagonal(self.adjacency_mask, False)

    @property
    def N(self) -> int:
        return len(self.labels)

    @property
    def F(self) -> int:
        return self.node_features.shape[1]


@dataclass
class AgentState:
    """Mutable state of a single agent traversing the topology."""
    telemetry: np.ndarray           # (F,) — current desire/intent vector
    position: int                   # index into Topology.labels
    history: list[int] = field(default_factory=list)
    step_count: int = 0

    def __post_init__(self):
        self.history = [self.position]


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class DynamicTopologyKernel:
    """
    The core routing engine. Computes the full N×N transition probability
    matrix in a single vectorized pass, with no Python loops.

    Enhancements over baseline:
        - Softplus weight floor (smooth, gradient-preserving)
        - Temperature τ for exploration/exploitation control
        - Per-node baseline bias (prevents zero-feature black holes)
        - Entropy-injected feedback (prevents telemetry lock-in)
        - Fully vectorized batch simulation
        - Additive sponsor friction channel S_ij (alignment-independent)
    """

    def __init__(
        self,
        topology: Topology,
        alpha: float = 1.0,
        beta: Optional[np.ndarray] = None,
        feedback_fn: Optional[Callable] = None,
        feedback_rate: float = 0.1,
        temperature: float = 1.0,
        temperature_fn: Optional[Callable[[int], float]] = None,
        feedback_noise: float = 0.02,
        weight_floor: float = 0.1,
        floor_sharpness: float = 5.0,
        node_bias: Optional[np.ndarray] = None,
        sponsor_friction: Optional[np.ndarray] = None,
        sponsor_decay: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        topology : Topology
            The static graph structure.
        alpha : float
            Physical friction scalar applied to base distances.
        beta : np.ndarray or None
            Morphing strength tensor. Shape options:
              - None        -> defaults to scalar 2.0 broadcast to (N, N)
              - scalar-like  -> broadcast to (N, N)
              - (N,)        -> per-destination column broadcast
              - (N, N)      -> full per-edge control (sponsors bid here)
        feedback_fn : callable or None
            Environment-to-telemetry feedback operator.
            Signature: (agent_telemetry, visited_node_features) -> new_telemetry
            If None, uses the default exponential moving blend with
            entropy injection.
        feedback_rate : float
            Blending rate for the default feedback function (0 = no feedback,
            1 = fully overwrite telemetry with visited node features).
        temperature : float
            Softmax temperature. Higher = more exploration, lower = more
            greedy. At tau->0 the agent follows the minimum-weight edge
            deterministically.
        temperature_fn : callable or None
            Optional temperature schedule. Signature: (step_count) -> tau.
            If provided, overrides the static temperature at each step.
        feedback_noise : float
            Standard deviation of Gaussian noise injected into telemetry
            after feedback, to prevent convergence to fixed-point attractors.
            Set to 0.0 to disable. Default 0.02 provides mild exploration.
        weight_floor : float
            Minimum effective edge weight. Enforced via softplus (smooth)
            rather than hard clipping to preserve probability discrimination.
        floor_sharpness : float
            Controls the sharpness of the softplus floor transition.
            Higher = sharper (closer to hard clip), lower = smoother.
        node_bias : np.ndarray or None
            Per-node baseline attractiveness, shape (N,). Added to the
            alignment vector before morphing. This ensures nodes with
            zero or sparse feature vectors still participate in the
            dynamic routing. If None, defaults to zeros.
        sponsor_friction : np.ndarray or None
            Additive friction reduction tensor, shape (N, N). Subtracted
            directly from W_ij independent of agent alignment:

                W_ij = α·D_ij − β_ij·alignment_j − S_ij

            Unlike β which requires alignment to have effect, S provides
            100% sponsor efficiency regardless of agent telemetry. Models
            physical friction reduction (e.g. "closer to entrance") rather
            than desire amplification. If None, defaults to all zeros.
        sponsor_decay : float
            Per-TIME-STEP multiplicative decay applied to the *sponsored
            delta* of both β and S tensors via tick_decay(). Decay targets
            the amounts above the at-init baseline, so routing never
            collapses below the unsponsored state.

            Formula (applied each time-step):
                delta_beta    = _beta - _beta_baseline
                _beta         = _beta_baseline + delta_beta * (1 - decay)

            Decay lives in the tick, not the agent-step: step() applies it
            (one agent stepped manually IS the tick) unless called with
            decay=False, simulate_batch() applies it once per time-step
            regardless of population size, and external tick loops that
            drive transition_matrix_batch directly must call tick_decay()
            once per tick.

            At 0.0 (default), no decay — sponsorships are permanent.
            At 0.01, a sponsorship halves in ~69 steps.
        rng : np.random.Generator or None
            Random source for transition sampling and feedback noise.
            If None (default), the legacy global numpy RNG is used, so
            existing np.random.seed-based workflows are unchanged. Pass
            np.random.default_rng(seed) for isolated, reproducible runs
            (required for paired common-random-number experiments).
        """
        self.topo = topology
        self.alpha = alpha
        self.feedback_rate = feedback_rate
        self.temperature = temperature
        self._temperature_fn = temperature_fn
        self._feedback_noise = feedback_noise
        self._weight_floor = weight_floor
        self._floor_sharpness = floor_sharpness
        self._sponsor_decay = max(0.0, min(sponsor_decay, 1.0))
        self._rng = rng

        N = topology.N

        # --- Resolve beta into (N, N) tensor ---
        if beta is None:
            self._beta = np.full((N, N), 2.0)
        else:
            beta = np.asarray(beta, dtype=np.float64)
            if beta.ndim == 0:
                self._beta = np.full((N, N), beta.item())
            elif beta.ndim == 1 and beta.shape[0] == N:
                # Per-destination: each column j gets beta[j]
                self._beta = np.tile(beta, (N, 1))
            elif beta.shape == (N, N):
                self._beta = beta.copy()
            else:
                raise ValueError(
                    f"beta must be scalar, (N,), or (N,N). Got shape {beta.shape}"
                )
        # Snapshot at-init baseline — decay targets the *delta* above this.
        self._beta_baseline = self._beta.copy()

        # --- Per-node baseline bias ---
        if node_bias is None:
            self._node_bias = np.zeros(N)
        else:
            node_bias = np.asarray(node_bias, dtype=np.float64)
            if node_bias.shape != (N,):
                raise ValueError(
                    f"node_bias must have shape ({N},). Got {node_bias.shape}"
                )
            self._node_bias = node_bias.copy()

        # --- Additive sponsor friction matrix S (N, N) ---
        if sponsor_friction is None:
            self._sponsor_friction = np.zeros((N, N))
        else:
            sponsor_friction = np.asarray(sponsor_friction, dtype=np.float64)
            if sponsor_friction.shape != (N, N):
                raise ValueError(
                    f"sponsor_friction must have shape ({N},{N}). "
                    f"Got {sponsor_friction.shape}"
                )
            self._sponsor_friction = sponsor_friction.copy()
        # Snapshot at-init friction baseline for decay targeting.
        self._friction_baseline = self._sponsor_friction.copy()

        # --- Precompute the -inf mask for non-edges ---
        self._neg_inf_mask = np.where(
            topology.adjacency_mask, 0.0, -np.inf
        )

        # --- Feedback function ---
        if feedback_fn is not None:
            self._feedback_fn = feedback_fn
            self._custom_feedback = True
        else:
            self._feedback_fn = self._default_feedback
            self._custom_feedback = False

        # --- Preference-memory update law (Axis 5) — opt-in, off by default ---
        self._memory_law: Optional[dict] = None
        self._reward_expectation: Optional[np.ndarray] = None
        self._last_eta_effective: Optional[np.ndarray] = None
        self._last_opportunity_diagnostic: Optional[dict] = None
        self._edge_learning: Optional[dict] = None
        self._edge_reward_estimate: Optional[np.ndarray] = None
        self._edge_visit_count: Optional[np.ndarray] = None
        self._source_visit_count: Optional[np.ndarray] = None
        self._edge_policy_weight: Optional[np.ndarray] = None
        self._last_edge_policy_mix: Optional[np.ndarray] = None
        self._last_edge_policy_reliability: Optional[np.ndarray] = None

    # -------------------------------------------------------------------
    # Soft floor — smooth alternative to hard clipping
    # -------------------------------------------------------------------

    def _soft_floor(self, W: np.ndarray) -> np.ndarray:
        """
        Apply a smooth minimum to W using a shifted softplus.

        softplus_floor(x) = floor + log(1 + exp(k * (x - floor))) / k

        When x >> floor:  result ≈ x  (passes through unchanged)
        When x << floor:  result ≈ floor  (smooth clamp)
        Transition around x = floor is smooth, preserving gradients.

        DOCUMENTED NONLINEARITY: because the floor is applied before the
        row softmax, it breaks softmax shift-invariance — a uniform shift
        of a row's weights changes post-floor differences. The derivative
        of the floored weight wrt the raw weight is sigmoid(k (W - floor)),
        so intervention efficiency decays smoothly to zero once
        sponsorship pushes an edge below the floor. This is a deliberate
        diminishing-returns property of the sponsorship channel (numeric
        check: leverage overestimated ~250x below the floor if the factor
        is dropped), and edge_leverage / stationary_leverage include the
        sigmoid gate for exactly this reason.
        """
        f = self._weight_floor
        k = self._floor_sharpness
        delta = k * (W - f)
        # Numerically stable softplus:
        #   large delta  → pass-through (W unchanged)
        #   small delta  → smoothly approach floor
        #   delta is inf/nan (from inf distances) → pass-through
        with np.errstate(over='ignore', invalid='ignore'):
            result = f + np.where(
                (delta > 20.0) | ~np.isfinite(delta),
                (W - f),                     # pass-through
                np.log1p(np.exp(delta)) / k  # smooth floor
            )
        return result

    def _edge_learning_context_indices(self, telemetry_batch: np.ndarray) -> np.ndarray:
        centroids = self._edge_learning.get("context_centroids")
        if centroids is None:
            raise ValueError("edge learning is not contextual")
        telemetry_batch = np.asarray(telemetry_batch, dtype=np.float64)
        if telemetry_batch.ndim == 1:
            telemetry_batch = telemetry_batch[np.newaxis, :]
        scores = telemetry_batch @ centroids.T
        return np.argmax(scores, axis=1)

    def _edge_learning_context_reliability(
        self,
        telemetry: Optional[np.ndarray] = None,
        telemetry_batch: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if (
            self._edge_learning is None
            or self._edge_learning.get("policy_reliability", "none") == "none"
            or not self._edge_learning.get("contextual", False)
        ):
            if telemetry_batch is None:
                return np.array(1.0, dtype=np.float64)
            return np.ones(len(telemetry_batch), dtype=np.float64)

        if telemetry_batch is None:
            if telemetry is None:
                return np.array(1.0, dtype=np.float64)
            telemetry_batch = np.asarray(telemetry, dtype=np.float64)[np.newaxis, :]
            scalar = True
        else:
            telemetry_batch = np.asarray(telemetry_batch, dtype=np.float64)
            scalar = False

        centroids = self._edge_learning["context_centroids"]
        scores = telemetry_batch @ centroids.T
        if scores.shape[1] == 1:
            margin = np.maximum(scores[:, 0], 0.0)
        else:
            top2 = np.sort(scores, axis=1)[:, -2:]
            margin = np.maximum(top2[:, 1] - top2[:, 0], 0.0)

        scale = max(self._edge_learning["policy_reliability_scale"], 1e-12)
        floor = self._edge_learning["policy_reliability_floor"]
        reliability = floor + (1.0 - floor) * margin / (margin + scale)
        reliability = np.clip(reliability, floor, 1.0)
        if scalar:
            return np.array(float(reliability[0]), dtype=np.float64)
        return reliability

    def _edge_learning_potential_for_stats(
        self,
        reward: np.ndarray,
        edge_count: np.ndarray,
        source_count: np.ndarray,
    ) -> np.ndarray:
        source_term = np.log1p(source_count)[:, np.newaxis]
        bonus = self._edge_learning["ucb_c"] * np.sqrt(
            source_term / (1.0 + edge_count)
        )
        potential = (
            self._edge_learning["reward_gain"] * reward
            + self._edge_learning["uncertainty_gain"] * bonus
        )
        return np.where(self.topo.adjacency_mask, potential, 0.0)

    def _edge_learning_uncertainty_for_stats(
        self,
        edge_count: np.ndarray,
        source_count: np.ndarray,
    ) -> np.ndarray:
        source_term = np.log1p(source_count)[:, np.newaxis]
        bonus = self._edge_learning["ucb_c"] * np.sqrt(
            source_term / (1.0 + edge_count)
        )
        return np.where(self.topo.adjacency_mask, bonus, 0.0)

    def _edge_learning_potential(
        self,
        telemetry: Optional[np.ndarray] = None,
        telemetry_batch: Optional[np.ndarray] = None,
        context_index: Optional[int] = None,
    ) -> np.ndarray:
        """Return the opt-in edge-learning utility added before softmax."""
        if self._edge_learning is None:
            if telemetry_batch is None:
                return np.zeros((self.topo.N, self.topo.N), dtype=np.float64)
            return np.zeros(
                (len(telemetry_batch), self.topo.N, self.topo.N),
                dtype=np.float64,
            )
        if self._edge_learning["mode"] == "exp3":
            weights = self._edge_policy_weight
            if weights is None:
                if telemetry_batch is None:
                    return np.zeros((self.topo.N, self.topo.N), dtype=np.float64)
                return np.zeros(
                    (len(telemetry_batch), self.topo.N, self.topo.N),
                    dtype=np.float64,
                )

            def log_weight_for(weight_slice: np.ndarray) -> np.ndarray:
                return np.where(
                    self.topo.adjacency_mask,
                    np.log(np.maximum(weight_slice, 1e-12)),
                    0.0,
                )

            if not self._edge_learning.get("contextual", False):
                potential = log_weight_for(weights)
                if telemetry_batch is None:
                    return potential
                return np.broadcast_to(
                    potential,
                    (len(telemetry_batch), self.topo.N, self.topo.N),
                )
            if context_index is not None:
                return log_weight_for(weights[context_index])
            if telemetry_batch is not None:
                indices = self._edge_learning_context_indices(telemetry_batch)
                return np.stack(
                    [log_weight_for(weights[idx]) for idx in indices],
                    axis=0,
                )
            if telemetry is not None:
                idx = int(self._edge_learning_context_indices(telemetry)[0])
                return log_weight_for(weights[idx])
            return np.stack(
                [log_weight_for(weights[idx]) for idx in range(weights.shape[0])],
                axis=0,
            )

        if self._edge_learning["mode"] != "ucb":
            if telemetry_batch is None:
                return np.zeros((self.topo.N, self.topo.N), dtype=np.float64)
            return np.zeros(
                (len(telemetry_batch), self.topo.N, self.topo.N),
                dtype=np.float64,
            )

        reward = self._edge_reward_estimate
        edge_count = self._edge_visit_count
        source_count = self._source_visit_count
        if reward is None or edge_count is None or source_count is None:
            if telemetry_batch is None:
                return np.zeros((self.topo.N, self.topo.N), dtype=np.float64)
            return np.zeros(
                (len(telemetry_batch), self.topo.N, self.topo.N),
                dtype=np.float64,
            )

        if not self._edge_learning.get("contextual", False):
            potential = self._edge_learning_potential_for_stats(
                reward,
                edge_count,
                source_count,
            )
            if telemetry_batch is None:
                return potential
            return np.broadcast_to(
                potential,
                (len(telemetry_batch), self.topo.N, self.topo.N),
            )

        if context_index is not None:
            return self._edge_learning_potential_for_stats(
                reward[context_index],
                edge_count[context_index],
                source_count[context_index],
            )
        if telemetry_batch is not None:
            indices = self._edge_learning_context_indices(telemetry_batch)
            return np.stack(
                [
                    self._edge_learning_potential_for_stats(
                        reward[idx],
                        edge_count[idx],
                        source_count[idx],
                    )
                    for idx in indices
                ],
                axis=0,
            )
        if telemetry is not None:
            idx = int(self._edge_learning_context_indices(telemetry)[0])
            return self._edge_learning_potential_for_stats(
                reward[idx],
                edge_count[idx],
                source_count[idx],
            )
        return np.stack(
            [
                self._edge_learning_potential_for_stats(
                    reward[idx],
                    edge_count[idx],
                    source_count[idx],
                )
                for idx in range(reward.shape[0])
            ],
            axis=0,
        )

    def _edge_learning_uncertainty(
        self,
        telemetry: Optional[np.ndarray] = None,
        telemetry_batch: Optional[np.ndarray] = None,
        context_index: Optional[int] = None,
    ) -> np.ndarray:
        if self._edge_learning is None or self._edge_learning["mode"] != "ucb":
            if telemetry_batch is None:
                return np.zeros((self.topo.N, self.topo.N), dtype=np.float64)
            return np.zeros(
                (len(telemetry_batch), self.topo.N, self.topo.N),
                dtype=np.float64,
            )
        edge_count = self._edge_visit_count
        source_count = self._source_visit_count
        if edge_count is None or source_count is None:
            if telemetry_batch is None:
                return np.zeros((self.topo.N, self.topo.N), dtype=np.float64)
            return np.zeros(
                (len(telemetry_batch), self.topo.N, self.topo.N),
                dtype=np.float64,
            )
        if not self._edge_learning.get("contextual", False):
            uncertainty = self._edge_learning_uncertainty_for_stats(
                edge_count,
                source_count,
            )
            if telemetry_batch is None:
                return uncertainty
            return np.broadcast_to(
                uncertainty,
                (len(telemetry_batch), self.topo.N, self.topo.N),
            )
        if context_index is not None:
            return self._edge_learning_uncertainty_for_stats(
                edge_count[context_index],
                source_count[context_index],
            )
        if telemetry_batch is not None:
            indices = self._edge_learning_context_indices(telemetry_batch)
            return np.stack(
                [
                    self._edge_learning_uncertainty_for_stats(
                        edge_count[idx],
                        source_count[idx],
                    )
                    for idx in indices
                ],
                axis=0,
            )
        if telemetry is not None:
            idx = int(self._edge_learning_context_indices(telemetry)[0])
            return self._edge_learning_uncertainty_for_stats(
                edge_count[idx],
                source_count[idx],
            )
        return np.stack(
            [
                self._edge_learning_uncertainty_for_stats(
                    edge_count[idx],
                    source_count[idx],
                )
                for idx in range(edge_count.shape[0])
            ],
            axis=0,
        )

    def _edge_learning_uses_arbitration(self) -> bool:
        return (
            self._edge_learning is not None
            and self._edge_learning["mode"] in ("ucb", "exp3")
            and self._edge_learning.get("policy", "additive") == "arbitrated"
        )

    def _edge_learning_uniform_policy(self, batch_size: Optional[int] = None) -> np.ndarray:
        mask = self.topo.adjacency_mask.astype(np.float64)
        row_sums = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        uniform = mask / row_sums
        if batch_size is None:
            return uniform
        return np.broadcast_to(uniform, (batch_size, self.topo.N, self.topo.N))

    def _masked_row_softmax(self, scores: np.ndarray, tau: float) -> np.ndarray:
        if scores.ndim == 2:
            logits = (scores / tau) + self._neg_inf_mask
            axis = 1
        elif scores.ndim == 3:
            logits = (scores / tau) + self._neg_inf_mask[np.newaxis, :, :]
            axis = 2
        else:
            raise ValueError("scores must have shape (N,N) or (K,N,N)")
        row_max = np.max(logits, axis=axis, keepdims=True)
        row_max = np.where(np.isfinite(row_max), row_max, 0.0)
        exp_scores = np.exp(logits - row_max)
        row_sums = np.sum(exp_scores, axis=axis, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return exp_scores / row_sums

    def _edge_learning_policy_mix(
        self,
        dte_policy: np.ndarray,
        potential: np.ndarray,
        uncertainty: np.ndarray,
        reliability: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self._edge_learning_uses_arbitration():
            self._last_edge_policy_mix = None
            self._last_edge_policy_reliability = None
            return dte_policy

        policy_tau = max(self._edge_learning["policy_temperature"], 1e-8)
        candidate_policy = self._masked_row_softmax(potential, policy_tau)
        if self._edge_learning["mode"] == "exp3":
            gamma = self._edge_learning["exp3_gamma"]
            batch_size = None if potential.ndim == 2 else potential.shape[0]
            uniform = self._edge_learning_uniform_policy(batch_size=batch_size)
            candidate_policy = (1.0 - gamma) * candidate_policy + gamma * uniform

        if self._edge_learning["mode"] == "exp3":
            if uncertainty.ndim == 2:
                signal = np.ones(self.topo.N, dtype=np.float64)
                reshape = (self.topo.N, 1)
            else:
                signal = np.ones(
                    (uncertainty.shape[0], self.topo.N),
                    dtype=np.float64,
                )
                reshape = (uncertainty.shape[0], self.topo.N, 1)
        elif uncertainty.ndim == 2:
            active_counts = self.topo.adjacency_mask.sum(axis=1)
            signal = np.divide(
                np.sum(uncertainty, axis=1),
                np.maximum(active_counts, 1),
            )
            reshape = (self.topo.N, 1)
        else:
            active_counts = self.topo.adjacency_mask.sum(axis=1)[np.newaxis, :]
            signal = np.divide(
                np.sum(uncertainty, axis=2),
                np.maximum(active_counts, 1),
            )
            reshape = (uncertainty.shape[0], self.topo.N, 1)

        scale = max(self._edge_learning["policy_uncertainty_scale"], 1e-12)
        mix_min = self._edge_learning["policy_mix_min"]
        mix_max = self._edge_learning["policy_mix_max"]
        mix = mix_min + (mix_max - mix_min) * signal / (signal + scale)
        mix = np.clip(mix, mix_min, mix_max)
        if reliability is not None:
            reliability = np.asarray(reliability, dtype=np.float64)
            if uncertainty.ndim == 2:
                rel = float(reliability)
                mix *= rel
                self._last_edge_policy_reliability = np.full_like(mix, rel)
            else:
                rel = reliability.reshape(-1, 1)
                mix *= rel
                self._last_edge_policy_reliability = np.broadcast_to(
                    rel,
                    mix.shape,
                ).copy()
        else:
            self._last_edge_policy_reliability = np.ones_like(mix)
        self._last_edge_policy_mix = mix.copy()
        return (
            (1.0 - mix.reshape(reshape)) * dte_policy
            + mix.reshape(reshape) * candidate_policy
        )

    # -------------------------------------------------------------------
    # Core computation — fully vectorized, zero loops
    # -------------------------------------------------------------------

    def transition_matrix(
        self, telemetry: np.ndarray, step: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute the full N*N transition probability matrix for a given
        telemetry vector.

        Parameters
        ----------
        telemetry : np.ndarray, shape (F,)
        step : int or None
            Current simulation step, used for temperature scheduling.

        Returns
        -------
        P : np.ndarray, shape (N, N)
            Row-stochastic matrix. P[i, j] = probability of moving
            from node i to node j. Rows for nodes with no neighbors
            sum to 0.
        """
        F_mat = self.topo.node_features   # (N, F)
        D = self.topo.distance_matrix     # (N, N)

        # 1. Alignment: how much each node matches the agent
        #    alignment[j] = telemetry . features_j + bias_j
        alignment = F_mat @ telemetry + self._node_bias  # (N,)

        # 2. Broadcast alignment into (N, N): every row i sees
        #    the alignment of destination j
        A_broadcast = alignment[np.newaxis, :]   # (1, N)

        # 3. Dynamic weight matrix
        #    W_ij = alpha*D_ij - beta_ij * alignment_j - S_ij
        W = (self.alpha * D) - (self._beta * A_broadcast) - self._sponsor_friction
        edge_potential = self._edge_learning_potential(telemetry=telemetry)
        if not self._edge_learning_uses_arbitration():
            W = W - edge_potential

        # 4. Softplus floor (smooth, gradient-preserving)
        W = self._soft_floor(W)

        # 5. Resolve temperature
        tau = self._resolve_temperature(step)

        # 6. Row-wise softmax over -W/tau, with non-edges masked to -inf
        neg_W = (-W / tau) + self._neg_inf_mask   # (N, N)

        # Numerically stable softmax: subtract row max
        row_max = np.max(neg_W, axis=1, keepdims=True)
        # Handle rows that are all -inf (isolated nodes)
        row_max = np.where(np.isfinite(row_max), row_max, 0.0)

        exp_W = np.exp(neg_W - row_max)
        row_sums = np.sum(exp_W, axis=1, keepdims=True)

        # Avoid division by zero for isolated nodes
        row_sums = np.where(row_sums > 0, row_sums, 1.0)

        P = exp_W / row_sums
        if self._edge_learning_uses_arbitration():
            uncertainty = self._edge_learning_uncertainty(telemetry=telemetry)
            reliability = self._edge_learning_context_reliability(telemetry=telemetry)
            P = self._edge_learning_policy_mix(
                P,
                edge_potential,
                uncertainty,
                reliability=reliability,
            )
        return P

    def _resolve_temperature(self, step: Optional[int] = None) -> float:
        """Get effective temperature, applying schedule if available."""
        if self._temperature_fn is not None and step is not None:
            return max(self._temperature_fn(step), 1e-8)
        return max(self.temperature, 1e-8)

    # -------------------------------------------------------------------
    # Batched transition matrix — K agents simultaneously
    # -------------------------------------------------------------------

    def transition_matrix_batch(
        self, telemetry_batch: np.ndarray, step: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute K transition matrices simultaneously using fully
        vectorized tensor operations. No Python loops.

        Parameters
        ----------
        telemetry_batch : np.ndarray, shape (K, F)
        step : int or None

        Returns
        -------
        P : np.ndarray, shape (K, N, N)
        """
        K = telemetry_batch.shape[0]
        F_mat = self.topo.node_features     # (N, F)
        D = self.topo.distance_matrix       # (N, N)

        # Alignment: (K, N)
        alignment = telemetry_batch @ F_mat.T + self._node_bias[np.newaxis, :]

        # Broadcast to (K, N, N): each agent sees all destination alignments
        A_broadcast = alignment[:, np.newaxis, :]   # (K, 1, N)

        # Dynamic weights: (K, N, N)
        #   W_ij = alpha*D_ij - beta_ij*alignment_j - S_ij
        W = (
            (self.alpha * D[np.newaxis, :, :])
            - (self._beta[np.newaxis, :, :] * A_broadcast)
            - self._sponsor_friction[np.newaxis, :, :]
        )
        edge_potential = self._edge_learning_potential(telemetry_batch=telemetry_batch)
        if not self._edge_learning_uses_arbitration():
            W = W - edge_potential

        # Softplus floor
        W = self._soft_floor(W)

        # Temperature
        tau = self._resolve_temperature(step)

        # Softmax with mask: (K, N, N)
        neg_W = (-W / tau) + self._neg_inf_mask[np.newaxis, :, :]

        row_max = np.max(neg_W, axis=2, keepdims=True)
        row_max = np.where(np.isfinite(row_max), row_max, 0.0)

        exp_W = np.exp(neg_W - row_max)
        row_sums = np.sum(exp_W, axis=2, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)

        P = exp_W / row_sums
        if self._edge_learning_uses_arbitration():
            uncertainty = self._edge_learning_uncertainty(
                telemetry_batch=telemetry_batch
            )
            reliability = self._edge_learning_context_reliability(
                telemetry_batch=telemetry_batch
            )
            P = self._edge_learning_policy_mix(
                P,
                edge_potential,
                uncertainty,
                reliability=reliability,
            )
        return P

    # -------------------------------------------------------------------
    # Agent stepping
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Random source — instance Generator when injected, legacy global
    # np.random otherwise (preserves np.random.seed-based workflows)
    # -------------------------------------------------------------------

    def _draw_choice(self, n: int, p: np.ndarray) -> int:
        if self._rng is not None:
            return int(self._rng.choice(n, p=p))
        return int(np.random.choice(n, p=p))

    def _draw_uniform(self, shape: tuple[int, ...]) -> np.ndarray:
        if self._rng is not None:
            return self._rng.random(shape)
        return np.random.random(shape)

    def _draw_normal(self, shape: tuple[int, ...]) -> np.ndarray:
        if self._rng is not None:
            return self._rng.standard_normal(shape)
        return np.random.randn(*shape)

    def tick_decay(self) -> None:
        """
        Advance sponsorship decay by ONE time-step.

        Erodes the sponsored delta of β and S above their at-init
        baselines. This is time semantics, not agent semantics: call it
        once per simulation tick no matter how many agents moved. step()
        and simulate_batch() call it automatically; external loops that
        drive transition_matrix_batch directly must call it themselves.
        """
        if self._sponsor_decay > 0:
            beta_delta = self._beta - self._beta_baseline
            self._beta = self._beta_baseline + beta_delta * (1.0 - self._sponsor_decay)
            fric_delta = self._sponsor_friction - self._friction_baseline
            self._sponsor_friction = (
                self._friction_baseline + fric_delta * (1.0 - self._sponsor_decay)
            )

    def step(self, agent: AgentState, decay: bool = True) -> tuple[int, np.ndarray]:
        """
        Advance the agent one step through the topology.

        1. Compute transition matrix from current telemetry.
        2. Sample next node from the agent's current row.
        3. Apply environment->telemetry feedback.
        4. Update agent state.

        Parameters
        ----------
        decay : bool
            Apply one tick of sponsorship decay (default True). A single
            agent stepped manually IS the tick. When stepping SEVERAL
            agents through one kernel within the same logical time-step,
            pass decay=False and call tick_decay() once per tick yourself,
            otherwise decay compounds per agent-step.

        Returns
        -------
        next_node : int
            Index of the node the agent moved to.
        P : np.ndarray
            The full transition matrix (for inspection/logging).
        """
        P = self.transition_matrix(agent.telemetry, step=agent.step_count)
        row = P[agent.position]

        if np.sum(row) == 0:
            # Terminal / isolated node
            return agent.position, P

        next_node = self._draw_choice(self.topo.N, row)

        # --- Environment feedback: the visited node writes back ---
        visited_features = self.topo.node_features[next_node]
        agent.telemetry = self._feedback_fn(agent.telemetry, visited_features)

        # --- Update agent state ---
        agent.position = next_node
        agent.history.append(next_node)
        agent.step_count += 1

        if decay:
            self.tick_decay()

        return next_node, P

    def simulate(
        self, agent: AgentState, steps: int, verbose: bool = False
    ) -> list[int]:
        """
        Run a multi-step traversal, returning the full path.
        """
        if verbose:
            print(f"{'='*60}")
            print(f"  SIMULATION START")
            print(f"  Telemetry : {agent.telemetry}")
            print(f"  Origin    : {self.topo.labels[agent.position]}")
            print(f"  Temp      : {self.temperature}")
            print(f"  Noise     : {self._feedback_noise}")
            print(f"{'='*60}\n")

        for s in range(steps):
            prev = agent.position
            next_node, P = self.step(agent)

            if verbose:
                self._print_step(s + 1, prev, next_node, P, agent.telemetry)

        return agent.history

    # -------------------------------------------------------------------
    # Batch simulation — fully vectorized
    # -------------------------------------------------------------------

    def simulate_batch(
        self,
        telemetry_batch: np.ndarray,
        start_indices: np.ndarray,
        steps: int,
    ) -> np.ndarray:
        """
        Simulate K agents in parallel over the same topology.
        Uses fully vectorized transition matrix computation.

        Parameters
        ----------
        telemetry_batch : np.ndarray, shape (K, F)
        start_indices : np.ndarray, shape (K,), dtype int
        steps : int

        Returns
        -------
        paths : np.ndarray, shape (K, steps+1)
            Full path history for each agent.
        """
        K = telemetry_batch.shape[0]
        N = self.topo.N
        positions = start_indices.copy()
        telemetries = telemetry_batch.copy()
        paths = np.zeros((K, steps + 1), dtype=int)
        paths[:, 0] = positions

        for s in range(steps):
            # Vectorized: compute all K transition matrices at once
            P_all = self.transition_matrix_batch(telemetries, step=s)  # (K, N, N)

            # Extract each agent's current row: P_all[k, positions[k], :]
            rows = P_all[np.arange(K), positions, :]  # (K, N)

            # Vectorized multinomial sampling via inverse CDF
            cumsum = np.cumsum(rows, axis=1)           # (K, N)
            u = self._draw_uniform((K, 1))             # (K, 1)
            next_nodes = np.argmax(cumsum >= u, axis=1) # (K,)

            # Handle isolated nodes (row sums to 0): stay in place
            row_sums = rows.sum(axis=1)
            isolated = row_sums == 0
            next_nodes[isolated] = positions[isolated]

            visited_features = self.topo.node_features[next_nodes]  # (K, F)
            if self._custom_feedback:
                # A user-supplied feedback_fn must be honored in batch mode
                # too; it owns its noise/normalization, same contract as
                # step(). Per-agent call — correctness over vectorization.
                for k in range(K):
                    telemetries[k] = self._feedback_fn(
                        telemetries[k], visited_features[k]
                    )
            else:
                # Default feedback — fully vectorized EMA + noise + renorm
                lam = self.feedback_rate
                telemetries = (1.0 - lam) * telemetries + lam * visited_features
                if self._feedback_noise > 0:
                    telemetries += (
                        self._draw_normal(telemetries.shape) * self._feedback_noise
                    )
                # Re-normalize each agent's telemetry to the unit sphere
                norms = np.linalg.norm(telemetries, axis=1, keepdims=True)
                telemetries = np.where(norms > 0, telemetries / norms, telemetries)

            positions = next_nodes
            paths[:, s + 1] = positions

            # Sponsorship decay is time semantics: once per step, not per agent
            self.tick_decay()

        return paths

    # -------------------------------------------------------------------
    # Effective-topology rewriting — activate / prune edges over a fixed
    # physical substrate. The graph is derived from the distance matrix
    # (0 < D_ij < inf => edge), so toggling an edge is a local update to
    # D, the adjacency mask, and the softmax -inf mask. This is the minimal
    # combinatorial-rewrite primitive: physical topology (which edges can
    # ever exist) is fixed; effective topology (which are currently usable)
    # changes. Callers own the consolidation policy that decides when.
    # -------------------------------------------------------------------

    def set_edge_active(
        self,
        from_idx: int,
        to_idx: int,
        active: bool,
        distance: Optional[float] = None,
    ) -> None:
        """
        Activate or prune a single directed edge at runtime.

        Activating requires a finite positive `distance` (the physical
        traversal cost of the edge); if omitted, a previously stored latent
        distance for this edge is reused. Pruning sets D_ij = inf. The
        edge's beta and sponsor-friction entries are left at their existing
        values (default beta, zero friction for a never-used latent edge),
        and the stationary / leverage / diagnostic machinery picks up the
        new admissibility on its next call.
        """
        if from_idx == to_idx:
            raise ValueError("cannot toggle a self-loop")
        if not hasattr(self, "_latent_distance"):
            self._latent_distance: dict[tuple[int, int], float] = {}
        D = self.topo.distance_matrix
        if active:
            d = distance if distance is not None else self._latent_distance.get((from_idx, to_idx))
            if d is None or not (0.0 < float(d) < np.inf):
                raise ValueError(
                    "activating an edge needs a finite positive distance"
                )
            D[from_idx, to_idx] = float(d)
            self.topo.adjacency_mask[from_idx, to_idx] = True
            self._neg_inf_mask[from_idx, to_idx] = 0.0
        else:
            if np.isfinite(D[from_idx, to_idx]):
                self._latent_distance[(from_idx, to_idx)] = float(D[from_idx, to_idx])
            D[from_idx, to_idx] = np.inf
            self.topo.adjacency_mask[from_idx, to_idx] = False
            self._neg_inf_mask[from_idx, to_idx] = -np.inf

    def edge_active(self, from_idx: int, to_idx: int) -> bool:
        """Whether an edge is currently in the effective topology."""
        return bool(self.topo.adjacency_mask[from_idx, to_idx])

    # -------------------------------------------------------------------
    # Sponsor API — mutate the beta tensor at runtime
    # -------------------------------------------------------------------

    def sponsor_edge(
        self, from_idx: int, to_idx: int, boost: float
    ) -> None:
        """
        A sponsor bids on a specific edge, increasing its morphing
        strength. This makes the destination node more attractive
        to agents whose telemetry aligns with it.
        """
        self._beta[from_idx, to_idx] += boost

    def sponsor_node(self, node_idx: int, boost: float) -> None:
        """
        Sponsor an entire node: boost all inbound edges.
        """
        self._beta[:, node_idx] += boost

    def set_beta_for_cluster(
        self,
        edge_mask: np.ndarray,
        value: float,
    ) -> None:
        """
        Set beta values for a subset of edges, identified by a boolean
        mask of shape (N, N).
        """
        self._beta[edge_mask] = value

    # -------------------------------------------------------------------
    # Sponsor friction API — alignment-independent traversal cost reduction
    # -------------------------------------------------------------------

    def sponsor_edge_friction(
        self, from_idx: int, to_idx: int, reduction: float
    ) -> None:
        """
        Reduce the traversal cost of a specific edge by a fixed amount,
        independent of agent alignment.

        Unlike sponsor_edge() which scales with alignment (β · alignment_j),
        this directly lowers W_ij by `reduction` for ALL agents, regardless
        of their telemetry. A reduction of 1.0 is equivalent to shortening
        the physical distance by 1/alpha units.

        Parameters
        ----------
        from_idx : int
            Source node index.
        to_idx : int
            Destination node index.
        reduction : float
            Amount to subtract from W_ij. Positive = cheaper edge.
        """
        self._sponsor_friction[from_idx, to_idx] += reduction

    def sponsor_node_friction(self, node_idx: int, reduction: float) -> None:
        """
        Apply friction reduction to all inbound edges of a node.
        All agents find this node easier to reach, regardless of alignment.
        """
        self._sponsor_friction[:, node_idx] += reduction

    def reset_friction(self, from_idx: int = None, to_idx: int = None) -> None:
        """
        Reset sponsor friction. If both indices given, reset that edge only.
        If only one index given, reset all edges into/out of that node.
        If neither given, reset the entire friction matrix to zero.
        """
        if from_idx is None and to_idx is None:
            self._sponsor_friction[:] = 0.0
        elif from_idx is not None and to_idx is not None:
            self._sponsor_friction[from_idx, to_idx] = 0.0
        elif from_idx is not None:
            self._sponsor_friction[from_idx, :] = 0.0
        else:
            self._sponsor_friction[:, to_idx] = 0.0

    # -------------------------------------------------------------------
    # Edge-learning potential (DTE-native UCB) — opt-in
    #
    # This layer internalizes a no-regret idea without replacing the DTE
    # topology. Each admissible edge can carry a reward estimate and an
    # uncertainty bonus, injected as an additional utility before softmax:
    #
    #   W^UCB_ij = W_ij - reward_gain*rhat_ij
    #                    - uncertainty_gain*c*sqrt(log(1+n_i)/(1+n_ij))
    #
    # The default remains static; no transition probabilities change unless
    # configure_edge_learning(mode="ucb") is called.
    # -------------------------------------------------------------------

    def configure_edge_learning(
        self,
        mode: str = "static",
        reward_gain: float = 1.0,
        uncertainty_gain: float = 1.0,
        ucb_c: float = 1.0,
        initial_reward: float = 0.0,
        context_centroids: Optional[np.ndarray] = None,
        policy: str = "additive",
        policy_mix_min: float = 0.0,
        policy_mix_max: float = 0.35,
        policy_uncertainty_scale: float = 0.10,
        policy_temperature: float = 0.25,
        policy_reliability: str = "none",
        policy_reliability_floor: float = 0.0,
        policy_reliability_scale: float = 0.10,
        exp3_gamma: float = 0.10,
        exp3_eta: float = 0.20,
        exp3_weight_cap: float = 1e50,
        exp3_ix: Optional[float] = None,
    ) -> None:
        """
        Configure opt-in DTE-native edge learning.

        Modes
        -----
        static : no edge-learning potential (default)
        ucb    : add learned reward and UCB optimism to admissible edges
        exp3   : multiplicative adversarial edge weighting, arbitrated only

        The exp3 gain estimator is importance-weighted with implicit
        exploration (EXP3-IX): gain = r / (p_hat + exp3_ix) on traffic
        support, where p_hat is the realized per-source selection
        frequency. exp3_ix defaults to exp3_eta / 2 (Neu 2015). Weights
        are renormalized per source row after every update (softmax over
        log-weights is shift-invariant), so exp3_weight_cap is inert and
        retained only for API compatibility.
        """
        modes = ("static", "ucb", "exp3")
        if mode not in modes:
            raise ValueError(f"mode must be one of {modes}. Got {mode!r}")
        policies = ("additive", "arbitrated")
        if policy not in policies:
            raise ValueError(f"policy must be one of {policies}. Got {policy!r}")
        if mode == "exp3" and policy != "arbitrated":
            raise ValueError("exp3 edge learning requires policy='arbitrated'")
        reliability_modes = ("none", "centroid_margin")
        if policy_reliability not in reliability_modes:
            raise ValueError(
                f"policy_reliability must be one of {reliability_modes}. "
                f"Got {policy_reliability!r}"
            )
        for name, value in {
            "reward_gain": reward_gain,
            "uncertainty_gain": uncertainty_gain,
            "ucb_c": ucb_c,
            "policy_mix_min": policy_mix_min,
            "policy_mix_max": policy_mix_max,
            "policy_uncertainty_scale": policy_uncertainty_scale,
            "policy_temperature": policy_temperature,
            "policy_reliability_floor": policy_reliability_floor,
            "policy_reliability_scale": policy_reliability_scale,
            "exp3_gamma": exp3_gamma,
            "exp3_eta": exp3_eta,
            "exp3_weight_cap": exp3_weight_cap,
        }.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if exp3_ix is not None and (not np.isfinite(exp3_ix) or exp3_ix < 0):
            raise ValueError("exp3_ix must be finite and non-negative")
        if policy_mix_min > policy_mix_max:
            raise ValueError("policy_mix_min must be <= policy_mix_max")
        if policy_reliability_floor > 1.0:
            raise ValueError("policy_reliability_floor must be <= 1")
        if exp3_gamma > 1.0:
            raise ValueError("exp3_gamma must be <= 1")
        if exp3_weight_cap < 1.0:
            raise ValueError("exp3_weight_cap must be >= 1")
        if not np.isfinite(initial_reward):
            raise ValueError("initial_reward must be finite")
        centroids = None
        if context_centroids is not None:
            centroids = np.asarray(context_centroids, dtype=np.float64)
            if centroids.ndim != 2 or centroids.shape[1] != self.topo.F:
                raise ValueError(
                    "context_centroids must have shape (K,F), "
                    f"where F={self.topo.F}. Got {centroids.shape}"
                )
            if len(centroids) == 0:
                raise ValueError("context_centroids must contain at least one centroid")
            if np.any(~np.isfinite(centroids)):
                raise ValueError("context_centroids must be finite")
            centroids = centroids / np.maximum(
                np.linalg.norm(centroids, axis=1, keepdims=True),
                1e-12,
            )

        if mode == "static":
            self._edge_learning = None
            self._edge_reward_estimate = None
            self._edge_visit_count = None
            self._source_visit_count = None
            self._edge_policy_weight = None
            self._last_edge_policy_mix = None
            self._last_edge_policy_reliability = None
            return

        N = self.topo.N
        self._edge_learning = {
            "mode": mode,
            "reward_gain": float(reward_gain),
            "uncertainty_gain": float(uncertainty_gain),
            "ucb_c": float(ucb_c),
            "initial_reward": float(initial_reward),
            "contextual": centroids is not None,
            "context_centroids": centroids,
            "policy": policy,
            "policy_mix_min": float(policy_mix_min),
            "policy_mix_max": float(policy_mix_max),
            "policy_uncertainty_scale": float(policy_uncertainty_scale),
            "policy_temperature": float(policy_temperature),
            "policy_reliability": policy_reliability,
            "policy_reliability_floor": float(policy_reliability_floor),
            "policy_reliability_scale": float(policy_reliability_scale),
            "exp3_gamma": float(exp3_gamma),
            "exp3_eta": float(exp3_eta),
            "exp3_weight_cap": float(exp3_weight_cap),
            "exp3_ix": float(exp3_ix) if exp3_ix is not None else float(exp3_eta) / 2.0,
        }
        base_reward = np.where(
            self.topo.adjacency_mask,
            float(initial_reward),
            0.0,
        )
        base_weight = np.where(self.topo.adjacency_mask, 1.0, 0.0)
        if centroids is None:
            self._edge_reward_estimate = base_reward
            self._edge_visit_count = np.zeros((N, N), dtype=np.float64)
            self._source_visit_count = np.zeros(N, dtype=np.float64)
            self._edge_policy_weight = base_weight.copy()
        else:
            K = centroids.shape[0]
            self._edge_reward_estimate = np.broadcast_to(
                base_reward,
                (K, N, N),
            ).copy()
            self._edge_visit_count = np.zeros((K, N, N), dtype=np.float64)
            self._source_visit_count = np.zeros((K, N), dtype=np.float64)
            self._edge_policy_weight = np.broadcast_to(
                base_weight,
                (K, N, N),
            ).copy()

    def _exp3_weight_update(
        self,
        weight_view: np.ndarray,
        realized: np.ndarray,
        traffic: np.ndarray,
    ) -> np.ndarray:
        """
        EXP3-IX multiplicative weight update from realized population traffic.

        The gain estimator is importance-weighted with implicit exploration:
            gain_ij = realized_ij / (p_hat_ij + ix)   on traffic support,
        where p_hat_ij = traffic_ij / sum_k traffic_ik is the realized
        selection frequency out of source i. Dividing (rather than
        multiplying) by the selection probability keeps the estimator
        unbiased up to the IX smoothing term: a high-reward, rarely-taken
        edge is not throttled by its own unpopularity, so the lane cannot
        reproduce the deadly-familiarity lock-in it exists to mitigate.

        After the multiplicative update every source row is renormalized by
        its maximum weight. softmax over log-weights is invariant to this
        rescaling, and it prevents unbounded weight growth (which would
        otherwise saturate exp3_weight_cap and collapse learned gaps).
        """
        ix = self._edge_learning["exp3_ix"]
        row_mass = traffic.sum(axis=1, keepdims=True)
        p_hat = np.where(
            row_mass > 0,
            traffic / np.maximum(row_mass, 1e-12),
            0.0,
        )
        gain = np.where(traffic > 0, realized / (p_hat + ix), 0.0)
        weight_next = weight_view * np.exp(self._edge_learning["exp3_eta"] * gain)
        weight_next = np.where(
            self.topo.adjacency_mask,
            np.minimum(weight_next, self._edge_learning["exp3_weight_cap"]),
            0.0,
        )
        row_max = np.max(weight_next, axis=1, keepdims=True)
        return np.where(
            self.topo.adjacency_mask,
            weight_next / np.maximum(row_max, 1e-12),
            0.0,
        )

    def edge_learning_step(
        self,
        traffic: np.ndarray,
        node_reward: Optional[np.ndarray] = None,
        edge_reward: Optional[np.ndarray] = None,
        context_index: Optional[int] = None,
    ) -> dict:
        """
        Update DTE-native edge-learning statistics from realized traffic.

        Provide either:
          - node_reward, shape (N,), interpreted as destination reward; or
          - edge_reward, shape (N,N), interpreted as realized edge reward.
        """
        if self._edge_learning is None:
            return {"mode": "static", "traffic_mass": 0.0}
        traffic = np.asarray(traffic, dtype=np.float64)
        N = self.topo.N
        if traffic.shape != (N, N):
            raise ValueError(f"traffic must have shape ({N},{N}). Got {traffic.shape}")
        if np.any(~np.isfinite(traffic)) or np.any(traffic < 0):
            raise ValueError("traffic must be finite and non-negative")
        traffic = np.where(self.topo.adjacency_mask, traffic, 0.0)
        if self._edge_learning.get("contextual", False):
            if context_index is None:
                raise ValueError("contextual edge learning requires context_index")
            context_index = int(context_index)
            if context_index < 0 or context_index >= self._edge_reward_estimate.shape[0]:
                raise ValueError("context_index is out of range")
            reward_view = self._edge_reward_estimate[context_index]
            count_view = self._edge_visit_count[context_index]
            source_view = self._source_visit_count[context_index]
            weight_view = self._edge_policy_weight[context_index]
        else:
            if context_index is not None:
                raise ValueError("context_index requires contextual edge learning")
            reward_view = self._edge_reward_estimate
            count_view = self._edge_visit_count
            source_view = self._source_visit_count
            weight_view = self._edge_policy_weight

        if (node_reward is None) == (edge_reward is None):
            raise ValueError("provide exactly one of node_reward or edge_reward")
        if node_reward is not None:
            node_reward = np.asarray(node_reward, dtype=np.float64)
            if node_reward.shape != (N,):
                raise ValueError(f"node_reward must have shape ({N},). Got {node_reward.shape}")
            if np.any(~np.isfinite(node_reward)):
                raise ValueError("node_reward must be finite")
            realized = np.broadcast_to(node_reward[np.newaxis, :], (N, N))
        else:
            edge_reward = np.asarray(edge_reward, dtype=np.float64)
            if edge_reward.shape != (N, N):
                raise ValueError(f"edge_reward must have shape ({N},{N}). Got {edge_reward.shape}")
            if np.any(~np.isfinite(edge_reward)):
                raise ValueError("edge_reward must be finite")
            realized = edge_reward
        realized = np.where(self.topo.adjacency_mask, realized, 0.0)

        previous_count = count_view.copy()
        new_count = previous_count + traffic
        weighted_reward = (
            reward_view * previous_count
            + realized * traffic
        )
        reward_next = np.where(
            new_count > 0,
            weighted_reward / np.maximum(new_count, 1e-12),
            reward_view,
        )
        reward_next = np.where(
            self.topo.adjacency_mask,
            reward_next,
            0.0,
        )
        if self._edge_learning.get("contextual", False):
            self._edge_reward_estimate[context_index] = reward_next
            self._edge_visit_count[context_index] = new_count
            self._source_visit_count[context_index] += traffic.sum(axis=1)
            if self._edge_learning["mode"] == "exp3":
                self._edge_policy_weight[context_index] = self._exp3_weight_update(
                    weight_view, realized, traffic
                )
            potential = self._edge_learning_potential(context_index=context_index)
        else:
            self._edge_reward_estimate = reward_next
            self._edge_visit_count = new_count
            self._source_visit_count += traffic.sum(axis=1)
            if self._edge_learning["mode"] == "exp3":
                self._edge_policy_weight = self._exp3_weight_update(
                    weight_view, realized, traffic
                )
            potential = self._edge_learning_potential()
        return {
            "mode": self._edge_learning["mode"],
            "context_index": context_index,
            "traffic_mass": float(traffic.sum()),
            "mean_reward_estimate": float(
                np.mean(reward_next[self.topo.adjacency_mask])
            ) if np.any(self.topo.adjacency_mask) else 0.0,
            "max_ucb_bonus": float(
                np.max(potential)
            ),
        }

    def edge_learning_state(self) -> dict:
        """Diagnostic snapshot of DTE-native edge learning."""
        if self._edge_learning is None:
            return {"mode": "static"}
        return {
            **self._edge_learning,
            "edge_reward_estimate": self._edge_reward_estimate.copy(),
            "edge_visit_count": self._edge_visit_count.copy(),
            "source_visit_count": self._source_visit_count.copy(),
            "edge_policy_weight": (
                None
                if self._edge_policy_weight is None
                else self._edge_policy_weight.copy()
            ),
            "potential": self._edge_learning_potential().copy(),
            "uncertainty": self._edge_learning_uncertainty().copy(),
            "last_policy_mix": (
                None
                if self._last_edge_policy_mix is None
                else self._last_edge_policy_mix.copy()
            ),
            "last_policy_reliability": (
                None
                if self._last_edge_policy_reliability is None
                else self._last_edge_policy_reliability.copy()
            ),
        }

    # -------------------------------------------------------------------
    # Preference-memory update law (Axis 5) — opt-in
    #
    # Promoted from the design-space program (DTE_DESIGN_SPACE_REPORT.md,
    # DTE_DESIGN_SPACE_SECOND_PASS_REPORT.md). Stated reason: stale lock-in
    # and its mitigation were previously witnessed only in per-experiment
    # pheromone loops; the second pass showed surprise-gated evaporation
    # removes the deadly-familiarity region on both the two-route and ant
    # witnesses without losing the adaptive-memory region. The DEFAULT
    # remains "static" (no endogenous reinforcement): existing behavior is
    # unchanged unless configure_memory_law() is called.
    # -------------------------------------------------------------------

    def configure_memory_law(
        self,
        mode: str = "static",
        channel: str = "friction",
        rho: float = 0.0,
        eta: float = 0.0,
        eta_max: float = 0.5,
        surprise_gain: float = 2.0,
        opportunity_gain: float = 0.0,
        reward_track_rate: float = 0.25,
        initial_expectation: float = 1.0,
    ) -> None:
        """
        Configure the endogenous preference-memory update law.

        Modes
        -----
        static       : no endogenous update (DEFAULT — kernel behaves as
                       before; sponsorship changes only via the sponsor API).
        traffic      : delta_{t+1} = (1-eta)*delta_t + rho*traffic
                       (pheromone-style; reinforces visited edges regardless
                       of outcome — note the design-space result that this
                       lowers the unrecovered-lock-in threshold ~5x).
        reward_gated : delta_{t+1} = (1-eta)*delta_t
                       + rho*traffic*node_reward[dest]
                       (reinforcement requires the visit to have paid off).
        adaptive_eta : reward_gated deposit, but evaporation per destination
                        rises with reward-expectation surprise:
                        eta_j = min(eta_max, eta + gain*w_j*max(0, rhat_j-r_j))
                        where w_j is the destination's share of current
                        traffic and rhat_j is an EMA of observed reward.
                        If opportunity_gain > 0, eta also rises with local
                        opportunity cost: traffic that chose a destination
                        whose reward is below a currently reachable alternative.

        channel : "friction" updates the sponsor-friction tensor S (the
                  pheromone-as-friction convention used by the ant domain);
                  "beta" updates the alignment-coupled tensor instead.
        The law operates on the *delta above the at-init baseline*, like
        sponsor_decay; avoid combining both on the same channel unless the
        double evaporation is intended.
        """
        modes = ("static", "traffic", "reward_gated", "adaptive_eta")
        channels = ("friction", "beta")
        if mode not in modes:
            raise ValueError(f"mode must be one of {modes}. Got {mode!r}")
        if channel not in channels:
            raise ValueError(f"channel must be one of {channels}. Got {channel!r}")
        if mode == "static":
            self._memory_law = None
            self._reward_expectation = None
            self._last_eta_effective = None
            self._last_opportunity_diagnostic = None
            return
        self._memory_law = {
            "mode": mode,
            "channel": channel,
            "rho": float(rho),
            "eta": float(eta),
            "eta_max": float(eta_max),
            "surprise_gain": float(surprise_gain),
            "opportunity_gain": float(opportunity_gain),
            "reward_track_rate": float(reward_track_rate),
        }
        self._reward_expectation = np.full(self.topo.N, float(initial_expectation))
        self._last_eta_effective = None
        self._last_opportunity_diagnostic = None

    def opportunity_cost_diagnostic(
        self,
        traffic: np.ndarray,
        node_reward: np.ndarray,
    ) -> dict:
        """
        Diagnose local regret in realized traffic.

        For each traversed edge i->j, compare reward[j] with the best reward
        currently reachable from i. This detects stale but non-catastrophic
        memory: routes that still pay something, yet leave better reachable
        alternatives unused.
        """
        traffic = np.asarray(traffic, dtype=np.float64)
        node_reward = np.asarray(node_reward, dtype=np.float64)
        N = self.topo.N
        if traffic.shape != (N, N):
            raise ValueError(f"traffic must have shape ({N},{N}). Got {traffic.shape}")
        if node_reward.shape != (N,):
            raise ValueError(f"node_reward must have shape ({N},). Got {node_reward.shape}")
        if np.any(~np.isfinite(traffic)) or np.any(traffic < 0):
            raise ValueError("traffic must be finite and non-negative")
        if np.any(~np.isfinite(node_reward)):
            raise ValueError("node_reward must be finite")

        admissible = self.topo.adjacency_mask
        masked_traffic = np.where(admissible, traffic, 0.0)
        reward_by_source = np.where(
            admissible,
            node_reward[np.newaxis, :],
            -np.inf,
        )
        best_reward = np.max(reward_by_source, axis=1)
        best_reward = np.where(np.isfinite(best_reward), best_reward, node_reward)
        edge_regret = np.maximum(
            0.0,
            best_reward[:, np.newaxis] - node_reward[np.newaxis, :],
        )
        edge_regret = np.where(admissible, edge_regret, 0.0)
        regret_mass = masked_traffic * edge_regret
        total_traffic = float(masked_traffic.sum())
        total_regret = float(regret_mass.sum())
        destination_regret = (
            regret_mass.sum(axis=0) / total_traffic
            if total_traffic > 0
            else np.zeros(N, dtype=np.float64)
        )
        stale_flow = masked_traffic[edge_regret > 0].sum()
        return {
            "edge_regret": edge_regret,
            "regret_mass": regret_mass,
            "destination_regret": destination_regret,
            "total_traffic": total_traffic,
            "total_regret": total_regret,
            "mean_opportunity_cost": total_regret / total_traffic if total_traffic > 0 else 0.0,
            "stale_flow_share": float(stale_flow / total_traffic) if total_traffic > 0 else 0.0,
        }

    def memory_law_step(
        self,
        traffic: np.ndarray,
        node_reward: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Apply one step of the configured memory law.

        Parameters
        ----------
        traffic : np.ndarray, shape (N, N)
            Non-negative edge traversal mass for this step (counts or
            probability mass).
        node_reward : np.ndarray, shape (N,), optional
            Realized reward at each destination this step. Required for
            reward_gated and adaptive_eta modes.
        """
        law = self._memory_law
        if law is None:
            return {"mode": "static", "traffic_mass": 0.0, "deposit_mass": 0.0}
        traffic = np.asarray(traffic, dtype=np.float64)
        if traffic.shape != (self.topo.N, self.topo.N):
            raise ValueError(
                f"traffic must have shape ({self.topo.N},{self.topo.N}). "
                f"Got {traffic.shape}"
            )
        if np.any(~np.isfinite(traffic)) or np.any(traffic < 0):
            raise ValueError("traffic must be finite and non-negative")
        traffic = np.where(self.topo.adjacency_mask, traffic, 0.0)

        mode = law["mode"]
        if mode in ("reward_gated", "adaptive_eta"):
            if node_reward is None:
                raise ValueError(f"mode {mode!r} requires node_reward")
            node_reward = np.asarray(node_reward, dtype=np.float64)
            if node_reward.shape != (self.topo.N,):
                raise ValueError(
                    f"node_reward must have shape ({self.topo.N},). "
                    f"Got {node_reward.shape}"
                )
            if np.any(~np.isfinite(node_reward)):
                raise ValueError("node_reward must be finite")
            deposit = law["rho"] * traffic * node_reward[np.newaxis, :]
        else:
            deposit = law["rho"] * traffic

        if mode == "adaptive_eta":
            visits = traffic.sum(axis=0)               # (N,) inbound mass
            total = visits.sum()
            weight = visits / total if total > 0 else np.zeros_like(visits)
            surprise = np.maximum(0.0, self._reward_expectation - node_reward)
            eta_eff = np.clip(
                law["eta"] + law["surprise_gain"] * weight * surprise,
                law["eta"],
                law["eta_max"],
            )
            opp_diag = None
            if law.get("opportunity_gain", 0.0) > 0.0:
                opp_diag = self.opportunity_cost_diagnostic(traffic, node_reward)
                eta_eff = np.clip(
                    eta_eff + law["opportunity_gain"] * opp_diag["destination_regret"],
                    law["eta"],
                    law["eta_max"],
                )
            self._last_opportunity_diagnostic = opp_diag
            self._reward_expectation += (
                law["reward_track_rate"] * weight
                * (node_reward - self._reward_expectation)
            )
            eta_row = eta_eff[np.newaxis, :]            # per-destination column
        else:
            eta_eff = np.full(self.topo.N, law["eta"])
            eta_row = law["eta"]
            self._last_opportunity_diagnostic = None
        self._last_eta_effective = np.broadcast_to(
            eta_eff, (self.topo.N,)
        ).copy()

        if law["channel"] == "friction":
            tensor, baseline = self._sponsor_friction, self._friction_baseline
        else:
            tensor, baseline = self._beta, self._beta_baseline
        delta = tensor - baseline
        delta = delta * (1.0 - eta_row) + deposit
        if law["channel"] == "friction":
            self._sponsor_friction = baseline + delta
        else:
            self._beta = baseline + delta
        return {
            "mode": mode,
            "channel": law["channel"],
            "traffic_mass": float(traffic.sum()),
            "deposit_mass": float(deposit.sum()),
            "mean_eta_effective": float(np.mean(self._last_eta_effective)),
            "max_eta_effective": float(np.max(self._last_eta_effective)),
            "opportunity_cost": None if self._last_opportunity_diagnostic is None else {
                "mean_opportunity_cost": float(
                    self._last_opportunity_diagnostic["mean_opportunity_cost"]
                ),
                "stale_flow_share": float(
                    self._last_opportunity_diagnostic["stale_flow_share"]
                ),
                "total_regret": float(
                    self._last_opportunity_diagnostic["total_regret"]
                ),
            },
        }

    def memory_law_state(self) -> dict:
        """Diagnostic snapshot of the memory law (mode, expectations, eta)."""
        if self._memory_law is None:
            return {"mode": "static"}
        return {
            **self._memory_law,
            "reward_expectation": (
                None
                if self._reward_expectation is None
                else self._reward_expectation.copy()
            ),
            "last_eta_effective": (
                None
                if self._last_eta_effective is None
                else self._last_eta_effective.copy()
            ),
            "last_opportunity_diagnostic": (
                None
                if self._last_opportunity_diagnostic is None
                else {
                    "destination_regret": self._last_opportunity_diagnostic[
                        "destination_regret"
                    ].copy(),
                    "mean_opportunity_cost": self._last_opportunity_diagnostic[
                        "mean_opportunity_cost"
                    ],
                    "stale_flow_share": self._last_opportunity_diagnostic[
                        "stale_flow_share"
                    ],
                    "total_regret": self._last_opportunity_diagnostic[
                        "total_regret"
                    ],
                }
            ),
        }

    # -------------------------------------------------------------------
    # Node bias API — control baseline attractiveness
    # -------------------------------------------------------------------

    def set_node_bias(self, node_idx: int, bias: float) -> None:
        """Set baseline attractiveness for a specific node."""
        self._node_bias[node_idx] = bias

    def set_all_node_bias(self, bias: np.ndarray) -> None:
        """Set baseline attractiveness for all nodes at once."""
        self._node_bias[:] = np.asarray(bias)

    # -------------------------------------------------------------------
    # Feedback
    # -------------------------------------------------------------------

    def _default_feedback(
        self, telemetry: np.ndarray, node_features: np.ndarray
    ) -> np.ndarray:
        """
        Default environment->telemetry feedback: exponential moving blend
        with entropy injection to prevent fixed-point lock-in.

        a_{t+1} = (1 - lam) * a_t + lam * N_visited + eps * noise

        Then re-normalize so the telemetry stays on the unit sphere.
        The noise term ensures the agent never fully converges to a
        deterministic attractor, maintaining exploratory capacity.
        """
        lam = self.feedback_rate
        new_t = (1.0 - lam) * telemetry + lam * node_features

        # Entropy injection: prevent fixed-point convergence
        if self._feedback_noise > 0:
            noise = self._draw_normal((len(new_t),)) * self._feedback_noise
            new_t += noise

        # Re-normalize to unit sphere
        norm = np.linalg.norm(new_t)
        if norm > 0:
            new_t = new_t / norm
        return new_t

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def _print_step(self, step_num, prev, next_node, P, telemetry):
        labels = self.topo.labels
        row = P[prev]
        nonzero = np.where(row > 0)[0]

        print(f"  Step {step_num}: {labels[prev]}")
        for j in nonzero:
            marker = " ***" if j == next_node else ""
            # ASCII arrow instead of Unicode to avoid encoding issues
            print(f"    -> {labels[j]:25s}  P={row[j]:.4f}{marker}")
        entropy = self.row_entropy(row)
        print(f"    Moved to: {labels[next_node]}")
        print(f"    Telemetry now: {np.round(telemetry, 4)}")
        print(f"    Row entropy:   {entropy:.4f} bits\n")

    def row_entropy(self, row: np.ndarray) -> float:
        """
        Shannon entropy of a probability distribution (in bits).
        H = -sum(p * log2(p)) for p > 0.

        Higher entropy = more uniform/exploratory transitions.
        Lower entropy = more deterministic/concentrated transitions.
        """
        p = row[row > 0]
        if len(p) == 0:
            return 0.0
        return -np.sum(p * np.log2(p))

    def transition_entropy(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Compute Shannon entropy for each row of the transition matrix.

        Returns
        -------
        H : np.ndarray, shape (N,)
            Entropy in bits for each source node's transition distribution.
        """
        P = self.transition_matrix(telemetry)
        plogp = np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0)), 0.0)
        return -plogp.sum(axis=1)

    def effective_rank(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Effective number of reachable destinations per node, defined as
        2^H where H is the Shannon entropy. A row with uniform probability
        over k neighbors has effective rank k.

        Returns
        -------
        ranks : np.ndarray, shape (N,)
        """
        H = self.transition_entropy(telemetry)
        return np.power(2.0, H)

    def mixing_time_estimate(
        self,
        telemetry: np.ndarray,
        method: str = "tv",
        threshold: float = 0.25,
        horizon: int = 2000,
    ) -> float:
        """
        Estimate the mixing time of the frozen-telemetry Markov chain.

        method="tv" (default)
            Exact total-variation decay from the worst-case deterministic
            start:  t_mix = min{ t : max_i TV(P^t(i, .), pi) <= threshold }.
            Valid for non-reversible chains — which these generally are:
            flow_diagnostic measures nonzero irreversible edge current on
            the same matrices. Returns inf if the threshold is not reached
            within `horizon` steps; genuinely periodic or reducible chains
            do not converge in TV, and inf is the honest answer for the
            raw (non-lazy) chain.

        method="spectral"
            Legacy heuristic 1 / (1 - |lambda_2|). This is a rigorous
            relaxation-time bound only for REVERSIBLE chains; for
            non-normal P it ignores transient amplification, and on
            periodic structure it conflates periodicity with trapping.
            Retained for comparability with earlier design-space sweeps.

        Dangling rows are treated as absorbing self-states, mirroring
        stationary_distribution.
        """
        P = self.transition_matrix(telemetry)
        P = P.copy()
        row_sums = P.sum(axis=1)
        dangling = row_sums < 1e-12
        if np.any(dangling):
            P[dangling, :] = 0.0
            P[dangling, dangling] = 1.0

        if method == "spectral":
            eigenvalues = np.linalg.eigvals(P)
            mags_sorted = np.sort(np.abs(eigenvalues))[::-1]
            if len(mags_sorted) < 2:
                return np.inf
            spectral_gap = 1.0 - mags_sorted[1]
            if spectral_gap < 1e-12:
                return np.inf
            return 1.0 / spectral_gap

        if method != "tv":
            raise ValueError(f"method must be 'tv' or 'spectral'. Got {method!r}")

        pi = self.stationary_distribution(telemetry)
        M = P.copy()
        for t in range(1, int(horizon) + 1):
            tv_worst = 0.5 * np.max(np.abs(M - pi[np.newaxis, :]).sum(axis=1))
            if tv_worst <= threshold:
                return float(t)
            M = M @ P
        return np.inf

    def stationary_distribution(
        self,
        telemetry: np.ndarray,
        max_iter: int = 10000,
        tol: float = 1e-12,
    ) -> np.ndarray:
        """
        Estimate the stationary distribution pi for the telemetry-conditioned
        Markov chain using power iteration.

        Rows with no outbound probability are treated as absorbing self states
        for the analytic estimate. This avoids NaNs on directed or partially
        disconnected topologies while keeping the original transition matrix
        semantics unchanged for simulation.
        """
        P = self.transition_matrix(np.asarray(telemetry, dtype=np.float64))
        P = P.copy()
        row_sums = P.sum(axis=1)
        dangling = row_sums < 1e-12
        if np.any(dangling):
            P[dangling, :] = 0.0
            P[dangling, dangling] = 1.0

        n = self.topo.N
        pi = np.full(n, 1.0 / n)
        for _ in range(max_iter):
            next_pi = pi @ P
            total = next_pi.sum()
            if total > 0:
                next_pi = next_pi / total
            if np.linalg.norm(next_pi - pi, ord=1) < tol:
                return next_pi
            pi = next_pi
        return pi

    def stationary_distribution_direct(
        self,
        telemetry: np.ndarray,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """
        Solve pi P = pi directly with the normalization constraint sum(pi)=1.

        This is faster than power iteration for small dense matrices.  If the
        solve is ill-conditioned or returns an invalid probability vector, it
        falls back to the power-iteration estimator to preserve robustness on
        reducible or nearly reducible topologies.
        """
        P = self.transition_matrix(np.asarray(telemetry, dtype=np.float64))
        P = P.copy()
        row_sums = P.sum(axis=1)
        dangling = row_sums < 1e-12
        if np.any(dangling):
            P[dangling, :] = 0.0
            P[dangling, dangling] = 1.0

        n = self.topo.N
        A = P.T - np.eye(n)
        b = np.zeros(n)
        A[-1, :] = 1.0
        b[-1] = 1.0

        try:
            pi = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return self.stationary_distribution(telemetry)

        if (
            not np.all(np.isfinite(pi))
            or np.any(pi < -tol)
            or abs(float(pi.sum()) - 1.0) > tol
            or np.linalg.norm(pi @ P - pi, ord=1) > max(10.0 * tol, 1e-9)
        ):
            return self.stationary_distribution(telemetry)

        pi = np.clip(pi, 0.0, None)
        total = pi.sum()
        if total <= 0:
            return self.stationary_distribution(telemetry)
        return pi / total

    # -------------------------------------------------------------------
    # Intervention leverage field — the continuous form of the
    # choice-point principle (Proposition 1 is the P_ij -> 1 limit)
    # -------------------------------------------------------------------

    def _softmax_lane(
        self, telemetry: np.ndarray, step: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Recompute the pure softmax lane (pre-arbitration) and return
        (P, W_prefloor, tau). Mirrors transition_matrix steps 1-6 without
        the edge-learning arbitration blend, which is what the analytic
        leverage formulas describe.
        """
        F_mat = self.topo.node_features
        D = self.topo.distance_matrix
        alignment = F_mat @ telemetry + self._node_bias
        W = (self.alpha * D) - (self._beta * alignment[np.newaxis, :]) - self._sponsor_friction
        if self._edge_learning is not None and not self._edge_learning_uses_arbitration():
            W = W - self._edge_learning_potential(telemetry=telemetry)
        W_floored = self._soft_floor(W)
        tau = self._resolve_temperature(step)
        neg_W = (-W_floored / tau) + self._neg_inf_mask
        row_max = np.max(neg_W, axis=1, keepdims=True)
        row_max = np.where(np.isfinite(row_max), row_max, 0.0)
        exp_W = np.exp(neg_W - row_max)
        row_sums = np.sum(exp_W, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return exp_W / row_sums, W, tau

    def edge_leverage(
        self, telemetry: np.ndarray, step: Optional[int] = None
    ) -> np.ndarray:
        """
        First-order sensitivity of each admissible transition to its own
        friction intervention:

            L_ij = dP_ij / dS_ij
                 = sigma(k (W_ij - floor)) * P_ij (1 - P_ij) / tau

        P_ij(1-P_ij) is the Bernoulli variance of the edge under softmax;
        the sigmoid factor is the softplus floor gate (below the floor,
        further sponsorship has vanishing effect — dropping this factor
        overestimates leverage on heavily sponsored edges by orders of
        magnitude). The choice-point invariance proposition is the
        P_ij -> 1 limit, where L_ij -> 0: an intervention on the only
        admissible edge cannot move the next-state distribution.

        This is a FROZEN-TELEMETRY object: it screens where interventions
        can act at the current preference state. It does not account for
        agent adaptation, downstream feasibility gates, or the arbitration
        mix when edge learning is active (which scales realized sensitivity
        by 1 - mix). Use it to predict and prioritize; confirm with paired
        adaptive simulation.

        Returns
        -------
        L : np.ndarray, shape (N, N)
            dP_ij/dS_ij on admissible edges, 0 elsewhere. Non-negative:
            sponsoring an edge never lowers its own probability.
        """
        telemetry = np.asarray(telemetry, dtype=np.float64)
        P, W, tau = self._softmax_lane(telemetry, step=step)
        gate = 1.0 / (
            1.0 + np.exp(-self._floor_sharpness * (W - self._weight_floor))
        )
        leverage = gate * P * (1.0 - P) / tau
        return np.where(self.topo.adjacency_mask, leverage, 0.0)

    def stationary_leverage(
        self, telemetry: np.ndarray, step: Optional[int] = None
    ) -> np.ndarray:
        """
        Sensitivity of the stationary distribution to each admissible
        friction intervention, via Schweitzer's perturbation formula.

        With Z = (I - P + 1 pi^T)^(-1) the fundamental matrix,
        dpi^T = pi^T dP Z. A perturbation of S_ij changes only row i:

            dP_ik/dS_ij = sigma_ij P_ij (delta_jk - P_ik) / tau

        so

            dpi/dS_ij = pi_i sigma_ij P_ij (Z_j: - (P Z)_i:) / tau.

        Returns
        -------
        G : np.ndarray, shape (N, N, N)
            G[i, j, :] = d pi / d S_ij for admissible (i, j), else 0.
            Each G[i, j, :] sums to ~0 (pi stays normalized).

        Same frozen-telemetry caveat as edge_leverage; additionally
        inherits the stationary estimator's treatment of dangling rows
        (absorbing-self) on reducible topologies.
        """
        telemetry = np.asarray(telemetry, dtype=np.float64)
        P, W, tau = self._softmax_lane(telemetry, step=step)
        P_chain = P.copy()
        row_sums = P_chain.sum(axis=1)
        dangling = row_sums < 1e-12
        if np.any(dangling):
            P_chain[dangling, :] = 0.0
            P_chain[dangling, dangling] = 1.0
        pi = self.stationary_distribution(telemetry)
        n = self.topo.N
        Z = np.linalg.inv(np.eye(n) - P_chain + np.outer(np.ones(n), pi))
        PZ = P_chain @ Z
        gate = 1.0 / (
            1.0 + np.exp(-self._floor_sharpness * (W - self._weight_floor))
        )
        # coef[i, j] = pi_i * sigma_ij * P_ij / tau
        coef = pi[:, np.newaxis] * gate * P * (1.0 / tau)  # (N, N)
        coef = np.where(self.topo.adjacency_mask, coef, 0.0)
        G = coef[:, :, np.newaxis] * (
            Z[np.newaxis, :, :] - PZ[:, np.newaxis, :]
        )
        return np.where(self.topo.adjacency_mask[:, :, np.newaxis], G, 0.0)

    # -------------------------------------------------------------------
    # Controllability of circulation — the reachable set of stationary
    # shifts under budgeted edge interventions. J is the discrete
    # control-input operator; choice-point invariance is its zero columns.
    # -------------------------------------------------------------------

    def leverage_operator(
        self,
        telemetry: np.ndarray,
        edges: Optional[list] = None,
        step: Optional[int] = None,
    ) -> tuple[np.ndarray, list]:
        """
        Flatten the stationary-leverage Jacobian into a control-input
        operator J of shape (N, m).

        J[:, e] = d pi / d S_e  for the e-th admissible edge, so a friction
        control u in R^m produces first-order shift d pi ~ J u. Two
        structural facts hold exactly and are the basis of the
        controllability theory:

          - column-sum zero: 1^T J[:, e] = 0 for all e (mass conserved),
            so range(J) lies in the simplex tangent T = {v : 1^T v = 0};
          - choice-point columns vanish: if edge e leaves a singleton-
            outdegree node, P_e -> 1, the Bernoulli factor P_e(1-P_e) -> 0,
            and J[:, e] = 0. Proposition 1 is exactly col_e(J) = 0.

        Parameters
        ----------
        edges : list of (i, j) or None
            Actuator edges, in the returned order. None uses all admissible
            edges in row-major order.

        Returns
        -------
        J : np.ndarray, shape (N, m)
        edge_list : list of (int, int)
            The edge for each column, in order.
        """
        G = self.stationary_leverage(telemetry, step=step)
        if edges is None:
            edge_list = [tuple(map(int, e)) for e in np.argwhere(self.topo.adjacency_mask)]
        else:
            edge_list = [(int(i), int(j)) for (i, j) in edges]
            for (i, j) in edge_list:
                if not self.topo.adjacency_mask[i, j]:
                    raise ValueError(f"edge ({i},{j}) is not admissible")
        J = np.stack([G[i, j] for (i, j) in edge_list], axis=1) if edge_list else \
            np.zeros((self.topo.N, 0))
        return J, edge_list

    def controllability_gramian(
        self,
        telemetry: np.ndarray,
        edges: Optional[list] = None,
        step: Optional[int] = None,
    ) -> dict:
        """
        Stationary-response Gram matrix W_c = J J^T of the frozen-telemetry
        chain. (Named to avoid confusion with the classical LTI controllability
        Gramian, which describes temporal reachability of a dynamical system;
        this is a static equilibrium-response object — the Gram matrix of the
        leverage columns, whose range is the first-order reachable set.)

        W_c is symmetric PSD and acts on the simplex tangent T. Its
        eigenvectors are the principal reachable directions of stationary-
        distribution motion, ordered by eigenvalue (how cheaply mass moves
        that way per unit L2 budget). The rank of W_c equals rank(J): the
        first-order stationary-reachable dimension. Full reachability
        of simplex-tangent targets is rank(J) = N - 1. The smallest positive
        eigenvalue is the weakest locally reachable stationary direction.

        Returns
        -------
        dict with keys:
            gramian : (N, N) symmetric PSD matrix
            eigenvalues : (N,) descending, clipped at 0
            principal_directions : (N, N) columns are eigenvectors,
                same order as eigenvalues (the top ones are the cheapest
                reachable directions; the smallest nonzero one is the
                structural bottleneck)
            rank : int, numerical rank of J at tol
            controllable_dim : int, N - 1 (target dimension for full control)
        """
        J, _ = self.leverage_operator(telemetry, edges=edges, step=step)
        W_c = J @ J.T
        # symmetric eigendecomposition; W_c is PSD up to roundoff
        evals, evecs = np.linalg.eigh(W_c)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        evecs = evecs[:, order]
        rank = int(np.linalg.matrix_rank(J, tol=1e-9)) if J.shape[1] else 0
        return {
            "gramian": W_c,
            "eigenvalues": evals,
            "principal_directions": evecs,
            "rank": rank,
            "controllable_dim": self.topo.N - 1,
        }

    def reachable_min_budget(
        self,
        telemetry: np.ndarray,
        target: np.ndarray,
        channel: str = "mixed",
        edges: Optional[list] = None,
        step: Optional[int] = None,
        enforce_simplex: bool = True,
    ) -> dict:
        """
        Minimum-L1-budget edge intervention reaching a target first-order
        stationary shift, as a linear program:

            minimize  ||u||_1
            s.t.      J u = target,
                      u in U (channel),
                      pi + J u >= 0  (simplex feasibility, if enforced).

        channel:
            "mixed"   : u unconstrained (friction may rise or fall).
            "penalty" : u >= 0 (friction only increases; penalize edges).
            "subsidy" : u <= 0 (friction only decreases; subsidize edges).

        The target must lie in the simplex tangent (1^T target = 0) since
        range(J) does; a nonzero component along 1 is unreachable and
        reported as infeasible. The returned support is the constructive
        design: which edges to actuate and by how much.

        Returns
        -------
        dict with keys:
            feasible : bool
            budget : float (L1 norm of u, inf if infeasible)
            control : (m,) friction deltas per actuator edge
            support : list of (edge, delta) for |delta| > tol
            residual : ||J u - target||_2
            edge_list : actuator edges in column order
        """
        from scipy.optimize import linprog

        J, edge_list = self.leverage_operator(telemetry, edges=edges, step=step)
        N, m = J.shape
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (N,):
            raise ValueError(f"target must have shape ({N},). Got {target.shape}")
        if channel not in ("mixed", "penalty", "subsidy"):
            raise ValueError("channel must be 'mixed', 'penalty', or 'subsidy'")

        infeasible = {
            "feasible": False,
            "budget": float("inf"),
            "control": np.zeros(m),
            "support": [],
            "residual": float("inf"),
            "edge_list": edge_list,
        }
        # target must be simplex-tangent to be in range(J)
        if abs(float(target.sum())) > 1e-8:
            return infeasible
        if m == 0:
            return infeasible

        pi = self.stationary_distribution(telemetry)

        # Split u = u+ - u- (both >= 0) to linearize the L1 objective.
        # For penalty/subsidy channels one half is pinned to zero.
        obj = np.ones(2 * m)
        A_eq = np.hstack([J, -J])
        b_eq = target
        bounds = []
        for _ in range(m):  # u+
            bounds.append((0.0, None))
        for _ in range(m):  # u-
            bounds.append((0.0, None))
        if channel == "penalty":
            bounds = [(0.0, None)] * m + [(0.0, 0.0)] * m
        elif channel == "subsidy":
            bounds = [(0.0, 0.0)] * m + [(0.0, None)] * m

        A_ub = b_ub = None
        if enforce_simplex:
            # pi + J u >= 0  ->  -J u <= pi  ->  -J(u+ - u-) <= pi
            A_ub = np.hstack([-J, J])
            b_ub = pi

        res = linprog(
            obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
            bounds=bounds, method="highs",
        )
        if not res.success:
            return infeasible
        u = res.x[:m] - res.x[m:]
        residual = float(np.linalg.norm(J @ u - target))
        if residual > 1e-6:
            return infeasible
        tol = 1e-9
        support = [
            (edge_list[e], float(u[e])) for e in range(m) if abs(u[e]) > tol
        ]
        return {
            "feasible": True,
            "budget": float(np.abs(u).sum()),
            "control": u,
            "support": support,
            "residual": residual,
            "edge_list": edge_list,
        }

    def linearization_validity_radius(
        self,
        telemetry: np.ndarray,
        control: np.ndarray,
        eps: float = 0.01,
        edges: Optional[list] = None,
        step: Optional[int] = None,
        max_scale: float = 5.0,
        n_samples: int = 40,
    ) -> dict:
        """
        Largest scaling t of a friction control for which the first-order
        prediction pi + t J u stays within eps (L1) of the exact stationary
        distribution pi(S + t u).

        This bounds the linear regime honestly: below the returned radius
        the analytic reachable set is trustworthy; above it, paired
        simulation must take over. The friction perturbation is applied on
        the actuator edges and reverted (no persistent kernel mutation).

        Returns
        -------
        dict:
            radius : float (largest validated t, 0 if even the smallest
                sampled step already exceeds eps)
            samples : list of (t, l1_error)
        """
        J, edge_list = self.leverage_operator(telemetry, edges=edges, step=step)
        control = np.asarray(control, dtype=np.float64)
        pi0 = self.stationary_distribution(telemetry)
        predicted_dir = J @ control
        saved = self._sponsor_friction.copy()
        samples = []
        radius = 0.0
        try:
            for t in np.linspace(max_scale / n_samples, max_scale, n_samples):
                for e, (i, j) in enumerate(edge_list):
                    self._sponsor_friction[i, j] = saved[i, j] + t * control[e]
                pi_exact = self.stationary_distribution(telemetry)
                pi_pred = pi0 + t * predicted_dir
                l1 = float(np.abs(pi_exact - pi_pred).sum())
                samples.append((float(t), l1))
                if l1 <= eps:
                    radius = float(t)
                else:
                    break
        finally:
            self._sponsor_friction = saved
        return {"radius": radius, "samples": samples}

    def flow_diagnostic(
        self,
        telemetry: np.ndarray,
        eps: float = 1e-12,
    ) -> dict:
        """
        Non-equilibrium flow diagnostics for the frozen-telemetry chain.

        edge_flux[i,j] is the stationary path-space mass pi_i P_ij.
        edge_current[i,j] is the antisymmetric circulation
        pi_i P_ij - pi_j P_ji.  entropy_production is finite-only:
        irreversible one-way mass is reported separately instead of turning
        the scalar into inf for directed support.
        """
        P = self.transition_matrix(telemetry)
        pi = self.stationary_distribution(telemetry)
        flux = pi[:, np.newaxis] * P
        reverse_flux = flux.T
        current = flux - reverse_flux

        bidirectional = (flux > eps) & (reverse_flux > eps)
        entropy_terms = np.zeros_like(flux)
        entropy_terms[bidirectional] = (
            flux[bidirectional]
            * np.log(flux[bidirectional] / reverse_flux[bidirectional])
        )
        entropy_production = 0.5 * float(np.sum(entropy_terms))

        one_way = (flux > eps) & (reverse_flux <= eps)
        irreversible_flux = float(np.sum(flux[one_way]))

        return {
            "edge_flux": flux,
            "edge_current": current,
            "entropy_production": entropy_production,
            "irreversible_flux": irreversible_flux,
        }

    def get_diagnostic(self, telemetry: np.ndarray) -> dict:
        """
        Return a comprehensive diagnostic snapshot for a given telemetry.
        """
        alignment = self.topo.node_features @ telemetry + self._node_bias
        A_broadcast = alignment[np.newaxis, :]
        W_raw = (
            (self.alpha * self.topo.distance_matrix)
            - (self._beta * A_broadcast)
            - self._sponsor_friction
        )
        W = self._soft_floor(W_raw)
        P = self.transition_matrix(telemetry)
        H = self.transition_entropy(telemetry)
        eff_rank = self.effective_rank(telemetry)
        mixing = self.mixing_time_estimate(telemetry)
        flow = self.flow_diagnostic(telemetry)

        return {
            "alignment": alignment,
            "weight_matrix_raw": W_raw,
            "weight_matrix": W,
            "transition_matrix": P,
            "beta_tensor": self._beta.copy(),
            "sponsor_friction": self._sponsor_friction.copy(),
            "node_bias": self._node_bias.copy(),
            "row_entropy": H,
            "effective_rank": eff_rank,
            "mixing_time": mixing,
            "temperature": self.temperature,
            "edge_flux": flow["edge_flux"],
            "edge_current": flow["edge_current"],
            "entropy_production": flow["entropy_production"],
            "irreversible_flux": flow["irreversible_flux"],
        }


# ===========================================================================
# Factory: build topologies from common formats
# ===========================================================================

def topology_from_edges(
    nodes: dict[str, np.ndarray],
    edges: list[tuple[str, str, float]],
    undirected: bool = True,
) -> Topology:
    """
    Convenience constructor.

    Parameters
    ----------
    nodes : dict mapping label -> feature vector
    edges : list of (label_a, label_b, base_distance)
    undirected : bool, default True
        If True, edges are mirrored: (A→B) also creates (B→A) with the
        same cost. If False, edges are one-way as specified — enabling
        directed topologies (one-way corridors, escalators, pipelines,
        supply chains where back-flow is physically impossible).
    """
    labels = list(nodes.keys())
    N = len(labels)
    F = len(next(iter(nodes.values())))
    idx = {label: i for i, label in enumerate(labels)}

    features = np.zeros((N, F))
    for label, vec in nodes.items():
        features[idx[label]] = vec

    D = np.full((N, N), np.inf)
    np.fill_diagonal(D, 0.0)
    for a, b, dist in edges:
        i, j = idx[a], idx[b]
        D[i, j] = dist
        if undirected:
            D[j, i] = dist  # mirror for undirected graphs

    return Topology(
        node_features=features,
        distance_matrix=D,
        labels=labels,
    )


# ===========================================================================
# Demo: Mall scenario
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # --- Build the mall topology ---
    topo = topology_from_edges(
        nodes={
            "Entrance":          np.array([0.1, 0.1, 0.1]),  # weakly attracts all
            "Food Court":        np.array([0.1, 0.9, 0.0]),
            "Tech Store":        np.array([0.0, 0.1, 0.9]),
            "Premium Outlet A":  np.array([0.8, 0.0, 0.1]),
            "Premium Outlet B":  np.array([0.9, 0.1, 0.0]),
        },
        edges=[
            ("Entrance", "Food Court", 5.0),
            ("Entrance", "Tech Store", 4.0),
            ("Entrance", "Premium Outlet A", 10.0),
            ("Food Court", "Premium Outlet B", 6.0),
            ("Premium Outlet A", "Premium Outlet B", 3.0),
            ("Tech Store", "Premium Outlet A", 7.0),
        ],
    )

    # Give Entrance a baseline bias so it's not a black hole
    kernel = DynamicTopologyKernel(
        topology=topo,
        alpha=1.0,
        beta=np.full((5, 5), 5.0),
        feedback_rate=0.15,
        temperature=1.0,
        feedback_noise=0.02,
        node_bias=np.array([0.3, 0.0, 0.0, 0.0, 0.0]),  # Entrance gets bias
    )

    print("=" * 60)
    print("  SCENARIO 1: Fashion-biased agent (with exploration)")
    print("=" * 60)

    agent = AgentState(
        telemetry=np.array([1.0, 0.0, 0.0]),
        position=0,  # Entrance
    )
    path = kernel.simulate(agent, steps=4, verbose=True)
    labels = topo.labels
    print(f"  Path: {' -> '.join(labels[i] for i in path)}\n")

    # --- Scenario 2: Same agent, but Food Court sponsors aggressively ---
    print("=" * 60)
    print("  SCENARIO 2: Food Court sponsors the Entrance->Food Court edge")
    print("=" * 60)

    kernel.sponsor_edge(0, 1, boost=8.0)  # Entrance -> Food Court

    agent2 = AgentState(
        telemetry=np.array([1.0, 0.0, 0.0]),
        position=0,
    )
    path2 = kernel.simulate(agent2, steps=4, verbose=True)
    print(f"  Path: {' -> '.join(labels[i] for i in path2)}\n")

    # --- Scenario 3: Batch simulation ---
    print("=" * 60)
    print("  SCENARIO 3: Batch -- 1000 agents, distribution analysis")
    print("=" * 60)

    K = 1000
    rng = np.random.default_rng(42)
    raw = rng.dirichlet(np.ones(3), size=K)
    starts = np.zeros(K, dtype=int)

    import time
    t0 = time.perf_counter()
    all_paths = kernel.simulate_batch(raw, starts, steps=5)
    t1 = time.perf_counter()

    final_nodes = all_paths[:, -1]
    print(f"\n  Final destination distribution ({K} agents, 5 steps):")
    for i, label in enumerate(labels):
        count = np.sum(final_nodes == i)
        print(f"    {label:25s}: {count:4d}  ({count/K*100:5.1f}%)")
    print(f"\n  Batch time: {t1-t0:.3f}s")

    # --- Diagnostic snapshot ---
    print(f"\n{'='*60}")
    print("  DIAGNOSTIC: Transition matrix for pure-fashion telemetry")
    print("=" * 60)
    diag = kernel.get_diagnostic(np.array([1.0, 0.0, 0.0]))
    P = diag["transition_matrix"]
    print(f"\n  Alignment vector: {np.round(diag['alignment'], 3)}")
    print(f"  Row entropy (bits): {np.round(diag['row_entropy'], 3)}")
    print(f"  Effective rank:     {np.round(diag['effective_rank'], 2)}")
    print(f"  Mixing time est.:   {diag['mixing_time']:.1f} steps")
    print(f"\n  Transition matrix P:")
    header = "  " + " " * 22 + "".join(f"{l:>14s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row_str = "".join(f"{P[i,j]:14.4f}" for j in range(len(labels)))
        print(f"  {label:20s} {row_str}")


# Public brand alias — the package is presented as the "Dynamic Circulator".
# The class name is retained for API/test stability.
DynamicCirculator = DynamicTopologyKernel
