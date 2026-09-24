"""Dynamic traffic generation system for network links.

Implements an Ornstein-Uhlenbeck (OU) mean-reverting stochastic process
combined with sinusoidal diurnal background variations and Poisson traffic bursts.
Strictly units-consistent: traffic load is generated in Gigabits per second (Gbps).
Includes explicit seed threading for bit-exact reproducibility.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np


class TrafficGenerator:
    """Generates continuous-time style stochastic traffic demands over discrete timesteps.

    All generated loads are strictly in Gigabits per second (Gbps).

    Attributes:
        edges: List of undirected edge tuples (u, v).
        link_capacities: Mapping of edge -> nominal capacity in Gbps.
        theta: Mean-reversion speed for the OU process.
        sigma: Volatility / noise magnitude of the OU process.
        dt: Discrete timestep increment (e.g., 1.0).
        burst_prob: Probability of an anomalous traffic burst at any timestep.
        burst_magnitude_range: Min and max burst magnitude in Gbps.
        seed: Random seed for reproducible generation.
    """

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        link_capacities: Optional[Dict[Tuple[int, int], float]] = None,
        default_capacity_gbps: float = 10.0,
        mean_utilization: float = 0.45,
        theta: float = 0.25,
        sigma: float = 0.80,
        dt: float = 1.0,
        burst_prob: float = 0.08,
        burst_range_gbps: Tuple[float, float] = (1.5, 4.5),
        seed: Optional[int] = None,
    ):
        self.edges = [(min(u, v), max(u, v)) for (u, v) in edges]
        self.default_capacity_gbps = default_capacity_gbps
        self.link_capacities = (
            { (min(u, v), max(u, v)): cap for (u, v), cap in link_capacities.items() }
            if link_capacities is not None
            else { e: default_capacity_gbps for e in self.edges }
        )
        self.mean_utilization = mean_utilization
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.burst_prob = burst_prob
        self.burst_range_gbps = burst_range_gbps
        
        self.initial_seed = seed
        self.rng = np.random.default_rng(seed)

        # Baseline mean load per edge (Gbps)
        self.base_means: Dict[Tuple[int, int], float] = {
            e: self.link_capacities[e] * self.mean_utilization
            for e in self.edges
        }

        # Current traffic state (Gbps)
        self.current_loads: Dict[Tuple[int, int], float] = {}
        self.manual_spikes: Dict[Tuple[int, int], float] = {}
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> Dict[Tuple[int, int], float]:
        """Resets the random number generator and link traffic loads to baseline."""
        if seed is not None:
            self.initial_seed = seed
            self.rng = np.random.default_rng(seed)
        elif self.initial_seed is not None:
            # Re-seed with initial seed to reset run deterministically
            self.rng = np.random.default_rng(self.initial_seed)

        self.current_loads = {}
        self.manual_spikes = {}

        for e in self.edges:
            # Initialize with Gaussian jitter around mean baseline
            init_val = self.rng.normal(self.base_means[e], scale=0.5)
            cap = self.link_capacities[e]
            self.current_loads[e] = float(np.clip(init_val, 0.1 * cap, 0.85 * cap))

        return self.get_current_loads()

    def inject_spike(self, u: int, v: int, extra_load_gbps: float) -> None:
        """Injects a manual traffic surge onto link (u, v) in Gbps for the next step."""
        e = (min(u, v), max(u, v))
        self.manual_spikes[e] = self.manual_spikes.get(e, 0.0) + extra_load_gbps

    def step(self, t: int = 0) -> Dict[Tuple[int, int], float]:
        """Advances traffic state by one discrete timestep using OU dynamics + bursts.

        Args:
            t: Discrete timestep index (used for diurnal wave periodicity).

        Returns:
            Dict mapping edge (min(u, v), max(u, v)) -> traffic load in Gbps.
        """
        # Diurnal sinusoidal modulation factor (period = 40 steps)
        diurnal_factor = 1.0 + 0.15 * np.sin(2.0 * np.pi * t / 40.0)

        for e in self.edges:
            current = self.current_loads[e]
            cap = self.link_capacities[e]
            mean_target = self.base_means[e] * diurnal_factor

            # Ornstein-Uhlenbeck drift + diffusion
            drift = self.theta * (mean_target - current) * self.dt
            diffusion = self.sigma * np.sqrt(self.dt) * self.rng.standard_normal()

            # Check for random Poisson/Bernoulli burst
            burst = 0.0
            if self.rng.random() < self.burst_prob:
                burst = float(self.rng.uniform(self.burst_range_gbps[0], self.burst_range_gbps[1]))

            # Add any external manual spike
            spike = self.manual_spikes.pop(e, 0.0)

            # Updated load in Gbps
            new_load = current + drift + diffusion + burst + spike
            # Cap strictly between 0.0 and 1.1 * Capacity (to permit occasional saturation)
            self.current_loads[e] = float(np.clip(new_load, 0.0, 1.10 * cap))

        return self.get_current_loads()

    def get_current_loads(self) -> Dict[Tuple[int, int], float]:
        """Returns a copy of the current traffic loads per edge in Gbps."""
        return dict(self.current_loads)
