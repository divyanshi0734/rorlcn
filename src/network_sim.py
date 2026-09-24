"""NSFNET dynamic discrete-timestep simulation environment.

Integrates topology, stochastic traffic generator, and dynamic link-state engine.
Exposes RL-ready observation methods and telemetry recording.
"""

from typing import Dict, Tuple, Optional, Any, List
import pandas as pd
import numpy as np
import networkx as nx

from .topology import build_nsfnet_graph, NSFNET_EDGES
from .traffic_generator import TrafficGenerator
from .dynamic_link_state import DynamicLinkStateEngine


class NSFNETSimEnv:
    """Discrete-timestep network simulation environment for NSFNET routing.

    Attributes:
        graph: NetworkX graph.
        traffic_gen: TrafficGenerator instance.
        link_state_engine: DynamicLinkStateEngine instance.
        timestep: Current discrete timestep counter.
        seed: Master random seed.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        default_capacity_gbps: float = 10.0,
        mean_utilization: float = 0.45,
        theta: float = 0.25,
        sigma: float = 0.80,
        burst_prob: float = 0.08,
    ):
        self.seed = seed
        self.default_capacity_gbps = default_capacity_gbps
        self.mean_utilization = mean_utilization
        self.theta = theta
        self.sigma = sigma
        self.burst_prob = burst_prob

        # 1. Build graph
        self.graph = build_nsfnet_graph(default_capacity_gbps=default_capacity_gbps)
        edges = list(self.graph.edges())

        # 2. Build traffic generator
        capacities = {
            (min(u, v), max(u, v)): self.graph[u][v]["capacity_gbps"]
            for u, v in edges
        }
        self.traffic_gen = TrafficGenerator(
            edges=edges,
            link_capacities=capacities,
            default_capacity_gbps=default_capacity_gbps,
            mean_utilization=mean_utilization,
            theta=theta,
            sigma=sigma,
            burst_prob=burst_prob,
            seed=seed,
        )

        # 3. Build dynamic link-state engine
        self.link_state_engine = DynamicLinkStateEngine(
            graph=self.graph,
            seed=seed,
        )

        self.timestep = 0
        self.current_loads: Dict[Tuple[int, int], float] = {}
        self.routed_loads: Dict[Tuple[int, int], float] = {}
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Resets simulation environment to initial state with optional re-seeding."""
        if seed is not None:
            self.seed = seed

        self.timestep = 0
        initial_loads = self.traffic_gen.reset(seed=self.seed)
        self.current_loads = dict(initial_loads)
        self.routed_loads = {e: 0.0 for e in self.traffic_gen.edges}
        self.link_state_engine.reset()
        self.link_state_engine.update_states(initial_loads)

        global_state = self.link_state_engine.get_global_state_tensor()
        edge_matrix = self.link_state_engine.get_edge_feature_matrix()
        return global_state, edge_matrix

    def step(
        self,
        manual_loads: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Advances simulation by one discrete timestep.

        Args:
            manual_loads: Optional overrides or additive loads per link in Gbps.

        Returns:
            Tuple of:
              - global_state: np.ndarray of shape (14, 14, 4)
              - edge_matrix: np.ndarray of shape (21, 4)
              - info: Dict containing telemetry and link states for this step.
        """
        self.timestep += 1

        # 1. Generate stochastic background traffic loads
        step_loads = self.traffic_gen.step(t=self.timestep)

        # 2. Add any manual loads (e.g. from routing agents or persistent routed flows)
        if manual_loads is not None:
            for (u, v), extra_load in manual_loads.items():
                e = (min(u, v), max(u, v))
                step_loads[e] = step_loads.get(e, 0.0) + extra_load

        # 3. Update link states across network
        self.current_loads = dict(step_loads)
        self.routed_loads = {e: 0.0 for e in self.traffic_gen.edges}
        self.link_state_engine.update_states(step_loads)

        global_state = self.link_state_engine.get_global_state_tensor()
        edge_matrix = self.link_state_engine.get_edge_feature_matrix()

        info = {
            "timestep": self.timestep,
            "loads_gbps": step_loads,
            "mean_congestion": float(np.mean(edge_matrix[:, 1])),
            "max_congestion": float(np.max(edge_matrix[:, 1])),
            "mean_latency_ms": float(np.mean([self.graph[u][v]["latency"] for u, v in self.graph.edges()])),
            "mean_loss_rate": float(np.mean(edge_matrix[:, 3])),
        }

        return global_state, edge_matrix, info

    def apply_packet_load(self, edge_loads: Dict[Tuple[int, int], float]) -> None:
        """Applies additional traffic load (in Gbps) from routed packets to links.

        Recomputes live link metrics (congestion, queuing latency, packet loss rate)
        in-place so that subsequent routing decisions immediately perceive the
        congestion created by routed packets (intra-timestep closed-loop feedback).

        Args:
            edge_loads: Mapping of edge (u, v) -> additional traffic demand in Gbps.
        """
        for (u, v), load_delta in edge_loads.items():
            if load_delta <= 0:
                continue
            e = (min(u, v), max(u, v))
            new_load = self.current_loads.get(e, 0.0) + load_delta
            self.current_loads[e] = new_load
            self.routed_loads[e] = self.routed_loads.get(e, 0.0) + load_delta

            edge_data = self.graph[u][v]
            base_prop = edge_data["base_latency_ms"]
            cap = edge_data["capacity_gbps"]

            cong, avail_norm, avail_gbps, lat, loss = self.link_state_engine.compute_link_metrics(
                base_prop_ms=base_prop,
                capacity_gbps=cap,
                traffic_load_gbps=new_load,
            )

            edge_data["traffic_load_gbps"] = new_load
            edge_data["congestion"] = cong
            edge_data["available_bandwidth"] = avail_norm
            edge_data["available_bandwidth_gbps"] = avail_gbps
            edge_data["latency"] = lat
            edge_data["packet_loss_rate"] = loss

    def inject_traffic_spike(self, u: int, v: int, extra_gbps: float) -> None:
        """Injects a severe traffic surge onto link (u, v) in Gbps."""
        self.traffic_gen.inject_spike(u, v, extra_gbps)

    def get_node_observation(self, node_id: int) -> Dict[str, np.ndarray]:
        """Gets local observation for router at node_id."""
        return self.link_state_engine.get_node_observation(node_id)

    def get_telemetry_dataframe(self) -> pd.DataFrame:
        """Converts complete step history into a tabular pandas DataFrame."""
        records = []
        for snapshot in self.link_state_engine.history:
            t = snapshot["step"]
            for (u, v), metrics in snapshot["edges"].items():
                records.append({
                    "timestep": t,
                    "edge": f"({u},{v})",
                    "src": u,
                    "dst": v,
                    "load_gbps": metrics["load_gbps"],
                    "congestion": metrics["congestion"],
                    "available_bandwidth": metrics["available_bandwidth"],
                    "available_bandwidth_gbps": metrics["available_bandwidth_gbps"],
                    "latency_ms": metrics["latency"],
                    "packet_loss_rate": metrics["packet_loss_rate"],
                })
        return pd.DataFrame(records)
