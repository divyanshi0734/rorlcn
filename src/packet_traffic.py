"""Seeded Packet Traffic Generator and Closed-Loop Simulation Runner.

================================================================================
ARCHITECTURAL ASYMMETRY & CLOSED-LOOP FEEDBACK:
--------------------------------------------------------------------------------
1. Closed-Loop Link Load Feedback Loop:
   - Every routed packet of demand D_pkt (Gbps) traversing path P adds D_pkt load
     to each link e in P:
       load_e(t) = load_{e, OU}(t) + sum_{pkts traversing e} D_pkt
   - Both intra-step dynamic accumulation (where packets routed earlier in the step
     increase link latency, prompting subsequent packets to dynamically detour)
     and inter-step persistence are supported.

2. Dijkstra vs. Future RL Agent Decision Scope:
   - Dijkstra evaluates full-path global snapshots at the controller/source.
   - The future RL agent will execute decentralized per-hop sequential decisions.
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import networkx as nx

from .dijkstra import DijkstraRouter
from .network_sim import NSFNETSimEnv


@dataclass
class Packet:
    """Represents a discrete data packet or flow demand injected into the network.

    Attributes:
        packet_id: Unique monotonic sequence identifier.
        timestep: Discrete simulation timestep when packet was generated.
        src: Source router node ID.
        dst: Destination router node ID (src != dst).
        size_bytes: Transmission payload size in bytes (default 1500 B Ethernet MTU).
        demand_gbps: Traffic demand rate on links in Gbps (default 0.05 Gbps = 50 Mbps).
    """
    packet_id: int
    timestep: int
    src: int
    dst: int
    size_bytes: int = 1500
    demand_gbps: float = 0.05


class PacketTrafficGenerator:
    """Stochastic traffic stream generator producing packets between random (s, d) pairs.

    Attributes:
        nodes: Available network router nodes (e.g. [0..13] for NSFNET).
        packets_per_step: Mean number of packets injected per timestep.
        packet_demand_gbps: Link bandwidth load contributed by each packet flow.
        packet_size_bytes: Nominal packet size in bytes.
        seed: Random seed for deterministic reproducibility.
    """

    def __init__(
        self,
        nodes: Optional[List[int]] = None,
        packets_per_step: int = 50,
        packet_demand_gbps: float = 0.05,
        packet_size_bytes: int = 1500,
        seed: Optional[int] = None,
    ):
        self.nodes = nodes if nodes is not None else list(range(14))
        self.packets_per_step = packets_per_step
        self.packet_demand_gbps = packet_demand_gbps
        self.packet_size_bytes = packet_size_bytes
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.packet_counter = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets generator state, packet counter, and random number generator."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.packet_counter = 0

    def generate_packets(
        self,
        timestep: int,
        count: Optional[int] = None,
    ) -> List[Packet]:
        """Generates a batch of packets for the current discrete timestep.

        Source and destination nodes are sampled uniformly at random with s != d.

        Args:
            timestep: Current simulation timestep.
            count: Optional override for packet count (defaults to self.packets_per_step).

        Returns:
            List of Packet instances.
        """
        num_packets = count if count is not None else self.packets_per_step
        if num_packets <= 0 or len(self.nodes) < 2:
            return []

        packets = []
        for _ in range(num_packets):
            # Sample source and destination without replacement
            src, dst = self.rng.choice(self.nodes, size=2, replace=False)
            pkt = Packet(
                packet_id=self.packet_counter,
                timestep=timestep,
                src=int(src),
                dst=int(dst),
                size_bytes=self.packet_size_bytes,
                demand_gbps=self.packet_demand_gbps,
            )
            self.packet_counter += 1
            packets.append(pkt)

        return packets


class RoutingSimulationRunner:
    """Executes closed-loop routing simulations on NSFNETSimEnv with metric logging.

    Supports both Dynamic Dijkstra (recomputed fresh live latency) and
    Static Shortest Path First (static distance / base latency) baseline routing.
    """

    def __init__(
        self,
        env: NSFNETSimEnv,
        packet_gen: Optional[PacketTrafficGenerator] = None,
        router: Optional[DijkstraRouter] = None,
        rl_model: Optional[Any] = None,
        enable_packet_feedback: bool = True,
        seed: Optional[int] = None,
        max_hops: int = 10,
        enable_split_horizon: Optional[bool] = None,
        loop_avoidance_mode: str = "split_horizon",
        periodic_interval: int = 2,
    ):
        self.env = env
        self.packet_gen = packet_gen if packet_gen is not None else PacketTrafficGenerator(seed=seed)
        self.router = router if router is not None else DijkstraRouter(default_weight_attr="latency")
        self.rl_model = rl_model
        self.enable_packet_feedback = enable_packet_feedback
        self.seed = seed
        self.max_hops = max_hops
        # Resolve loop avoidance mode and backward-compatible enable_split_horizon flag
        if enable_split_horizon is not None:
            if enable_split_horizon:
                self.loop_avoidance_mode = loop_avoidance_mode if loop_avoidance_mode in ["split_horizon", "path_vector"] else "split_horizon"
            else:
                self.loop_avoidance_mode = "none"
        else:
            self.loop_avoidance_mode = loop_avoidance_mode
        self.enable_split_horizon = (self.loop_avoidance_mode in ["split_horizon", "path_vector"])
        self.periodic_interval = max(1, periodic_interval)
        self.rng = np.random.default_rng(seed)

        self.packet_records: List[Dict[str, Any]] = []
        self.timestep_records: List[Dict[str, Any]] = []

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets the environment, packet generator, router, and history."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.env.reset(seed=self.seed)
        self.packet_gen.reset(seed=self.seed)
        self.packet_records = []
        self.timestep_records = []

    def run_simulation(
        self,
        num_timesteps: int = 40,
        routing_mode: str = "dynamic_dijkstra",
        spike_schedule: Optional[Dict[int, Tuple[int, int, float]]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs multi-step routing simulation.

        Args:
            num_timesteps: Total discrete timesteps to simulate.
            routing_mode: 'dynamic_dijkstra', 'static_spf', 'periodic_dijkstra', or 'rl_agent'.
            spike_schedule: Optional mapping of timestep -> (u, v, extra_gbps) to inject spikes.

        Returns:
            Tuple of:
              - per_packet_df: pd.DataFrame with per-packet transmission logs.
              - aggregate_df: pd.DataFrame with per-timestep aggregate statistics.
        """
        stale_graph_snapshot = None

        for t in range(1, num_timesteps + 1):
            # 1. Handle scheduled traffic spikes
            if spike_schedule and t in spike_schedule:
                u, v, extra = spike_schedule[t]
                self.env.inject_traffic_spike(u, v, extra)

            # 2. Advance environment by one discrete timestep (background OU process)
            _, _, step_info = self.env.step()

            # Handle periodic link-state snapshot for periodic_dijkstra
            if routing_mode == "periodic_dijkstra":
                if stale_graph_snapshot is None or ((t - 1) % self.periodic_interval == 0):
                    stale_graph_snapshot = self.env.graph.copy()

            # 3. Generate packet stream for this timestep
            packets = self.packet_gen.generate_packets(timestep=t)

            step_delivered_count = 0
            step_latencies = []
            step_hops = []
            step_delivered_gbps = 0.0

            # 4. Route each packet individually
            for pkt in packets:
                if routing_mode == "dynamic_dijkstra":
                    # Dijkstra recomputes fresh minimum latency path from live link states
                    path, _ = self.router.find_shortest_path(
                        graph=self.env.graph,
                        src=pkt.src,
                        dst=pkt.dst,
                        weight_attr="latency",
                    )
                    metrics = self.router.compute_path_metrics(self.env.graph, path)
                    loss_prob = metrics["path_loss_prob"]
                    delivered = bool(self.rng.random() >= loss_prob) if path else False

                    if self.enable_packet_feedback and len(path) > 1:
                        edge_load_delta = {}
                        for i in range(len(path) - 1):
                            u_node, v_node = path[i], path[i + 1]
                            edge_load_delta[(u_node, v_node)] = pkt.demand_gbps
                        self.env.apply_packet_load(edge_load_delta)

                elif routing_mode == "periodic_dijkstra":
                    # Stale Dijkstra computes shortest path on stale snapshot, evaluated on live graph
                    path, _, metrics = self.router.find_stale_shortest_path(
                        live_graph=self.env.graph,
                        stale_graph=stale_graph_snapshot if stale_graph_snapshot is not None else self.env.graph,
                        src=pkt.src,
                        dst=pkt.dst,
                        weight_attr="latency",
                    )
                    loss_prob = metrics["path_loss_prob"]
                    delivered = bool(self.rng.random() >= loss_prob) if path else False

                    if self.enable_packet_feedback and len(path) > 1:
                        edge_load_delta = {}
                        for i in range(len(path) - 1):
                            u_node, v_node = path[i], path[i + 1]
                            edge_load_delta[(u_node, v_node)] = pkt.demand_gbps
                        self.env.apply_packet_load(edge_load_delta)

                elif routing_mode == "static_spf":
                    # Static SPF routes on physical distance
                    path, _, _ = self.router.find_static_shortest_path(
                        graph=self.env.graph,
                        src=pkt.src,
                        dst=pkt.dst,
                        static_attr="distance_km",
                    )
                    metrics = self.router.compute_path_metrics(self.env.graph, path)
                    loss_prob = metrics["path_loss_prob"]
                    delivered = bool(self.rng.random() >= loss_prob) if path else False

                    if self.enable_packet_feedback and len(path) > 1:
                        edge_load_delta = {}
                        for i in range(len(path) - 1):
                            u_node, v_node = path[i], path[i + 1]
                            edge_load_delta[(u_node, v_node)] = pkt.demand_gbps
                        self.env.apply_packet_load(edge_load_delta)

                elif routing_mode == "rl_agent":
                    if self.rl_model is None:
                        raise ValueError("rl_model must be provided to RoutingSimulationRunner for 'rl_agent' mode")

                    curr = pkt.src
                    path = [curr]
                    delivered = False
                    hops = 0
                    max_hops = self.max_hops
                    num_nodes = self.env.graph.number_of_nodes()
                    prev_node = None

                    while hops < max_hops:
                        node_obs = self.env.get_node_observation(curr)
                        port_mask = node_obs["port_mask"].copy()
                        neighbor_ids = node_obs["neighbor_ids"]

                        # Anti-looping ingress port masking
                        if self.loop_avoidance_mode == "path_vector":
                            # Path-Vector / Source-Routing: mask out all already-visited nodes in path
                            visited_set = set(path)
                            valid_indices = [idx for idx, nid in enumerate(neighbor_ids) if port_mask[idx] == 1.0]
                            unvisited_indices = [idx for idx in valid_indices if neighbor_ids[idx] not in visited_set]
                            if unvisited_indices:
                                for idx in valid_indices:
                                    if neighbor_ids[idx] in visited_set:
                                        port_mask[idx] = 0.0
                        elif (self.loop_avoidance_mode == "split_horizon" or self.enable_split_horizon) and prev_node is not None:
                            # Split-Horizon: mask out immediate ingress port if alternatives exist
                            for p_idx, n_id in enumerate(neighbor_ids):
                                if n_id == prev_node and port_mask[p_idx] == 1.0:
                                    if np.sum(port_mask) > 1.0:
                                        port_mask[p_idx] = 0.0
                                    break


                        action_masks = port_mask.astype(bool)

                        dest_one_hot = np.zeros(num_nodes, dtype=np.float32)
                        dest_one_hot[pkt.dst] = 1.0

                        obs = {
                            "neighbor_features": node_obs["neighbor_features"],
                            "neighbor_ids": node_obs["neighbor_ids"],
                            "port_mask": port_mask,
                            "current_node": np.int64(curr),
                            "destination": np.int64(pkt.dst),
                            "destination_one_hot": dest_one_hot,
                            "hops_taken": np.int64(hops),
                            "action_mask": port_mask.copy(),
                        }

                        action, _ = self.rl_model.predict(obs, action_masks=action_masks, deterministic=True)
                        action = int(action)

                        if action < 0 or action >= 4 or port_mask[action] == 0.0:
                            break

                        v = int(neighbor_ids[action])
                        edge = (min(curr, v), max(curr, v))

                        # Proactive closed-loop packet load feedback
                        if self.enable_packet_feedback:
                            self.env.apply_packet_load({edge: pkt.demand_gbps})

                        edge_data = self.env.graph[curr][v]
                        loss_prob = float(edge_data["packet_loss_rate"])

                        hops += 1
                        path.append(v)

                        # Termination checks
                        if v in path[:-1]:
                            # Looped
                            break
                        elif v == pkt.dst:
                            # Destination reached! Short-circuits drop check
                            delivered = True
                            break
                        else:
                            # Intermediate drop trial
                            if self.rng.random() < loss_prob:
                                break

                        prev_node = curr
                        curr = v

                    metrics = self.router.compute_path_metrics(self.env.graph, path)

                else:
                    raise ValueError(f"Unknown routing_mode: {routing_mode}")

                if delivered:
                    step_delivered_count += 1
                    step_latencies.append(metrics["total_latency_ms"])
                    step_hops.append(metrics["hop_count"])
                    step_delivered_gbps += pkt.demand_gbps

                # Record per-packet telemetry
                self.packet_records.append({
                    "packet_id": pkt.packet_id,
                    "timestep": t,
                    "routing_mode": routing_mode,
                    "src": pkt.src,
                    "dst": pkt.dst,
                    "path": str(path),
                    "hop_count": metrics["hop_count"],
                    "end_to_end_latency_ms": metrics["total_latency_ms"],
                    "base_latency_ms": metrics["base_latency_ms"],
                    "queuing_delay_ms": metrics["queuing_delay_ms"],
                    "path_loss_prob": metrics["path_loss_prob"],
                    "delivered": delivered,
                    "demand_gbps": pkt.demand_gbps,
                    "size_bytes": pkt.size_bytes,
                    "bottleneck_avail_bw_gbps": metrics["bottleneck_avail_bw_gbps"],
                    "max_congestion": metrics["max_congestion"],
                })

            # 6. Aggregate timestep summary
            total_pkts = len(packets)
            loss_rate = (total_pkts - step_delivered_count) / total_pkts if total_pkts > 0 else 0.0
            mean_lat = float(np.mean(step_latencies)) if step_latencies else 0.0
            p95_lat = float(np.percentile(step_latencies, 95)) if step_latencies else 0.0
            mean_hops = float(np.mean(step_hops)) if step_hops else 0.0

            self.timestep_records.append({
                "timestep": t,
                "routing_mode": routing_mode,
                "packets_sent": total_pkts,
                "packets_delivered": step_delivered_count,
                "packets_dropped": total_pkts - step_delivered_count,
                "packet_loss_rate": loss_rate,
                "mean_latency_ms": mean_lat,
                "p95_latency_ms": p95_lat,
                "mean_hops": mean_hops,
                "throughput_gbps": step_delivered_gbps,
                "network_mean_congestion": step_info["mean_congestion"],
                "network_max_congestion": step_info["max_congestion"],
                "network_mean_latency_ms": step_info["mean_latency_ms"],
            })

        return pd.DataFrame(self.packet_records), pd.DataFrame(self.timestep_records)
