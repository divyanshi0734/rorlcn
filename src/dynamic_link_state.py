"""Dynamic link-state tracking engine for network graphs.

Maintains edge metrics:
  - latency (ms) [propagation + safe bounded queuing delay]
  - congestion [0.0, 1.0]
  - available_bandwidth [0.0, 1.0] (and absolute Gbps)
  - packet_loss_rate [0.0, 1.0]

Exports single standardized state tensor representations for RL environments:
  - Global State Tensor: Shape (14, 14, 4)
  - Canonical Edge Matrix: Shape (21, 4)
  - Per-Node Local View: Shape (d_max, 4) with neighbor IDs and port mask
"""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import networkx as nx


class DynamicLinkStateEngine:
    """Engine that computes and updates physical link states across discrete timesteps.

    Feature vector order across all tensors:
      [0] latency_norm: Latency normalized by 100.0 ms, clipped to [0.0, 1.0]
      [1] congestion: Utilization ratio clipped to [0.0, 1.0]
      [2] available_bandwidth: Normalized remaining capacity [0.0, 1.0]
      [3] packet_loss_rate: Non-linear RED buffer drop fraction [0.0, 1.0]
    """

    FEATURE_NAMES = [
        "latency_norm",
        "congestion",
        "available_bandwidth",
        "packet_loss_rate",
    ]
    NUM_FEATURES = len(FEATURE_NAMES)
    LATENCY_NORM_SCALE_MS = 100.0  # Normalization factor for delay (100 ms)

    def __init__(
        self,
        graph: nx.Graph,
        d_max: int = 4,
        loss_threshold: float = 0.70,
        loss_max_rate: float = 0.20,
        queue_d_max: float = 0.50,
        queue_c_max: float = 0.98,
        queue_d_overflow: float = 200.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            graph: NetworkX graph (e.g. NSFNET).
            d_max: Maximum node degree for per-node observation padding (4 for NSFNET).
            loss_threshold: Congestion threshold where packet loss begins.
            loss_max_rate: Maximum packet loss fraction at 100% saturation.
            queue_d_max: Scaling constant for asymptotic queuing delay.
            queue_c_max: Asymptotic clamping point to prevent division by zero.
            queue_d_overflow: Linear slope penalty for saturation beyond queue_c_max.
            seed: Master random seed.
        """
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()
        self.d_max = d_max

        # Queuing and loss model parameters
        self.loss_threshold = loss_threshold
        self.loss_max_rate = loss_max_rate
        self.queue_d_max = queue_d_max
        self.queue_c_max = queue_c_max
        self.queue_d_overflow = queue_d_overflow

        self.initial_seed = seed
        self.rng = np.random.default_rng(seed)

        # Build stable edge index mapping
        self.edge_list: List[Tuple[int, int]] = sorted(
            [(min(u, v), max(u, v)) for u, v in graph.edges()]
        )
        self.edge_to_idx: Dict[Tuple[int, int], int] = {
            e: idx for idx, e in enumerate(self.edge_list)
        }

        # Node adjacency structure for fast per-node local observation
        self.node_neighbors: Dict[int, List[int]] = {
            u: sorted(list(graph.neighbors(u))) for u in graph.nodes()
        }

        # Initialize link states
        self.current_step = 0
        self.history: List[Dict[str, Any]] = []
        self._sync_graph_initial_state()

    def _sync_graph_initial_state(self) -> None:
        """Initializes all edge state attributes on the graph at zero-load."""
        for u, v in self.edge_list:
            data = self.graph[u][v]
            base_prop = data.get("base_latency_ms", 10.0)
            cap = data.get("capacity_gbps", 10.0)

            data["traffic_load_gbps"] = 0.0
            data["congestion"] = 0.0
            data["available_bandwidth"] = 1.0
            data["available_bandwidth_gbps"] = cap
            data["latency"] = base_prop
            data["packet_loss_rate"] = 0.0

    def compute_link_metrics(
        self,
        base_prop_ms: float,
        capacity_gbps: float,
        traffic_load_gbps: float,
    ) -> Tuple[float, float, float, float, float]:
        """Calculates congestion, available bandwidth, queuing delay, latency, and packet loss rate.

        Guarantees:
          - No division by zero.
          - Strictly continuous and bounded metrics.
          - Exact unit consistency (Gbps and ms).

        Returns:
            Tuple of (congestion, available_bw_norm, available_bw_gbps, total_latency_ms, packet_loss_rate)
        """
        # 1. Congestion ratio in [0.0, 1.0]
        raw_c = traffic_load_gbps / capacity_gbps if capacity_gbps > 0 else 1.0
        congestion = float(np.clip(raw_c, 0.0, 1.0))

        # 2. Available bandwidth (normalized and absolute Gbps)
        available_bw_norm = float(np.clip(1.0 - congestion, 0.0, 1.0))
        available_bw_gbps = float(np.clip(capacity_gbps - traffic_load_gbps, 0.0, capacity_gbps))

        # 3. Safe asymptotic queuing delay formula (prevents division-by-zero)
        c_eff = min(congestion, self.queue_c_max)
        # Asymptotic M/M/1 queuing term: D_max * (c / (1 - c))
        queue_delay_ms = self.queue_d_max * (c_eff / (1.0 - c_eff))
        # Linear continuation beyond c_max for saturated regimes
        if congestion > self.queue_c_max:
            queue_delay_ms += self.queue_d_overflow * (congestion - self.queue_c_max)

        total_latency_ms = float(base_prop_ms + queue_delay_ms)

        # 4. Packet loss rate (RED buffer threshold drop)
        if congestion < self.loss_threshold:
            packet_loss_rate = 0.0
        else:
            normalized_over = (congestion - self.loss_threshold) / (1.0 - self.loss_threshold)
            packet_loss_rate = float(np.clip(self.loss_max_rate * (normalized_over ** 2), 0.0, 1.0))

        return (
            congestion,
            available_bw_norm,
            available_bw_gbps,
            total_latency_ms,
            packet_loss_rate,
        )

    def update_states(self, traffic_loads_gbps: Dict[Tuple[int, int], float]) -> None:
        """Updates link states for all edges based on current traffic loads in Gbps.

        Args:
            traffic_loads_gbps: Dict mapping edge (u, v) -> traffic load in Gbps.
        """
        step_snapshot = {"step": self.current_step, "edges": {}}

        for u, v in self.edge_list:
            edge_key = (min(u, v), max(u, v))
            load = traffic_loads_gbps.get(edge_key, 0.0)

            edge_data = self.graph[u][v]
            base_prop = edge_data["base_latency_ms"]
            cap = edge_data["capacity_gbps"]

            cong, avail_norm, avail_gbps, lat, loss = self.compute_link_metrics(
                base_prop_ms=base_prop,
                capacity_gbps=cap,
                traffic_load_gbps=load,
            )

            # Update NetworkX edge data in-place
            edge_data["traffic_load_gbps"] = load
            edge_data["congestion"] = cong
            edge_data["available_bandwidth"] = avail_norm
            edge_data["available_bandwidth_gbps"] = avail_gbps
            edge_data["latency"] = lat
            edge_data["packet_loss_rate"] = loss

            step_snapshot["edges"][edge_key] = {
                "load_gbps": load,
                "congestion": cong,
                "available_bandwidth": avail_norm,
                "available_bandwidth_gbps": avail_gbps,
                "latency": lat,
                "packet_loss_rate": loss,
            }

        self.history.append(step_snapshot)
        self.current_step += 1

    def get_global_state_tensor(self) -> np.ndarray:
        """Constructs the canonical Global State Tensor S_global in R^(14 x 14 x 4).

        Features for edge (u, v):
          [0] latency_norm: clip(latency / 100.0 ms, 0, 1)
          [1] congestion: [0, 1]
          [2] available_bandwidth: [0, 1]
          [3] packet_loss_rate: [0, 1]
        Non-existent links are zeros.
        """
        s_global = np.zeros((self.num_nodes, self.num_nodes, self.NUM_FEATURES), dtype=np.float32)

        for u, v in self.edge_list:
            data = self.graph[u][v]
            feat = np.array([
                np.clip(data["latency"] / self.LATENCY_NORM_SCALE_MS, 0.0, 1.0),
                data["congestion"],
                data["available_bandwidth"],
                data["packet_loss_rate"],
            ], dtype=np.float32)

            # Symmetric for bidirectional links
            s_global[u, v, :] = feat
            s_global[v, u, :] = feat

        return s_global

    def get_edge_feature_matrix(self) -> np.ndarray:
        """Constructs canonical Edge Matrix S_edges in R^(21 x 4) indexed by edge_to_idx."""
        s_edge = np.zeros((self.num_edges, self.NUM_FEATURES), dtype=np.float32)

        for e, idx in self.edge_to_idx.items():
            u, v = e
            data = self.graph[u][v]
            s_edge[idx, :] = [
                np.clip(data["latency"] / self.LATENCY_NORM_SCALE_MS, 0.0, 1.0),
                data["congestion"],
                data["available_bandwidth"],
                data["packet_loss_rate"],
            ]

        return s_edge

    def get_node_observation(self, node_id: int) -> Dict[str, np.ndarray]:
        """Derives the per-node local observation vector for node router agents in Gym.

        Pads to fixed degree d_max (4 for NSFNET) so observation shapes are static.

        Returns:
            Dict with:
              - 'neighbor_features': np.ndarray of shape (d_max, 4) with link features.
              - 'neighbor_ids': np.ndarray of shape (d_max,) with neighbor node IDs (-1 for padded).
              - 'port_mask': np.ndarray of shape (d_max,) with 1.0 for valid link, 0.0 for padded.
              - 'adjacency_slice': np.ndarray of shape (14, 4) full row from global state tensor.
        """
        if node_id not in self.node_neighbors:
            raise ValueError(f"Invalid node_id: {node_id}")

        neighbors = self.node_neighbors[node_id]
        deg = len(neighbors)

        neighbor_features = np.zeros((self.d_max, self.NUM_FEATURES), dtype=np.float32)
        neighbor_ids = np.full((self.d_max,), -1, dtype=np.int32)
        port_mask = np.zeros((self.d_max,), dtype=np.float32)

        for port_idx, neighbor in enumerate(neighbors[: self.d_max]):
            edge_data = self.graph[node_id][neighbor]
            neighbor_features[port_idx, :] = [
                np.clip(edge_data["latency"] / self.LATENCY_NORM_SCALE_MS, 0.0, 1.0),
                edge_data["congestion"],
                edge_data["available_bandwidth"],
                edge_data["packet_loss_rate"],
            ]
            neighbor_ids[port_idx] = neighbor
            port_mask[port_idx] = 1.0

        # Also provide direct row slice from the global tensor
        global_tensor = self.get_global_state_tensor()
        adjacency_slice = global_tensor[node_id, :, :]  # Shape (14, 4)

        return {
            "neighbor_features": neighbor_features,
            "neighbor_ids": neighbor_ids,
            "port_mask": port_mask,
            "adjacency_slice": adjacency_slice,
        }

    def reset(self) -> None:
        """Clears history and resets graph states to zero-load baseline."""
        self.current_step = 0
        self.history = []
        self._sync_graph_initial_state()
