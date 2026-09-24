"""Dynamic Dijkstra Routing Engine for NSFNET and Arbitrary Graph Topologies.

================================================================================
ARCHITECTURAL ASYMMETRY NOTE (Dijkstra vs. Future RL Agent):
--------------------------------------------------------------------------------
1. Dynamic Dijkstra (Centralized Benchmark):
   - Employs FULL-PATH SNAPSHOT COMPUTATION.
   - At the routing decision instant, an ingress router (or centralized SDN controller)
     inspects the global network state snapshot across all links.
     It computes the complete end-to-end path P = [s, v_1, v_2, ..., d] that minimizes
     cumulative dynamic latency sum_{(u,v) in P} latency(u, v), and fixes the entire
     trajectory for that packet.

2. Future RL Agent (Decentralized Multi-Agent / Q-Routing):
   - Employs PER-HOP SEQUENTIAL DECISION MAKING.
   - In later project phases, each router node inspects only its local state (or immediate
     neighbor link features) and independently selects the next-hop neighbor v_next.
     No centralized controller enforces a pre-planned global path; the end-to-end route
     emerges dynamically as the packet is forwarded hop-by-hop across the topology.

This architectural asymmetry is intentional:
- Dynamic Dijkstra provides the theoretical minimum-latency full-knowledge benchmark.
- The RL agent tackles the harder, realistic distributed routing problem under partial
  observability, localized information, and dynamically shifting traffic.
================================================================================
"""

from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import heapq
import itertools
import networkx as nx
import numpy as np


class DijkstraRouter:
    """Computes shortest paths on dynamic network graphs using fresh live link weights.

    Attributes:
        default_weight_attr: Edge attribute name representing live edge weight ('latency').
    """

    def __init__(self, default_weight_attr: str = "latency"):
        self.default_weight_attr = default_weight_attr

    def find_shortest_path(
        self,
        graph: nx.Graph,
        src: int,
        dst: int,
        weight_attr: Optional[str] = None,
        weight_fn: Optional[Callable[[nx.Graph, int, int], float]] = None,
    ) -> Tuple[List[int], float]:
        """Calculates minimum-cost path from src to dst using Dijkstra's algorithm.

        Edge weights are recomputed fresh from the live link state at the moment
        of decision, ensuring the path adapts dynamically to current network congestion.

        Args:
            graph: NetworkX graph with dynamic edge attributes.
            src: Source node identifier.
            dst: Destination node identifier.
            weight_attr: Edge attribute name to use as cost (defaults to self.default_weight_attr).
            weight_fn: Optional custom weight callable f(graph, u, v) -> float.

        Returns:
            Tuple of (path, total_cost):
              - path: Ordered list of nodes [src, ..., dst], or empty list [] if unreachable.
              - total_cost: Cumulative path cost (e.g. latency in ms), or inf if unreachable.

        Raises:
            KeyError: If src or dst is not in the graph.
        """
        if src not in graph:
            raise KeyError(f"Source node {src} not found in graph.")
        if dst not in graph:
            raise KeyError(f"Destination node {dst} not found in graph.")

        # Trivial path
        if src == dst:
            return ([src], 0.0)

        attr_name = weight_attr if weight_attr is not None else self.default_weight_attr

        # Priority queue entry format: (cumulative_cost, tie_breaker, current_node, path_so_far)
        counter = itertools.count()
        dist: Dict[int, float] = {src: 0.0}
        pq = [(0.0, next(counter), src, [src])]
        visited = set()

        while pq:
            current_cost, _, u, path = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            # Destination reached - because edge weights >= 0, this is guaranteed optimal
            if u == dst:
                return (path, current_cost)

            for neighbor in graph.neighbors(u):
                if neighbor in visited:
                    continue

                # Live weight recomputed directly from current graph link state
                if weight_fn is not None:
                    edge_weight = float(weight_fn(graph, u, neighbor))
                else:
                    edge_data = graph[u][neighbor]
                    if attr_name not in edge_data:
                        raise KeyError(f"Attribute '{attr_name}' not found on edge ({u}, {neighbor}).")
                    edge_weight = float(edge_data[attr_name])

                if edge_weight < 0:
                    raise ValueError(f"Negative edge weight encountered on ({u}, {neighbor}): {edge_weight}")

                new_cost = current_cost + edge_weight

                if neighbor not in dist or new_cost < dist[neighbor]:
                    dist[neighbor] = new_cost
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (new_cost, next(counter), neighbor, new_path))

        # Target is in a disconnected component
        return ([], float("inf"))

    def find_static_shortest_path(
        self,
        graph: nx.Graph,
        src: int,
        dst: int,
        static_attr: str = "distance_km",
    ) -> Tuple[List[int], float, float]:
        """Calculates static Shortest Path First (SPF) using static distance or base latency.

        Returns both the static cost (e.g. physical distance in km) and the ACTUAL
        live dynamic latency that a packet routed along this static path experiences.

        Args:
            graph: NetworkX graph.
            src: Source node identifier.
            dst: Destination node identifier.
            static_attr: Edge attribute for static metric (default 'distance_km').

        Returns:
            Tuple of (path, static_cost, actual_live_latency_ms)
        """
        path, static_cost = self.find_shortest_path(
            graph=graph,
            src=src,
            dst=dst,
            weight_attr=static_attr,
        )

        if not path:
            return ([], float("inf"), float("inf"))

        # Compute actual dynamic latency experienced on this static path
        metrics = self.compute_path_metrics(graph, path)
        return (path, static_cost, metrics["total_latency_ms"])

    def find_stale_shortest_path(
        self,
        live_graph: nx.Graph,
        stale_graph: nx.Graph,
        src: int,
        dst: int,
        weight_attr: Optional[str] = None,
    ) -> Tuple[List[int], float, Dict[str, float]]:
        """Computes shortest path based on a stale/periodic link-state graph snapshot,

        and evaluates the resulting trajectory against the true live network graph.

        Models realistic Link-State Advertisement (LSA) flooding intervals in OSPF/IS-IS
        where routers make decisions on telemetry that is periodic rather than an
        instantaneous zero-delay oracle.

        Args:
            live_graph: Current live network state graph experiencing real traffic.
            stale_graph: Snapshot graph representing telemetry at prior update epoch.
            src: Source node identifier.
            dst: Destination node identifier.
            weight_attr: Metric attribute to optimize (defaults to self.default_weight_attr).

        Returns:
            Tuple of (path, computed_stale_cost, live_path_metrics)
        """
        path, stale_cost = self.find_shortest_path(
            graph=stale_graph,
            src=src,
            dst=dst,
            weight_attr=weight_attr,
        )
        if not path:
            return ([], float("inf"), self.compute_path_metrics(live_graph, []))

        live_metrics = self.compute_path_metrics(live_graph, path)
        return (path, stale_cost, live_metrics)


    @staticmethod
    def compute_path_metrics(graph: nx.Graph, path: List[int]) -> Dict[str, float]:
        """Computes comprehensive telemetry along a specified path from live link states.

        Args:
            graph: NetworkX graph with updated link states.
            path: List of nodes [v0, v1, ..., vk].

        Returns:
            Dict containing:
              - 'total_latency_ms': Cumulative latency (propagation + queuing) along path.
              - 'base_latency_ms': Physical propagation delay without queuing.
              - 'queuing_delay_ms': Total queuing delay accumulated across all hops.
              - 'hop_count': Number of edges traversed (len(path) - 1).
              - 'path_loss_prob': End-to-end drop probability: 1 - prod(1 - loss_e).
              - 'bottleneck_avail_bw_gbps': Minimum available bandwidth along path.
              - 'max_congestion': Maximum link congestion along path.
        """
        if len(path) <= 1:
            return {
                "total_latency_ms": 0.0,
                "base_latency_ms": 0.0,
                "queuing_delay_ms": 0.0,
                "hop_count": 0,
                "path_loss_prob": 0.0,
                "bottleneck_avail_bw_gbps": float("inf"),
                "max_congestion": 0.0,
            }

        total_lat = 0.0
        base_lat = 0.0
        success_prob = 1.0
        bottleneck_bw = float("inf")
        max_cong = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = graph[u][v]

            lat = float(data.get("latency", 0.0))
            b_lat = float(data.get("base_latency_ms", lat))
            loss = float(data.get("packet_loss_rate", 0.0))
            avail_bw = float(data.get("available_bandwidth_gbps", 0.0))
            cong = float(data.get("congestion", 0.0))

            total_lat += lat
            base_lat += b_lat
            success_prob *= (1.0 - np.clip(loss, 0.0, 1.0))
            bottleneck_bw = min(bottleneck_bw, avail_bw)
            max_cong = max(max_cong, cong)

        path_loss_prob = float(np.clip(1.0 - success_prob, 0.0, 1.0))
        queuing_delay = max(0.0, total_lat - base_lat)

        return {
            "total_latency_ms": total_lat,
            "base_latency_ms": base_lat,
            "queuing_delay_ms": queuing_delay,
            "hop_count": len(path) - 1,
            "path_loss_prob": path_loss_prob,
            "bottleneck_avail_bw_gbps": bottleneck_bw,
            "max_congestion": max_cong,
        }
