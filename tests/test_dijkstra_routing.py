"""Unit and property tests for Dynamic Dijkstra routing, packet traffic, and telemetry.

Includes:
  1. Hand-checkable 4-node diamond subgraph test.
  2. Closed-loop packet load feedback test.
  3. Edge cases: source == destination, disconnected components, missing nodes.
  4. NSFNET integration & optimality verification against all alternative paths.
  5. Telemetry & aggregate metric calculation integrity.
  6. Seed reproducibility across simulation runs.
"""

import pytest
import numpy as np
import networkx as nx

from src.topology import build_nsfnet_graph, FIBER_PROPAGATION_SPEED_KM_PER_MS
from src.dynamic_link_state import DynamicLinkStateEngine
from src.network_sim import NSFNETSimEnv
from src.dijkstra import DijkstraRouter
from src.packet_traffic import Packet, PacketTrafficGenerator, RoutingSimulationRunner
from src.metrics import compute_aggregate_metrics, compare_routing_policies


# ==============================================================================
# 1. HAND-CHECKABLE SUBGRAPH TESTS
# ==============================================================================

def build_diamond_subgraph():
    r"""Builds a small, hand-checkable 4-node diamond graph:
    
            (1: B)
           /      \
       (0: A)    (3: D)
           \      /
            (2: C)

    Static properties:
      - Path 1 (0 -> 1 -> 3): 400 km + 400 km = 800 km (base latency 4.0 ms)
      - Path 2 (0 -> 2 -> 3): 700 km + 700 km = 1400 km (base latency 7.0 ms)
    """
    g = nx.Graph()
    for n in range(4):
        g.add_node(n)

    edges = [
        (0, 1, 400.0),  # A - B
        (1, 3, 400.0),  # B - D
        (0, 2, 700.0),  # A - C
        (2, 3, 700.0),  # C - D
    ]
    for u, v, dist in edges:
        base_lat = dist / FIBER_PROPAGATION_SPEED_KM_PER_MS
        g.add_edge(
            u, v,
            distance_km=dist,
            base_latency_ms=base_lat,
            capacity_gbps=10.0,
            traffic_load_gbps=0.0,
            congestion=0.0,
            available_bandwidth=1.0,
            available_bandwidth_gbps=10.0,
            latency=base_lat,
            packet_loss_rate=0.0,
        )
    return g


def test_dijkstra_hand_checkable_subgraph():
    """Verify Dijkstra correctly finds minimum-latency path on a small hand-checkable subgraph.

    Hand Calculation Walkthrough:
      Case 1 (Zero Congestion):
        - Path A-B-D latency: 2.0 ms + 2.0 ms = 4.0 ms
        - Path A-C-D latency: 3.5 ms + 3.5 ms = 7.0 ms
        -> Dijkstra must pick [0, 1, 3] with cost 4.0 ms.

      Case 2 (Queuing Delay on B-D):
        - Congestion adds 8.0 ms queuing delay to link (1, 3), making latency(1, 3) = 10.0 ms.
        - Path A-B-D dynamic latency: 2.0 ms + 10.0 ms = 12.0 ms.
        - Path A-C-D dynamic latency: 3.5 ms + 3.5 ms = 7.0 ms.
        -> Hand-computed minimum: 7.0 ms < 12.0 ms.
        -> Static SPF still picks [0, 1, 3] (distance 800 km < 1400 km), suffering 12.0 ms latency.
        -> Dynamic Dijkstra MUST select [0, 2, 3] with exact cost 7.0 ms!

      Case 3 (Dynamic Reversal when Congestion Shifts):
        - Congestion on (1, 3) clears back to 2.0 ms.
        - Link (0, 2) is congested with 10.0 ms queuing delay (latency = 13.5 ms).
        - Path A-B-D dynamic latency: 4.0 ms.
        - Path A-C-D dynamic latency: 13.5 ms + 3.5 ms = 17.0 ms.
        -> Dynamic Dijkstra MUST dynamically switch back to [0, 1, 3] with cost 4.0 ms!
    """
    g = build_diamond_subgraph()
    router = DijkstraRouter()

    # --- Case 1: Uncongested State ---
    path, cost = router.find_shortest_path(g, src=0, dst=3, weight_attr="latency")
    assert path == [0, 1, 3], f"Expected [0, 1, 3], got {path}"
    assert np.isclose(cost, 4.0), f"Expected cost 4.0 ms, got {cost}"

    # --- Case 2: Congestion on Link (1, 3) ---
    # Add queuing delay: latency becomes 10.0 ms
    g[1][3]["latency"] = 10.0
    g[1][3]["congestion"] = 0.90
    g[1][3]["packet_loss_rate"] = 0.08

    # Static SPF uses distance_km
    static_path, static_dist, actual_lat = router.find_static_shortest_path(g, src=0, dst=3, static_attr="distance_km")
    assert static_path == [0, 1, 3], "Static SPF must still pick shortest distance path [0, 1, 3]"
    assert np.isclose(static_dist, 800.0)
    assert np.isclose(actual_lat, 12.0), f"Static SPF actual latency must be 12.0 ms, got {actual_lat}"

    # Dynamic Dijkstra recomputed fresh from live link states
    dyn_path, dyn_cost = router.find_shortest_path(g, src=0, dst=3, weight_attr="latency")
    assert dyn_path == [0, 2, 3], f"Dynamic Dijkstra must detour around congested link, got {dyn_path}"
    assert np.isclose(dyn_cost, 7.0), f"Hand calculation: 3.5 + 3.5 = 7.0 ms, got {dyn_cost}"
    assert dyn_cost < actual_lat, "Dynamic Dijkstra must achieve strictly lower latency than static SPF"

    # --- Case 3: Reversal when Congestion Shifts to (0, 2) ---
    g[1][3]["latency"] = 2.0  # Cleared
    g[1][3]["congestion"] = 0.0
    g[0][2]["latency"] = 13.5  # Heavy congestion on alternate path
    g[0][2]["congestion"] = 0.95

    rev_path, rev_cost = router.find_shortest_path(g, src=0, dst=3, weight_attr="latency")
    assert rev_path == [0, 1, 3], f"Dynamic Dijkstra must switch back to [0, 1, 3], got {rev_path}"
    assert np.isclose(rev_cost, 4.0), f"Hand calculation: 2.0 + 2.0 = 4.0 ms, got {rev_cost}"


# ==============================================================================
# 2. CLOSED-LOOP PACKET LOAD FEEDBACK TESTS
# ==============================================================================

def test_packet_routing_feedback_loop():
    """Verify that routing packets contributes to link load (load_e(t)), updating congestion and latency."""
    env = NSFNETSimEnv(seed=42)
    env.reset()

    # Select an arbitrary link (0, 1) Seattle - Palo Alto
    u, v = 0, 1
    initial_load = env.graph[u][v]["traffic_load_gbps"]
    initial_lat = env.graph[u][v]["latency"]

    # Apply packet loads: 20 packets each with 0.1 Gbps demand = 2.0 Gbps extra load
    packet_demand = 0.10
    num_packets = 20
    env.apply_packet_load({(u, v): packet_demand * num_packets})

    updated_load = env.graph[u][v]["traffic_load_gbps"]
    updated_lat = env.graph[u][v]["latency"]

    assert np.isclose(updated_load, initial_load + 2.0), f"Load must increase by 2.0 Gbps, got {updated_load}"
    assert updated_lat > initial_lat, f"Latency must increase with higher load: {updated_lat} > {initial_lat}"
    assert env.routed_loads[(min(u, v), max(u, v))] == 2.0, "Routed loads tracker must record 2.0 Gbps"


def test_packet_feedback_prompts_dynamic_detour():
    """Verify that accumulating packet load across successive decisions causes later packets to detour."""
    g = build_diamond_subgraph()
    router = DijkstraRouter()

    # Route 50 packets on diamond graph from 0 to 3 with feedback
    # Each packet traversing a link adds 0.25 Gbps
    paths_chosen = []
    for _ in range(30):
        path, cost = router.find_shortest_path(g, src=0, dst=3, weight_attr="latency")
        paths_chosen.append(path)

        # Accumulate load on chosen path
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            g[n1][n2]["traffic_load_gbps"] += 0.30
            # Recompute dynamic latency with simple load penalty
            g[n1][n2]["latency"] = g[n1][n2]["base_latency_ms"] + 0.8 * g[n1][n2]["traffic_load_gbps"]

    # Initial packets should choose [0, 1, 3] (cost 4.0 ms)
    assert paths_chosen[0] == [0, 1, 3]
    # As [0, 1, 3] saturates, later packets must dynamically detour to [0, 2, 3]
    assert [0, 2, 3] in paths_chosen, "Feedback loop must cause later packets to detour around congested link"


# ==============================================================================
# 3. EDGE CASES & ERROR HANDLING
# ==============================================================================

def test_dijkstra_edge_cases():
    """Test identical source/destination, disconnected components, and invalid nodes."""
    g = build_diamond_subgraph()
    router = DijkstraRouter()

    # Source == Destination
    path, cost = router.find_shortest_path(g, src=1, dst=1)
    assert path == [1]
    assert cost == 0.0

    # Disconnected component
    g.add_node(99)  # Isolated node
    path, cost = router.find_shortest_path(g, src=0, dst=99)
    assert path == []
    assert np.isinf(cost)

    # Missing node raises KeyError
    with pytest.raises(KeyError):
        router.find_shortest_path(g, src=0, dst=100)
    with pytest.raises(KeyError):
        router.find_shortest_path(g, src=-1, dst=3)


# ==============================================================================
# 4. NSFNET INTEGRATION & GLOBAL OPTIMALITY
# ==============================================================================

def test_dijkstra_on_live_nsfnet_optimality():
    """Verify Dijkstra on full live NSFNET environment finds globally minimal latency path."""
    env = NSFNETSimEnv(seed=100)
    env.reset()
    router = DijkstraRouter()

    # Step to generate realistic dynamic loads
    for _ in range(5):
        env.step()

    # Test all pairs shortest paths vs all simple paths in NetworkX
    test_pairs = [(0, 13), (1, 12), (2, 10), (6, 8)]
    for src, dst in test_pairs:
        dijk_path, dijk_cost = router.find_shortest_path(env.graph, src, dst, weight_attr="latency")
        assert len(dijk_path) >= 2
        assert dijk_path[0] == src and dijk_path[-1] == dst

        # Calculate exact manual sum along chosen path
        manual_sum = sum(env.graph[u][v]["latency"] for u, v in zip(dijk_path[:-1], dijk_path[1:]))
        assert np.isclose(dijk_cost, manual_sum), f"Dijkstra cost {dijk_cost} must match manual edge sum {manual_sum}"

        # Verify no alternate simple path has lower latency
        for alt_path in nx.all_simple_paths(env.graph, src, dst, cutoff=5):
            alt_cost = sum(env.graph[u][v]["latency"] for u, v in zip(alt_path[:-1], alt_path[1:]))
            assert dijk_cost <= alt_cost + 1e-6, f"Dijkstra path {dijk_path} ({dijk_cost}) was not optimal compared to {alt_path} ({alt_cost})"


# ==============================================================================
# 5. PACKET GENERATOR & SIMULATION RUNNER INTEGRITY
# ==============================================================================

def test_packet_generator_sampling_and_seed():
    """Verify PacketTrafficGenerator samples valid (s, d) pairs and respects seed reproducibility."""
    gen1 = PacketTrafficGenerator(seed=12345, packets_per_step=20)
    gen2 = PacketTrafficGenerator(seed=12345, packets_per_step=20)
    gen_diff = PacketTrafficGenerator(seed=99999, packets_per_step=20)

    pkts1 = gen1.generate_packets(timestep=1)
    pkts2 = gen2.generate_packets(timestep=1)
    pkts_d = gen_diff.generate_packets(timestep=1)

    assert len(pkts1) == 20
    for p in pkts1:
        assert p.src != p.dst, f"Source and destination must be distinct, got ({p.src}, {p.dst})"
        assert 0 <= p.src < 14 and 0 <= p.dst < 14

    # Identical seeds must match
    pairs1 = [(p.src, p.dst) for p in pkts1]
    pairs2 = [(p.src, p.dst) for p in pkts2]
    pairs_d = [(p.src, p.dst) for p in pkts_d]

    assert pairs1 == pairs2, "Matching seeds must generate identical packet streams"
    assert pairs1 != pairs_d, "Distinct seeds must generate distinct packet streams"


def test_simulation_runner_and_metrics_calculation():
    """Verify RoutingSimulationRunner executes full episode and calculates consistent metrics."""
    env = NSFNETSimEnv(seed=888)
    runner = RoutingSimulationRunner(env=env, seed=888)

    pkt_df, step_df = runner.run_simulation(num_timesteps=10, routing_mode="dynamic_dijkstra")

    assert len(step_df) == 10, "Step df must have 10 rows"
    assert len(pkt_df) == 10 * 50, "Total packets must equal 10 steps * 50 packets/step"

    metrics = compute_aggregate_metrics(pkt_df, total_timesteps=10)

    assert metrics["total_packets_sent"] == len(pkt_df)
    assert metrics["total_packets_delivered"] + metrics["total_packets_dropped"] == metrics["total_packets_sent"]
    assert 0.0 <= metrics["packet_loss_rate"] <= 1.0
    assert metrics["effective_throughput_gbps"] >= 0.0
    assert metrics["mean_latency_ms"] > 0.0
    assert metrics["p95_latency_ms"] >= metrics["mean_latency_ms"] or np.isclose(metrics["p95_latency_ms"], metrics["mean_latency_ms"], atol=1e-3)
    assert metrics["min_hops"] >= 1


def test_routing_policy_comparison():
    """Verify Dynamic Dijkstra achieves lower or equal latency/loss compared to static SPF."""
    # Seed 42 with a traffic spike at t=3 on link (1, 7) Palo Alto - Chicago
    env_d = NSFNETSimEnv(seed=42)
    env_s = NSFNETSimEnv(seed=42)

    runner_d = RoutingSimulationRunner(env=env_d, seed=42)
    runner_s = RoutingSimulationRunner(env=env_s, seed=42)

    spikes = {3: (1, 7, 6.0)}  # Heavy 6.0 Gbps surge on Palo Alto - Chicago

    d_pkts, d_steps = runner_d.run_simulation(num_timesteps=8, routing_mode="dynamic_dijkstra", spike_schedule=spikes)
    s_pkts, s_steps = runner_s.run_simulation(num_timesteps=8, routing_mode="static_spf", spike_schedule=spikes)

    comparison_df = compare_routing_policies(d_pkts, s_pkts, total_timesteps=8)
    assert not comparison_df.empty

    m_d = compute_aggregate_metrics(d_pkts, total_timesteps=8)
    m_s = compute_aggregate_metrics(s_pkts, total_timesteps=8)

    # Dynamic Dijkstra should avoid the saturated link (1, 7) during the spike
    assert m_d["total_packets_delivered"] >= m_s["total_packets_delivered"], "Dynamic Dijkstra should deliver at least as many packets as Static SPF"
    assert m_d["packet_loss_rate"] <= m_s["packet_loss_rate"], "Dynamic Dijkstra should have lower or equal loss rate"


def test_routing_simulation_runner_split_horizon():
    """Verify that enable_split_horizon suppresses bouncing loops in RL agent routing."""
    import os
    from sb3_contrib import MaskablePPO

    model_path = "checkpoints/final_model_200k.zip"
    if not os.path.exists(model_path):
        pytest.skip("200k model not available for split-horizon test")

    model = MaskablePPO.load(model_path)

    # Simulate 10 steps on seed 100 with and without split horizon
    env_off = NSFNETSimEnv(seed=100)
    gen_off = PacketTrafficGenerator(seed=100, packets_per_step=50)
    runner_off = RoutingSimulationRunner(
        env=env_off,
        packet_gen=gen_off,
        rl_model=model,
        enable_split_horizon=False,
        seed=100,
    )
    pkts_off, _ = runner_off.run_simulation(num_timesteps=10, routing_mode="rl_agent")

    env_on = NSFNETSimEnv(seed=100)
    gen_on = PacketTrafficGenerator(seed=100, packets_per_step=50)
    runner_on = RoutingSimulationRunner(
        env=env_on,
        packet_gen=gen_on,
        rl_model=model,
        enable_split_horizon=True,
        seed=100,
    )
    pkts_on, _ = runner_on.run_simulation(num_timesteps=10, routing_mode="rl_agent")

    # Split-Horizon should deliver strictly greater or equal packets than without Split-Horizon
    delivered_off = pkts_off["delivered"].sum()
    delivered_on = pkts_on["delivered"].sum()
    assert delivered_on >= delivered_off, f"Split-Horizon delivered {delivered_on} vs {delivered_off} without"


def test_custom_network_generation():
    """Verify custom network generation produces valid, connected topologies with physical attributes."""
    from src.topology import build_custom_network

    for n_nodes in [4, 8, 12]:
        for n_edges in [n_nodes, n_nodes + 4]:
            g = build_custom_network(num_nodes=n_nodes, num_edges=n_edges, seed=123)
            assert g.number_of_nodes() == n_nodes
            expected_edges = min(n_edges, n_nodes * (n_nodes - 1) // 2)
            assert g.number_of_edges() == expected_edges
            assert nx.is_connected(g)

            for u, v, data in g.edges(data=True):
                assert "distance_km" in data
                assert "base_latency_ms" in data
                assert "capacity_gbps" in data
                assert data["distance_km"] >= 300.0
                assert data["capacity_gbps"] > 0


def test_periodic_stale_dijkstra_runner():
    """Verify Periodic/Stale Dijkstra runs and yields valid telemetry."""
    env = NSFNETSimEnv(seed=42)
    router = DijkstraRouter()
    gen = PacketTrafficGenerator(seed=42, packets_per_step=20)
    runner = RoutingSimulationRunner(env=env, packet_gen=gen, router=router, seed=42, periodic_interval=3)
    pkts, steps = runner.run_simulation(num_timesteps=6, routing_mode="periodic_dijkstra")
    assert len(pkts) == 120
    assert len(steps) == 6
    assert "delivered" in pkts.columns
    assert pkts["delivered"].mean() > 0.80


def test_path_vector_masking_mode():
    """Verify loop_avoidance_mode='path_vector' operates correctly without runtime exceptions."""
    import os
    from sb3_contrib import MaskablePPO

    model_path = "checkpoints/reward_tuning/candidate_a.zip"
    if not os.path.exists(model_path):
        model_path = "checkpoints/final_model_200k.zip"
    if not os.path.exists(model_path):
        pytest.skip("RL model checkpoint not found")

    model = MaskablePPO.load(model_path)
    env = NSFNETSimEnv(seed=42)
    gen = PacketTrafficGenerator(seed=42, packets_per_step=20)
    runner = RoutingSimulationRunner(
        env=env,
        packet_gen=gen,
        rl_model=model,
        loop_avoidance_mode="path_vector",
        seed=42,
    )
    pkts, steps = runner.run_simulation(num_timesteps=5, routing_mode="rl_agent")
    assert len(pkts) == 100
    assert pkts["delivered"].sum() > 80

