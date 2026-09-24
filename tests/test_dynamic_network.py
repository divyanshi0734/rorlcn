"""Unit and property tests for NSFNET topology and dynamic link-state system."""

import pytest
import numpy as np
import networkx as nx

from src.topology import build_nsfnet_graph, NSFNET_NODES, NSFNET_EDGES, FIBER_PROPAGATION_SPEED_KM_PER_MS
from src.dynamic_link_state import DynamicLinkStateEngine
from src.traffic_generator import TrafficGenerator
from src.network_sim import NSFNETSimEnv


def test_nsfnet_topology_structure():
    """Verify graph has exactly 14 nodes, 21 edges, is connected, and valid degree distribution."""
    g = build_nsfnet_graph()

    assert g.number_of_nodes() == 14, "Graph must have 14 nodes"
    assert g.number_of_edges() == 21, "Graph must have 21 edges"
    assert nx.is_connected(g), "Graph must be fully connected"

    degrees = [d for _, d in g.degree()]
    assert min(degrees) >= 2, "Minimum degree must be at least 2"
    assert max(degrees) <= 4, "Maximum degree must be at most 4 (d_max = 4)"

    for u, v in g.edges():
        data = g[u][v]
        expected_base_lat = data["distance_km"] / FIBER_PROPAGATION_SPEED_KM_PER_MS
        assert np.isclose(data["base_latency_ms"], expected_base_lat), "Base latency must match distance / 200"
        assert data["capacity_gbps"] == 10.0, "Default capacity must be 10.0 Gbps"


def test_queuing_delay_safe_division_by_zero():
    """Verify safe asymptotic queuing formula never causes NaN/Inf as c approaches or equals 1.0."""
    g = build_nsfnet_graph()
    engine = DynamicLinkStateEngine(graph=g)

    # Test range from 0 to severe saturation (1.5)
    test_loads = [0.0, 3.0, 7.0, 9.8, 9.99, 10.0, 15.0]
    cap = 10.0
    base_prop = 5.0

    prev_lat = -1.0
    for load in test_loads:
        cong, avail_norm, avail_gbps, lat, loss = engine.compute_link_metrics(
            base_prop_ms=base_prop,
            capacity_gbps=cap,
            traffic_load_gbps=load,
        )

        assert not np.isnan(lat), f"Latency was NaN for load={load}"
        assert not np.isinf(lat), f"Latency was Inf for load={load}"
        assert 0.0 <= cong <= 1.0, f"Congestion {cong} out of bounds"
        assert 0.0 <= avail_norm <= 1.0, f"Avail BW {avail_norm} out of bounds"
        assert 0.0 <= loss <= 1.0, f"Loss {loss} out of bounds"
        assert lat >= prev_lat, "Latency must be monotonically non-decreasing with load"
        prev_lat = lat


def test_dynamic_link_stats_change_over_time():
    """Verify link statistics actually change across discrete timesteps (variance > 0)."""
    env = NSFNETSimEnv(seed=123)
    env.reset()

    # Step for 30 discrete timesteps
    for _ in range(30):
        env.step()

    df = env.get_telemetry_dataframe()
    assert len(df) == 31 * 21, "Must contain all timesteps for all 21 edges"

    # Group by edge and test variance
    grouped = df.groupby("edge")
    for edge_name, group in grouped:
        load_var = group["load_gbps"].var()
        cong_var = group["congestion"].var()
        lat_var = group["latency_ms"].var()
        avail_var = group["available_bandwidth"].var()

        assert load_var > 0.01, f"Edge {edge_name} load did not change over time (var={load_var})"
        assert cong_var > 0.0001, f"Edge {edge_name} congestion did not change over time (var={cong_var})"
        assert lat_var > 0.0001, f"Edge {edge_name} latency did not change over time (var={lat_var})"
        assert avail_var > 0.0001, f"Edge {edge_name} available bandwidth did not change over time (var={avail_var})"


def test_metric_relationships_and_bounds():
    """Verify strict relationships between congestion, bandwidth, loss, and latency."""
    env = NSFNETSimEnv(seed=456)
    env.reset()

    for _ in range(20):
        _, edge_matrix, info = env.step()

        lat_norm = edge_matrix[:, 0]
        cong = edge_matrix[:, 1]
        avail = edge_matrix[:, 2]
        loss = edge_matrix[:, 3]

        # Bounds checks
        assert np.all(cong >= 0.0) and np.all(cong <= 1.0)
        assert np.all(avail >= 0.0) and np.all(avail <= 1.0)
        assert np.all(loss >= 0.0) and np.all(loss <= 1.0)
        assert np.all(lat_norm >= 0.0) and np.all(lat_norm <= 1.0)

        # Complementarity: c + avail = 1.0
        assert np.allclose(cong + avail, 1.0, atol=1e-5), "Congestion and available bandwidth must sum to 1.0"

        # Packet loss threshold: loss > 0 only when congestion >= 0.70
        for i in range(len(cong)):
            if cong[i] < 0.70:
                assert loss[i] == 0.0, f"Expected 0 loss below threshold 0.70, got {loss[i]}"


def test_seed_reproducibility():
    """Verify that explicit seed parameter produces identical, deterministic trajectories."""
    env1 = NSFNETSimEnv(seed=777)
    env2 = NSFNETSimEnv(seed=777)
    env_diff = NSFNETSimEnv(seed=888)

    g1, e1 = env1.reset()
    g2, e2 = env2.reset()
    g_d, e_d = env_diff.reset()

    assert np.allclose(g1, g2), "Initial global states with same seed must be equal"
    assert np.allclose(e1, e2), "Initial edge matrices with same seed must be equal"

    for _ in range(10):
        g1, e1, _ = env1.step()
        g2, e2, _ = env2.step()
        g_d, e_d, _ = env_diff.step()

        assert np.allclose(g1, g2), "Stepped global state tensors must be bit-exact equal"
        assert np.allclose(e1, e2), "Stepped edge matrices must be bit-exact equal"

    # Different seed must yield different states
    assert not np.allclose(e1, e_d), "Different seeds must produce different trajectories"


def test_state_tensor_shapes_and_gym_local_view():
    """Verify global state tensor, edge matrix, and per-node local observation shapes."""
    env = NSFNETSimEnv(seed=999)
    env.reset()
    global_tensor, edge_matrix, _ = env.step()

    # Global state tensor: (14, 14, 4)
    assert global_tensor.shape == (14, 14, 4), f"Expected (14, 14, 4), got {global_tensor.shape}"
    # Canonical edge matrix: (21, 4)
    assert edge_matrix.shape == (21, 4), f"Expected (21, 4), got {edge_matrix.shape}"

    # Verify per-node observation for each router
    for node_id in range(14):
        obs = env.get_node_observation(node_id)
        assert "neighbor_features" in obs
        assert "neighbor_ids" in obs
        assert "port_mask" in obs
        assert "adjacency_slice" in obs

        # Padded degree shape (d_max=4, features=4)
        assert obs["neighbor_features"].shape == (4, 4)
        assert obs["neighbor_ids"].shape == (4,)
        assert obs["port_mask"].shape == (4,)
        assert obs["adjacency_slice"].shape == (14, 4)

        deg = env.graph.degree(node_id)
        assert int(np.sum(obs["port_mask"])) == deg, f"Active ports must equal degree {deg}"
        assert obs["adjacency_slice"].shape[0] == 14
