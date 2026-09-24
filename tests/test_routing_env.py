"""Unit and property tests for Gym-style NetworkRoutingEnv and per-hop reward function.

Verifies:
  1. Post-load proactive congestion computation (congestion_after > congestion_before).
  2. Exact hand-computed reward verification on the 4-node diamond subgraph.
  3. Loop detection with -8.0 penalty and episode termination.
  4. Max hops truncation at 10 hops (not treated as successful delivery).
  5. Action masking (action_masks) and invalid port fallback penalty (-2.0).
  6. Destination arrival (+10.0 reward) short-circuiting the packet drop check.
  7. Intermediate hop packet drop trial with calibrated penalty.
"""

import pytest
import numpy as np
import networkx as nx

from src.topology import FIBER_PROPAGATION_SPEED_KM_PER_MS
from src.network_sim import NSFNETSimEnv
from src.routing_env import NetworkRoutingEnv, RewardConfig


# ==============================================================================
# Helpers: Hand-Checkable Subgraph Setup
# ==============================================================================

def build_diamond_sim_env():
    r"""Builds a 4-node diamond simulation environment:
            (1: B)
           /      \
       (0: A)    (3: D)
           \      /
            (2: C)
    """
    g = nx.Graph()
    for n in range(4):
        g.add_node(n, name=f"Node_{n}", pos=(0.0, 0.0), state="XX")

    edges = [
        (0, 1, 400.0),  # A - B (base_lat = 2.0 ms)
        (1, 3, 400.0),  # B - D (base_lat = 2.0 ms)
        (0, 2, 700.0),  # A - C (base_lat = 3.5 ms)
        (2, 3, 700.0),  # C - D (base_lat = 3.5 ms)
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

    sim = NSFNETSimEnv(seed=42)
    # Substitute the 4-node graph
    sim.graph = g
    sim.link_state_engine.graph = g
    sim.link_state_engine.num_nodes = 4
    sim.link_state_engine.num_edges = 4
    sim.link_state_engine.edge_list = sorted([(min(u, v), max(u, v)) for u, v in g.edges()])
    sim.link_state_engine.edge_to_idx = {e: idx for idx, e in enumerate(sim.link_state_engine.edge_list)}
    sim.link_state_engine.node_neighbors = {u: sorted(list(g.neighbors(u))) for u in g.nodes()}
    sim.current_loads = {e: 0.0 for e in sim.link_state_engine.edge_list}
    sim.routed_loads = {e: 0.0 for e in sim.link_state_engine.edge_list}
    return sim


# ==============================================================================
# 1. Proactive Closed-Loop Congestion After Test
# ==============================================================================

def test_congestion_after_is_post_load():
    """Verify congestion_after is computed strictly post-load (not pre-load)."""
    sim = NSFNETSimEnv(seed=42)
    sim.reset()

    # Route a packet with demand 0.50 Gbps
    demand = 0.50
    env = NetworkRoutingEnv(env=sim, packet_demand_gbps=demand, seed=42)
    obs, info = env.reset(src=0, dst=13)

    # Node 0 has valid outgoing neighbors
    action = 0  # Port 0
    neighbor_target = int(obs["neighbor_ids"][action])

    congestion_before = float(sim.graph[0][neighbor_target]["congestion"])

    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert step_info["congestion_after"] > congestion_before, (
        f"congestion_after ({step_info['congestion_after']}) must exceed "
        f"congestion_before ({congestion_before}) after packet demand is added"
    )
    # Exact check: load should increase by demand (0.5 Gbps / 10 Gbps = +0.05 congestion)
    cap = sim.graph[0][neighbor_target]["capacity_gbps"]
    expected_delta = demand / cap
    actual_delta = step_info["congestion_after"] - congestion_before
    assert np.isclose(actual_delta, expected_delta, atol=1e-5), f"Expected congestion delta {expected_delta}, got {actual_delta}"


# ==============================================================================
# 2. Hand-Checkable Subgraph Reward Calculation Test (Parameterized)
# ==============================================================================

REWARD_CONFIG_CANDIDATES = [
    ("baseline", 3.0, 2.0, 5.0, 10),
    ("candidate_a", 2.0, 4.0, 5.0, 10),
    ("candidate_b", 3.0, 2.0, 8.0, 10),
    ("candidate_c", 2.0, 4.0, 8.0, 7),
]

@pytest.mark.parametrize("cfg_name, w1, w3, drop_base, max_hops", REWARD_CONFIG_CANDIDATES)
def test_reward_calculation_on_hand_checkable_subgraph(cfg_name, w1, w3, drop_base, max_hops):
    """Verify exact match of step reward on 4-node diamond subgraph across candidate reward configs."""
    sim = build_diamond_sim_env()
    cfg = RewardConfig(
        w1_congestion=w1,
        w2_latency=1.0,
        w3_loss=w3,
        w4_hop_penalty=0.05,
        terminal_drop_penalty_base=drop_base,
        max_hops=max_hops,
    )
    env = NetworkRoutingEnv(env=sim, reward_config=cfg, packet_demand_gbps=0.20, seed=42)

    obs, _ = env.reset(src=0, dst=3)
    # Node 0 neighbors are sorted: [1, 2] -> port 0 is Node 1
    assert obs["neighbor_ids"][0] == 1

    next_obs, reward, terminated, truncated, info = env.step(action=0)

    # Hand-computed components
    expected_c_after = 0.20 / 10.0  # 0.020
    q_delay = 0.5 * (expected_c_after / (1.0 - expected_c_after))
    expected_lat = 2.0 + q_delay
    expected_lat_norm = expected_lat / 100.0
    expected_loss = 0.0
    expected_r_hop = -(w1 * expected_c_after + 1.0 * expected_lat_norm + w3 * expected_loss + 0.05)

    assert np.isclose(info["congestion_after"], expected_c_after, atol=1e-5)
    assert np.isclose(info["latency_norm"], expected_lat_norm, atol=1e-5)
    assert np.isclose(info["packet_loss_prob"], expected_loss, atol=1e-5)
    assert np.isclose(reward, expected_r_hop, atol=1e-5), (
        f"[{cfg_name}] Reward {reward} does not match hand-calculated {expected_r_hop}"
    )
    assert not terminated
    assert not truncated


# ==============================================================================
# 3. Loop Detection and Penalty Test
# ==============================================================================

def test_loop_detection_and_penalty():
    """Verify that visiting a node already in path_so_far triggers -8.0 penalty and terminates."""
    sim = build_diamond_sim_env()
    env = NetworkRoutingEnv(env=sim, packet_demand_gbps=0.05, seed=42)

    obs, _ = env.reset(src=0, dst=3)

    # Hop 1: 0 -> 1
    next_obs, r1, term1, _, info1 = env.step(action=0)
    assert not term1
    assert env.current_node == 1
    assert env.path_so_far == [0, 1]

    # Hop 2: 1 -> 0 (revisiting node 0, forming a loop!)
    assert next_obs["neighbor_ids"][0] == 0

    next_obs, r2, term2, trunc2, info2 = env.step(action=0)

    assert term2, "Loop must terminate episode"
    assert not trunc2
    assert info2["looped"] is True
    assert not info2["reached_dest"]
    r_hop = info2["r_hop"]
    assert np.isclose(r2, r_hop - 8.0), f"Reward {r2} should equal r_hop ({r_hop}) - 8.0"


# ==============================================================================
# 4. Max Hops Truncation Test (Parameterized)
# ==============================================================================

@pytest.mark.parametrize("target_max_hops", [10, 7])
def test_max_hops_truncation(target_max_hops):
    """Verify episode terminates at max_hops and is flagged as truncated without destination delivery."""
    sim = NSFNETSimEnv(seed=123)
    cfg = RewardConfig(max_hops=target_max_hops)
    env = NetworkRoutingEnv(env=sim, reward_config=cfg, seed=123)

    # Destination is 4 (Boulder)
    # Simple path through distinct nodes:
    # 0 -> 1 -> 2 -> 5 -> 6 -> 11 -> 10 -> 13 -> 9 -> 8 -> 7
    path_targets = [1, 2, 5, 6, 11, 10, 13, 9, 8, 7][:target_max_hops]
    obs, _ = env.reset(src=0, dst=4)

    for step_idx, target_node in enumerate(path_targets):
        # Find which port leads to target_node
        neighbor_ids = list(obs["neighbor_ids"])
        action = neighbor_ids.index(target_node)

        obs, reward, term, trunc, info = env.step(action)

        if step_idx < target_max_hops - 1:
            assert not term, f"Step {step_idx+1} should not terminate"
            assert not trunc, f"Step {step_idx+1} should not truncate"
        else:
            # Last step must truncate
            assert not term
            assert trunc is True
            assert env.done is True

    assert env.hops_taken == target_max_hops
    assert info["reached_dest"] is False
    assert info["truncated"] is True
    assert info["dropped"] is False
    assert info["looped"] is False
    assert info["looped"] is False


# ==============================================================================
# 5. Action Masking and Invalid Port Penalty Test
# ==============================================================================

def test_action_masking_and_fallback_penalty():
    """Verify action_masks() matches valid ports and selecting invalid port applies -2.0 fallback."""
    sim = NSFNETSimEnv(seed=42)
    env = NetworkRoutingEnv(env=sim, seed=42)

    # Node 0 (Seattle) in NSFNET has degree 3 (neighbors: 1, 2, 3)
    obs, _ = env.reset(src=0, dst=13)
    masks = env.action_masks()

    # Degree is 3, d_max is 4 -> exactly 3 valid ports, 1 padded port
    assert np.sum(masks) == 3
    assert masks[3] is False or masks[3] == 0.0
    assert obs["port_mask"][3] == 0.0

    # Attempt to select padded port 3 (invalid action)
    next_obs, reward, terminated, truncated, info = env.step(action=3)

    assert terminated is True
    assert info["invalid_action"] is True
    assert reward == -float(env.reward_config.invalid_action_penalty)
    assert reward == -2.0, f"Expected fallback penalty -2.0, got {reward}"


# ==============================================================================
# 6. Destination Arrival Short-Circuits Drop Check
# ==============================================================================

def test_destination_arrival_short_circuits_drop_check():
    """Verify that reaching destination (v == d) grants +10.0 and short-circuits packet drop trial."""
    sim = build_diamond_sim_env()
    # Heavily congest link (0, 1) in simulation state
    sim.current_loads[(0, 1)] = 10.0
    sim.link_state_engine.loss_max_rate = 1.0  # Max loss probability 100%
    sim.link_state_engine.loss_threshold = 0.0

    env = NetworkRoutingEnv(env=sim, packet_demand_gbps=0.05, seed=42)

    # Multiple runs to verify drop trial never triggers on destination arrival
    for trial_seed in range(10):
        env.reset(seed=trial_seed, src=0, dst=1)
        next_obs, reward, terminated, truncated, info = env.step(action=0)

        assert terminated is True, "Destination arrival must terminate"
        assert info["reached_dest"] is True, "Must reach destination"
        assert info["dropped"] is False, "Destination arrival must short-circuit drop check"
        assert reward > 0.0, f"Reward should include +10.0 destination bonus, got {reward}"
        assert np.isclose(reward, info["r_hop"] + 10.0)


# ==============================================================================
# 7. Intermediate Packet Drop Test (Parameterized)
# ==============================================================================

@pytest.mark.parametrize("cfg_name, w1, w3, drop_base, max_hops", REWARD_CONFIG_CANDIDATES)
def test_intermediate_packet_drop(cfg_name, w1, w3, drop_base, max_hops):
    """Verify that on intermediate hops (v != d), packet loss trial drops packet with calibrated penalty."""
    sim = build_diamond_sim_env()
    # Heavily saturate link (0, 1) and set loss rate to 100%
    sim.current_loads[(0, 1)] = 10.0
    sim.link_state_engine.loss_threshold = 0.0
    sim.link_state_engine.loss_max_rate = 1.0  # Guarantees drop on intermediate hop

    cfg = RewardConfig(
        w1_congestion=w1,
        w2_latency=1.0,
        w3_loss=w3,
        w4_hop_penalty=0.05,
        terminal_drop_penalty_base=drop_base,
        max_hops=max_hops,
    )
    env = NetworkRoutingEnv(env=sim, reward_config=cfg, packet_demand_gbps=0.05, seed=42)
    # Destination is 3 (so reaching 1 is an intermediate hop)
    obs, _ = env.reset(src=0, dst=3)

    next_obs, reward, terminated, truncated, info = env.step(action=0)

    assert terminated is True
    assert info["dropped"] is True
    assert not info["reached_dest"]

    # Penalty formula: -drop_base * (1 + hops / max_hops) where hops=1
    expected_drop_penalty = drop_base * (1.0 + 1.0 / max_hops)
    assert np.isclose(reward, info["r_hop"] - expected_drop_penalty), (
        f"[{cfg_name}] Expected drop penalty delta {expected_drop_penalty}, got {info['r_hop'] - reward}"
    )


def test_progress_potential_reward_shaping():
    """Verify that w5_progress provides positive shaping when stepping closer to destination."""
    sim = build_diamond_sim_env()
    cfg = RewardConfig(w5_progress=2.0)
    env = NetworkRoutingEnv(env=sim, reward_config=cfg, seed=42)
    obs, _ = env.reset(src=0, dst=3)

    # In diamond graph: 0 -> 1 moves closer to dst=3 (hops before: 2, hops after: 1 -> delta = +1)
    next_obs, reward, _, _, info = env.step(action=0)
    # Hand check: r_hop should include +2.0 * (2 - 1) = +2.0
    base_r_hop = -(cfg.w1_congestion * info["congestion_after"] + cfg.w2_latency * info["latency_norm"] + cfg.w3_loss * info["packet_loss_prob"] + cfg.w4_hop_penalty)
    assert np.isclose(info["r_hop"], base_r_hop + 2.0)


def test_env_split_horizon_action_masking():
    """Verify that enable_split_horizon masks out immediate ingress port in action_masks()."""
    sim = build_diamond_sim_env()
    cfg = RewardConfig(enable_split_horizon=True)
    env = NetworkRoutingEnv(env=sim, reward_config=cfg, seed=42)
    obs, _ = env.reset(src=0, dst=3)

    # Step 0 -> 1
    next_obs, _, _, _, _ = env.step(action=0)
    # Node 1 has neighbors [0, 3]. Ingress port is node 0.
    # With split horizon, port pointing back to node 0 should be masked False!
    masks = env.action_masks()
    neighbor_ids = next_obs["neighbor_ids"]
    for idx, nid in enumerate(neighbor_ids):
        if nid == 0:
            assert not masks[idx], "Port pointing back to ingress node 0 must be masked False"
        elif nid == 3:
            assert masks[idx], "Port pointing forward to node 3 must be valid True"


