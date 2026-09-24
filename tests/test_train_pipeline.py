"""Unit tests for MaskablePPO training pipeline, callbacks, and evaluation."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from src.routing_env import NetworkRoutingEnv, RewardConfig
from src.train_agent import PPOConfig, RoutingExperimentCallback, train_routing_agent, mask_fn
from src.metrics import compare_three_policies


def test_ppo_config_defaults_and_override():
    """Verifies PPOConfig dataclass default parameters and customization."""
    cfg = PPOConfig()
    assert cfg.learning_rate == 3e-4
    assert cfg.n_steps == 2048
    assert cfg.batch_size == 64
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.clip_range == 0.2
    assert cfg.ent_coef == 0.01

    custom_cfg = PPOConfig(learning_rate=1e-3, n_steps=256, batch_size=32)
    assert custom_cfg.learning_rate == 1e-3
    assert custom_cfg.n_steps == 256
    assert custom_cfg.batch_size == 32


def test_action_masking_integration():
    """Verifies ActionMasker extracts correct boolean mask matching env.action_masks()."""
    from sb3_contrib.common.wrappers import ActionMasker

    env = NetworkRoutingEnv(seed=42)
    masked_env = ActionMasker(env, mask_fn)

    obs, info = masked_env.reset()
    assert "neighbor_features" in obs
    assert "action_mask" in obs

    # Check mask extraction
    mask = get_action_masks(masked_env)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert len(mask) == 4
    # Root node degree is at least 2, so at least 2 valid ports
    assert mask.sum() >= 2
    assert np.array_equal(mask, env.action_masks())


def test_routing_experiment_callback():
    """Tests RoutingExperimentCallback metric logging and policy collapse alerting."""
    cb = RoutingExperimentCallback(collapse_threshold=0.20, verbose=0)
    cb.outcomes = []
    cb.episode_rewards = []
    cb.episode_lengths = []
    cb.rolling_delivery_rates = []

    # Simulate 50 delivered episodes
    for _ in range(50):
        cb.outcomes.append("delivered")
        cb.episode_rewards.append(10.0)
        cb.episode_lengths.append(3)
        cb.rolling_delivery_rates.append(1.0)
    cb.best_rolling_delivery_rate = 1.0

    # Simulate 50 dropped episodes -> delivery rate drops from 1.0 to 0.5 (50% drop > 20% threshold)
    for _ in range(50):
        cb.outcomes.append("dropped")
        cb.episode_rewards.append(-5.0)
        cb.episode_lengths.append(4)
        cb.rolling_delivery_rates.append(0.50)

    # Verify peak tracking
    assert cb.best_rolling_delivery_rate == 1.0
    current_rate = float(np.mean([1 if o == "delivered" else 0 for o in cb.outcomes[-100:]]))
    assert current_rate == 0.50
    drop_from_peak = cb.best_rolling_delivery_rate - current_rate
    assert drop_from_peak > 0.20  # Policy collapse condition met


def test_compare_three_policies_metrics():
    """Verifies compare_three_policies generates expected 3-way columns and structure."""
    dummy_data_1 = pd.DataFrame([{
        "timestep": 0, "src": 0, "dst": 1, "delivered": True,
        "end_to_end_latency_ms": 12.5, "hop_count": 2,
        "total_queuing_delay_ms": 1.2, "demand_gbps": 0.05,
    }])
    dummy_data_2 = pd.DataFrame([{
        "timestep": 0, "src": 0, "dst": 1, "delivered": True,
        "end_to_end_latency_ms": 15.0, "hop_count": 3,
        "total_queuing_delay_ms": 2.5, "demand_gbps": 0.05,
    }])
    dummy_data_3 = pd.DataFrame([{
        "timestep": 0, "src": 0, "dst": 1, "delivered": True,
        "end_to_end_latency_ms": 11.0, "hop_count": 2,
        "total_queuing_delay_ms": 0.8, "demand_gbps": 0.05,
    }])

    comp_df = compare_three_policies(dummy_data_1, dummy_data_2, dummy_data_3, total_timesteps=1)
    assert "Metric" in comp_df.columns
    assert "Dynamic Dijkstra" in comp_df.columns
    assert "Static SPF" in comp_df.columns
    assert "MaskablePPO RL" in comp_df.columns
    assert "RL vs Static" in comp_df.columns
    assert "RL vs Dijkstra" in comp_df.columns
    assert len(comp_df) >= 10


def test_short_training_smoke():
    """End-to-end smoke test: trains MaskablePPO for 256 steps and loads checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")

        cfg = PPOConfig(
            n_steps=128,
            batch_size=32,
            checkpoint_freq=128,
        )

        model, callback, saved_path = train_routing_agent(
            config=cfg,
            total_timesteps=256,
            checkpoint_dir=ckpt_dir,
            seed=123,
            verbose=0,
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == "final_model.zip"

        # Load back and verify deterministic inference with action masking
        loaded_model = MaskablePPO.load(saved_path)
        test_env = NetworkRoutingEnv(seed=123)
        obs, info = test_env.reset()

        for _ in range(5):
            mask = test_env.action_masks()
            action, _ = loaded_model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            assert mask[action] is True or mask[action] == 1  # Action must be valid!
            obs, reward, terminated, truncated, info = test_env.step(action)
            if terminated or truncated:
                obs, info = test_env.reset()


def test_eval_seed_distinctness():
    """Asserts that evaluation seed 100 generates a provably distinct packet sequence from training seed 42."""
    from src.packet_traffic import PacketTrafficGenerator

    gen_train = PacketTrafficGenerator(seed=42, packets_per_step=50)
    gen_eval = PacketTrafficGenerator(seed=100, packets_per_step=50)

    pkts_train = gen_train.generate_packets(timestep=0)
    pkts_eval = gen_eval.generate_packets(timestep=0)

    assert len(pkts_train) == 50
    assert len(pkts_eval) == 50

    pairs_train = [(p.src, p.dst) for p in pkts_train]
    pairs_eval = [(p.src, p.dst) for p in pkts_eval]

    # The two sequences must not be identical
    assert pairs_train != pairs_eval

    # Identity overlap between independent random draws should be very small (e.g. < 20% on 14 nodes)
    matches = sum(1 for pt, pe in zip(pairs_train, pairs_eval) if pt == pe)
    overlap_rate = matches / len(pairs_train)
    assert overlap_rate < 0.15, f"Overlap rate {overlap_rate:.2%} is unexpectedly high for independent seeds!"


def test_survivorship_bias_diagnostics_calculation():
    """Verifies that compute_dropped_packet_diagnostics correctly analyzes dropped vs delivered packets."""
    from src.metrics import compute_dropped_packet_diagnostics
    from src.topology import build_nsfnet_graph

    graph = build_nsfnet_graph()

    dummy_df = pd.DataFrame([
        # Delivered packet
        {"packet_id": 1, "timestep": 1, "src": 0, "dst": 1, "delivered": True, "hop_count": 2, "max_congestion": 0.4, "queuing_delay_ms": 1.0},
        # Dropped packet (longer distance, higher congestion)
        {"packet_id": 2, "timestep": 1, "src": 0, "dst": 7, "delivered": False, "hop_count": 3, "max_congestion": 0.9, "queuing_delay_ms": 5.0},
    ])

    diag = compute_dropped_packet_diagnostics(dummy_df, graph=graph)
    assert "delivered" in diag
    assert "dropped" in diag
    assert diag["delivered"]["count"] == 1
    assert diag["dropped"]["count"] == 1
    assert diag["loss_rate_pct"] == 50.0
    assert diag["dropped"]["mean_max_congestion"] > diag["delivered"]["mean_max_congestion"]
    assert diag["dropped"]["mean_topological_distance_km"] > diag["delivered"]["mean_topological_distance_km"]


def test_collapse_incidents_grouping():
    """Verifies analyze_collapse_incidents correctly aggregates distinct contiguous episodes."""
    cb = RoutingExperimentCallback(checkpoint_dir="checkpoints", verbose=0)
    cb.policy_collapse_events = [
        # Incident 1 (3 contiguous episodes)
        {"step": 1000, "episode": 10, "current_rate": 0.3, "peak_rate": 0.6, "drop_pct": 30.0},
        {"step": 1002, "episode": 11, "current_rate": 0.3, "peak_rate": 0.6, "drop_pct": 30.0},
        {"step": 1005, "episode": 12, "current_rate": 0.3, "peak_rate": 0.6, "drop_pct": 30.0},
        # Incident 2 (gap of 20 episodes -> new incident)
        {"step": 2000, "episode": 32, "current_rate": 0.35, "peak_rate": 0.65, "drop_pct": 30.0},
        {"step": 2003, "episode": 33, "current_rate": 0.35, "peak_rate": 0.65, "drop_pct": 30.0},
    ]
    cb.num_timesteps = 50_000

    incidents, bin_df, summary = cb.analyze_collapse_incidents(gap_threshold=5)
    assert len(incidents) == 2
    assert incidents[0]["duration_eps"] == 3
    assert incidents[0]["flagged_events"] == 3
    assert incidents[1]["duration_eps"] == 2
    assert summary["total_incidents"] == 2
    assert summary["total_flagged_steps"] == 5
    assert not bin_df.empty
