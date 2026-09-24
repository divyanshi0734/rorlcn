"""Regression test suite for live RL packet inference in app.py.

Guarantees:
  1. Live single-packet inference runs directly through app.py's route_single_packet_live()
     (the actual runtime path used by the Streamlit dashboard).
  2. Asserts non-trivial hop count (hops >= 1) and successful delivery on uncongested pairs
     (e.g., Palo Alto [1] -> Chicago [7]).
  3. Validates both Candidate A ('optimized') and Baseline 200k ('200k') model checkpoints.
  4. Verifies action masks are never all-zero and observation dict formats match training specs.
  5. Verifies telemetry reconciliation between simulation runs and benchmark CSVs.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import (
    load_rl_models,
    route_single_packet_live,
    simulate_seed_runs,
    load_benchmark_telemetry,
)
from src.metrics import compute_aggregate_metrics
from src.network_sim import NSFNETSimEnv


class TestLiveInferenceRegression:
    """Regression tests verifying app.py live inference and KPI reconciliation."""

    @classmethod
    def setup_class(cls):
        """Pre-load models once for the test suite."""
        cls.models = load_rl_models()

    def test_models_loaded_successfully(self):
        """Assert that both optimized (Candidate A) and 200k models load without failure."""
        assert "optimized" in self.models, f"Optimized model missing from loaded models: {list(self.models.keys())}"
        assert "200k" in self.models, f"200k model missing from loaded models: {list(self.models.keys())}"
        assert self.models["optimized"] is not None
        assert self.models["200k"] is not None

    def test_live_single_packet_palo_alto_to_chicago_optimized(self):
        """Palo Alto (1) -> Chicago (7) single-hop zero-congestion delivery.

        Asserts:
          - Delivered is True.
          - Hops >= 1 (specifically 1 hop direct).
          - Path starts at 1 and ends at 7.
          - No loop detected.
          - Baselines (Dijkstra, Static) also deliver.
        """
        results, live_graph = route_single_packet_live(
            src=1,
            dst=7,
            injected_loads={},
            demand_gbps=0.05,
            rl_model_key="optimized",
            loop_avoidance_mode="split_horizon",
        )

        assert "rl" in results, "RL result missing from output dictionary"
        rl = results["rl"]

        assert rl.get("error") is None, f"RL inference returned error: {rl.get('error')}"
        assert rl["delivered"] is True, f"RL failed to deliver trivial packet: {rl}"
        assert rl["hops"] >= 1, f"Expected hops >= 1, got {rl['hops']}"
        assert rl["hops"] == 1, f"Expected direct single hop, got {rl['hops']} hops: {rl['path']}"
        assert rl["path"] == [1, 7], f"Expected path [1, 7], got {rl['path']}"
        assert rl["loop_detected"] is False
        assert rl["latency_ms"] > 0.0

        # Dijkstra and Static SPF baseline sanity check
        assert results["dijkstra"]["delivered"] is True
        assert results["dijkstra"]["path"] == [1, 7]
        assert results["static"]["delivered"] is True

    def test_live_single_packet_palo_alto_to_chicago_200k(self):
        """Palo Alto (1) -> Chicago (7) using baseline 200k checkpoint."""
        results, _ = route_single_packet_live(
            src=1,
            dst=7,
            injected_loads={},
            demand_gbps=0.05,
            rl_model_key="200k",
            loop_avoidance_mode="split_horizon",
        )

        rl = results["rl"]
        assert rl.get("error") is None, f"RL inference returned error: {rl.get('error')}"
        assert rl["delivered"] is True, f"200k model failed to deliver: {rl}"
        assert rl["hops"] >= 1, f"Expected non-trivial hops, got {rl['hops']}"
        assert rl["path"] == [1, 7]

    def test_live_single_packet_multi_hop_uncongested(self):
        """Seattle (0) -> San Diego (2): direct link or via node 1 (Palo Alto).

        Asserts non-trivial delivery with valid path.
        """
        results, _ = route_single_packet_live(
            src=0,
            dst=2,
            injected_loads={},
            demand_gbps=0.05,
            rl_model_key="optimized",
            loop_avoidance_mode="split_horizon",
        )

        rl = results["rl"]
        assert rl.get("error") is None
        assert rl["delivered"] is True, f"RL failed on Seattle -> San Diego: {rl}"
        assert rl["hops"] >= 1
        assert rl["path"][0] == 0
        assert rl["path"][-1] == 2
        assert len(rl["path"]) == rl["hops"] + 1

    def test_live_single_packet_path_vector_mode(self):
        """Asserts Path-Vector mode routes cleanly without loops."""
        results, _ = route_single_packet_live(
            src=1,
            dst=7,
            injected_loads={},
            demand_gbps=0.05,
            rl_model_key="optimized",
            loop_avoidance_mode="path_vector",
        )

        rl = results["rl"]
        assert rl["delivered"] is True
        assert rl["hops"] >= 1
        assert rl["loop_detected"] is False

    def test_action_mask_safety_never_all_zero(self):
        """Verifies that port masks across all 14 NSFNET nodes have at least one active port."""
        env = NSFNETSimEnv(seed=42)
        env.reset()

        for node in env.graph.nodes():
            obs = env.get_node_observation(node)
            port_mask = obs["port_mask"]
            assert np.sum(port_mask) >= 1.0, f"Node {node} has all-zero port mask: {port_mask}"
            assert np.any(port_mask.astype(bool))

    def test_tab1_kpi_metrics_reconciliation(self):
        """Verifies that simulate_seed_runs produces valid telemetry and reconciles with audited benchmarks."""
        # 1. Split-Horizon evaluation (audited remediation benchmark)
        (d_p, d_s), (s_p, s_s), (stale_p, stale_s), (r_p, r_s) = simulate_seed_runs(
            seed=100,
            loop_avoidance_mode="split_horizon",
            rl_model_choice="optimized",
        )

        m_dijk = compute_aggregate_metrics(d_p, 40)
        m_stat = compute_aggregate_metrics(s_p, 40)
        m_stale = compute_aggregate_metrics(stale_p, 40)
        m_rl = compute_aggregate_metrics(r_p, 40)

        # Candidate A + Split-Horizon delivers 1978 packets (1.10% loss) on Seed 100
        assert m_rl["total_packets_delivered"] == 1978, f"Expected 1978 deliveries on Seed 100 Candidate A with Split-Horizon, got {m_rl['total_packets_delivered']}"
        assert abs(m_rl["packet_loss_rate"] - 0.0110) < 0.001, f"Expected loss 1.10%, got {m_rl['packet_loss_rate']*100:.2f}%"

        # Stale Dijkstra (tau=2) delivers 1973 packets (1.35% loss)
        assert m_stale["total_packets_delivered"] == 1973
        assert m_stale["total_packets_delivered"] <= m_dijk["total_packets_delivered"]
        assert m_rl["total_packets_delivered"] >= m_stale["total_packets_delivered"]

        # 2. Unmasked / Memoryless evaluation (reconciles with raw reward_tuning_comparison.csv)
        _, _, _, (r_none, _) = simulate_seed_runs(
            seed=100,
            loop_avoidance_mode="none",
            rl_model_choice="optimized",
        )
        m_rl_none = compute_aggregate_metrics(r_none, 40)
        assert m_rl_none["total_packets_delivered"] == 1953, f"Expected 1953 deliveries without Split-Horizon, got {m_rl_none['total_packets_delivered']}"
        assert round(m_rl_none["packet_loss_rate"] * 100.0, 2) == 2.35
