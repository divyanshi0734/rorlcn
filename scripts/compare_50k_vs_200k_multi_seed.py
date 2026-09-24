"""Multi-Seed Evaluation Script: Comparing MaskablePPO 50k vs 200k Checkpoints.

Runs evaluations across multiple held-out seeds to determine whether
the delivery rate difference between 50k and 200k models is a consistent
systematic effect or sample noise.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sb3_contrib import MaskablePPO

# Ensure RORL root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.network_sim import NSFNETSimEnv
from src.dijkstra import DijkstraRouter
from src.packet_traffic import PacketTrafficGenerator, RoutingSimulationRunner
from src.metrics import compute_aggregate_metrics


def evaluate_rl_seed(
    model: MaskablePPO,
    seed: int,
    total_timesteps: int = 40,
    packets_per_step: int = 50,
    spike_timestep: int = 20,
    spike_edge: tuple = (1, 7),
    spike_magnitude: float = 5.0,
) -> Dict[str, Any]:
    spikes = {spike_timestep: (spike_edge[0], spike_edge[1], spike_magnitude)}
    env = NSFNETSimEnv(seed=seed)
    gen = PacketTrafficGenerator(seed=seed, packets_per_step=packets_per_step)
    router = DijkstraRouter(default_weight_attr="latency")
    runner = RoutingSimulationRunner(
        env=env,
        packet_gen=gen,
        router=router,
        rl_model=model,
        enable_packet_feedback=True,
        seed=seed,
    )
    pkts_df, steps_df = runner.run_simulation(
        num_timesteps=total_timesteps,
        routing_mode="rl_agent",
        spike_schedule=spikes,
    )
    metrics = compute_aggregate_metrics(pkts_df, total_timesteps=total_timesteps)
    return {
        "seed": seed,
        "delivered": int(metrics["total_packets_delivered"]),
        "dropped": int(metrics["total_packets_dropped"]),
        "loss_rate_pct": float(metrics["packet_loss_rate"]) * 100.0,
        "mean_latency_ms": float(metrics["mean_latency_ms"]),
        "median_latency_ms": float(metrics["median_latency_ms"]),
        "p95_latency_ms": float(metrics["p95_latency_ms"]),
        "max_latency_ms": float(metrics["max_latency_ms"]),
        "mean_queuing_delay_ms": float(metrics["mean_queuing_delay_ms"]),
        "mean_hops": float(metrics["mean_hop_count"]),
    }


def main():
    model_50k_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model_50k.zip")
    model_200k_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model_200k.zip")

    if not os.path.exists(model_50k_path) or not os.path.exists(model_200k_path):
        print(f"Error: Model files not found. Checked {model_50k_path} and {model_200k_path}")
        sys.exit(1)

    print("[*] Loading MaskablePPO checkpoints...")
    model_50k = MaskablePPO.load(model_50k_path)
    model_200k = MaskablePPO.load(model_200k_path)

    # Seeds to test:
    # 42 (in-distribution training seed)
    # 100 (primary held-out seed)
    # 200, 300, 400 (3 additional independent held-out seeds)
    seeds = [42, 100, 200, 300, 400]

    results_50k = []
    results_200k = []

    print(f"[*] Starting multi-seed evaluation across {len(seeds)} seeds: {seeds}")
    for seed in seeds:
        print(f"\n--- Evaluating Seed {seed} ---")
        m50 = evaluate_rl_seed(model_50k, seed)
        m200 = evaluate_rl_seed(model_200k, seed)

        results_50k.append(m50)
        results_200k.append(m200)

        diff_deliv = m200["delivered"] - m50["delivered"]
        sign = "+" if diff_deliv >= 0 else ""
        print(
            f"Seed {seed:3d} | 50k Delivered: {m50['delivered']} ({m50['loss_rate_pct']:.2f}% loss) | "
            f"200k Delivered: {m200['delivered']} ({m200['loss_rate_pct']:.2f}% loss) | "
            f"Diff: {sign}{diff_deliv} packets | "
            f"Mean Latency: 50k={m50['mean_latency_ms']:.2f}ms vs 200k={m200['mean_latency_ms']:.2f}ms"
        )

    df_50k = pd.DataFrame(results_50k)
    df_200k = pd.DataFrame(results_200k)

    print("\n" + "=" * 90)
    print("MULTI-SEED EVALUATION SUMMARY: MASKABLEPPO 50K VS. 200K")
    print("=" * 90)
    print(f"{'Seed':<8} | {'50k Delivered':<14} | {'200k Delivered':<14} | {'Deliv Diff':<12} | {'50k Lat (ms)':<14} | {'200k Lat (ms)':<14}")
    print("-" * 90)

    for s, r50, r200 in zip(seeds, results_50k, results_200k):
        diff = r200["delivered"] - r50["delivered"]
        sign = "+" if diff >= 0 else ""
        print(
            f"{s:<8} | {r50['delivered']:<14} | {r200['delivered']:<14} | "
            f"{sign + str(diff):<12} | {r50['mean_latency_ms']:<14.2f} | {r200['mean_latency_ms']:<14.2f}"
        )

    print("-" * 90)
    mean_deliv_50 = df_50k["delivered"].mean()
    std_deliv_50 = df_50k["delivered"].std()
    mean_deliv_200 = df_200k["delivered"].mean()
    std_deliv_200 = df_200k["delivered"].std()

    mean_lat_50 = df_50k["mean_latency_ms"].mean()
    std_lat_50 = df_50k["mean_latency_ms"].std()
    mean_lat_200 = df_200k["mean_latency_ms"].mean()
    std_lat_200 = df_200k["mean_latency_ms"].std()

    mean_hops_50 = df_50k["mean_hops"].mean()
    mean_hops_200 = df_200k["mean_hops"].mean()

    mean_q_50 = df_50k["mean_queuing_delay_ms"].mean()
    mean_q_200 = df_200k["mean_queuing_delay_ms"].mean()

    diff_mean_deliv = mean_deliv_200 - mean_deliv_50
    diff_sign = "+" if diff_mean_deliv >= 0 else ""

    print(
        f"{'Mean±Std':<8} | {mean_deliv_50:.1f} ± {std_deliv_50:.1f}   | {mean_deliv_200:.1f} ± {std_deliv_200:.1f}   | "
        f"{diff_sign}{diff_mean_deliv:.1f} pkts   | {mean_lat_50:.2f} ± {std_lat_50:.2f}  | {mean_lat_200:.2f} ± {std_lat_200:.2f}"
    )
    print("=" * 90)

    # Save summary CSV
    summary_path = os.path.join(PROJECT_ROOT, "assets", "multi_seed_50k_vs_200k_comparison.csv")
    combined_rows = []
    for s, r50, r200 in zip(seeds, results_50k, results_200k):
        combined_rows.append({
            "seed": s,
            "seed_type": "in_distribution" if s == 42 else "held_out",
            "50k_delivered": r50["delivered"],
            "200k_delivered": r200["delivered"],
            "delivered_diff": r200["delivered"] - r50["delivered"],
            "50k_loss_rate_pct": r50["loss_rate_pct"],
            "200k_loss_rate_pct": r200["loss_rate_pct"],
            "50k_mean_latency_ms": r50["mean_latency_ms"],
            "200k_mean_latency_ms": r200["mean_latency_ms"],
            "50k_mean_hops": r50["mean_hops"],
            "200k_mean_hops": r200["mean_hops"],
            "50k_mean_queuing_ms": r50["mean_queuing_delay_ms"],
            "200k_mean_queuing_ms": r200["mean_queuing_delay_ms"],
        })
    pd.DataFrame(combined_rows).to_csv(summary_path, index=False)
    print(f"[*] Saved multi-seed comparison summary to: {summary_path}")


if __name__ == "__main__":
    main()
