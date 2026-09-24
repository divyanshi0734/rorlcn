"""Reward-Tuning Experiment for Packet-Loss Gap Reduction.

Trains and evaluates 3 reward reweighting candidates plus 80k baseline control:
  - Baseline (80k):    w1=3.0, w3=2.0, drop_base=5.0, max_hops=10
  - Candidate A (80k):   w1=2.0, w3=4.0, drop_base=5.0, max_hops=10
  - Candidate B (80k):   w1=3.0, w3=2.0, drop_base=8.0, max_hops=10
  - Candidate C (80k):   w1=2.0, w3=4.0, drop_base=8.0, max_hops=7

Evaluates on the 5-seed battery (42, 100, 200, 300, 400), leading with the 4 held-out
seeds (100, 200, 300, 400) and reporting Seed 42 strictly as in-distribution reference.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO

# Ensure RORL root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.network_sim import NSFNETSimEnv
from src.dijkstra import DijkstraRouter
from src.packet_traffic import PacketTrafficGenerator, RoutingSimulationRunner
from src.metrics import compute_aggregate_metrics
from src.routing_env import RewardConfig
from src.train_agent import PPOConfig, train_routing_agent


CANDIDATE_CONFIGS = {
    "baseline_80k": {
        "name": "Baseline (80k)",
        "reward_config": RewardConfig(w1_congestion=3.0, w3_loss=2.0, terminal_drop_penalty_base=5.0, max_hops=10),
        "max_hops": 10,
        "description": "w1=3.0, w3=2.0, drop_base=5.0, max_hops=10",
    },
    "candidate_a": {
        "name": "Candidate A (80k)",
        "reward_config": RewardConfig(w1_congestion=2.0, w3_loss=4.0, terminal_drop_penalty_base=5.0, max_hops=10),
        "max_hops": 10,
        "description": "w1=2.0, w3=4.0, drop_base=5.0, max_hops=10",
    },
    "candidate_b": {
        "name": "Candidate B (80k)",
        "reward_config": RewardConfig(w1_congestion=3.0, w3_loss=2.0, terminal_drop_penalty_base=8.0, max_hops=10),
        "max_hops": 10,
        "description": "w1=3.0, w3=2.0, drop_base=8.0, max_hops=10",
    },
    "candidate_c": {
        "name": "Candidate C (80k)",
        "reward_config": RewardConfig(w1_congestion=2.0, w3_loss=4.0, terminal_drop_penalty_base=8.0, max_hops=7),
        "max_hops": 7,
        "description": "w1=2.0, w3=4.0, drop_base=8.0, max_hops=7",
    },
}

EVAL_SEEDS = [42, 100, 200, 300, 400]
HELDOUT_SEEDS = [100, 200, 300, 400]
IN_DIST_SEED = 42


def train_candidates(checkpoint_dir: str = "checkpoints/reward_tuning", force_retrain: bool = False):
    """Trains each candidate configuration from scratch on seed 42 for 80,000 steps."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    for key, spec in CANDIDATE_CONFIGS.items():
        model_file = os.path.join(checkpoint_dir, f"{key}.zip")
        if os.path.exists(model_file) and not force_retrain:
            print(f"[*] Found existing checkpoint for {spec['name']}: {model_file}. Skipping training.")
            continue

        print("\n" + "=" * 80)
        print(f"[*] Training {spec['name']} from scratch (80,000 steps, seed 42)")
        print(f"    Config: {spec['description']}")
        print("=" * 80)

        ppo_cfg = PPOConfig(
            total_timesteps=80_000,
            seed=42,
            checkpoint_freq=40_000,
        )

        sub_ckpt_dir = os.path.join(checkpoint_dir, key)
        os.makedirs(sub_ckpt_dir, exist_ok=True)

        t0 = time.time()
        model, callback, final_path = train_routing_agent(
            config=ppo_cfg,
            reward_config=spec["reward_config"],
            checkpoint_dir=sub_ckpt_dir,
            verbose=1,
        )
        t_elapsed = time.time() - t0
        print(f"[✓] Completed training for {spec['name']} in {t_elapsed:.1f}s")

        # Save main candidate checkpoint
        model.save(model_file)
        print(f"[✓] Saved model to: {model_file}")


def evaluate_policy_seed(
    policy_name: str,
    routing_mode: str,
    seed: int,
    rl_model: Any = None,
    max_hops: int = 10,
    total_timesteps: int = 40,
    packets_per_step: int = 50,
    spike_timestep: int = 20,
    spike_edge: Tuple[int, int] = (1, 7),
    spike_magnitude: float = 5.0,
) -> Dict[str, Any]:
    """Evaluates a policy on a given seed with traffic shock at t=20 on (1, 7)."""
    spikes = {spike_timestep: (spike_edge[0], spike_edge[1], spike_magnitude)}
    env = NSFNETSimEnv(seed=seed)
    gen = PacketTrafficGenerator(seed=seed, packets_per_step=packets_per_step)
    router = DijkstraRouter(default_weight_attr="latency")

    runner = RoutingSimulationRunner(
        env=env,
        packet_gen=gen,
        router=router,
        rl_model=rl_model,
        enable_packet_feedback=True,
        seed=seed,
        max_hops=max_hops,
    )

    pkts_df, steps_df = runner.run_simulation(
        num_timesteps=total_timesteps,
        routing_mode=routing_mode,
        spike_schedule=spikes,
    )

    metrics = compute_aggregate_metrics(pkts_df, total_timesteps=total_timesteps)

    return {
        "policy": policy_name,
        "seed": seed,
        "seed_type": "in_distribution" if seed == IN_DIST_SEED else "held_out",
        "delivered": int(metrics["total_packets_delivered"]),
        "dropped": int(metrics["total_packets_dropped"]),
        "loss_rate_pct": float(metrics["packet_loss_rate"]) * 100.0,
        "mean_latency_ms": float(metrics["mean_latency_ms"]),
        "median_latency_ms": float(metrics["median_latency_ms"]),
        "p95_latency_ms": float(metrics["p95_latency_ms"]),
        "max_latency_ms": float(metrics["max_latency_ms"]),
        "mean_queuing_delay_ms": float(metrics["mean_queuing_delay_ms"]),
        "mean_hops": float(metrics["mean_hop_count"]),
        "throughput_gbps": float(metrics["mean_throughput_aggregate_gbps"]),
    }


def run_full_evaluation_battery(
    checkpoint_dir: str = "checkpoints/reward_tuning",
    output_csv: str = "assets/reward_tuning_comparison.csv",
    output_plot: str = "assets/reward_tuning_comparison.png",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the 5-seed evaluation battery across all candidates and baselines."""
    print("\n" + "=" * 90)
    print("EXECUTING 5-SEED EVALUATION BATTERY (Seeds: 42, 100, 200, 300, 400)")
    print("=" * 90)

    # 1. Prepare policies to evaluate
    policies = []

    # Dynamic Dijkstra
    policies.append({
        "id": "dynamic_dijkstra",
        "name": "Dynamic Dijkstra",
        "mode": "dynamic_dijkstra",
        "model": None,
        "max_hops": 10,
    })

    # Static SPF
    policies.append({
        "id": "static_spf",
        "name": "Static SPF",
        "mode": "static_spf",
        "model": None,
        "max_hops": 10,
    })

    # 200k Reference Baseline
    ref_200k_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model_200k.zip")
    if os.path.exists(ref_200k_path):
        print(f"[*] Loading 200k Baseline Reference: {ref_200k_path}")
        m_200k = MaskablePPO.load(ref_200k_path)
        policies.append({
            "id": "baseline_200k",
            "name": "Baseline RL (200k Reference)",
            "mode": "rl_agent",
            "model": m_200k,
            "max_hops": 10,
        })

    # The 4 trained candidates
    for key, spec in CANDIDATE_CONFIGS.items():
        m_path = os.path.join(checkpoint_dir, f"{key}.zip")
        if not os.path.exists(m_path):
            print(f"[!] Warning: Checkpoint {m_path} not found. Skipping {spec['name']}.")
            continue
        print(f"[*] Loading candidate: {spec['name']} from {m_path}")
        m = MaskablePPO.load(m_path)
        policies.append({
            "id": key,
            "name": spec["name"],
            "mode": "rl_agent",
            "model": m,
            "max_hops": spec["max_hops"],
        })

    all_results = []

    for p in policies:
        print(f"\n--- Evaluating Policy: {p['name']} ---")
        for s in EVAL_SEEDS:
            res = evaluate_policy_seed(
                policy_name=p["name"],
                routing_mode=p["mode"],
                seed=s,
                rl_model=p["model"],
                max_hops=p["max_hops"],
            )
            all_results.append(res)
            print(f"  Seed {s:3d} ({res['seed_type']:<15}): Delivered {res['delivered']:4d}/2000 "
                  f"({res['loss_rate_pct']:.2f}% loss) | Latency: {res['mean_latency_ms']:.2f}ms | Hops: {res['mean_hops']:.2f}")

    df_raw = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_raw.to_csv(output_csv, index=False)
    print(f"\n[*] Raw per-seed telemetry saved to: {output_csv}")

    # Compute Aggregations:
    # 1. Mean across the 4 held-out seeds only (100, 200, 300, 400) - LEAD METRIC
    # 2. Seed 42 reported strictly separately as In-Distribution Reference
    summary_rows = []

    for p in policies:
        p_name = p["name"]
        df_p = df_raw[df_raw["policy"] == p_name]

        df_heldout = df_p[df_p["seed"].isin(HELDOUT_SEEDS)]
        df_indist = df_p[df_p["seed"] == IN_DIST_SEED]

        row = {
            "Policy": p_name,
            # Held-out 4-seed statistics (LEAD METRICS)
            "Held-Out Delivered (Mean±Std)": f"{df_heldout['delivered'].mean():.1f} ± {df_heldout['delivered'].std():.1f}",
            "Held-Out Loss Rate (%)": f"{df_heldout['loss_rate_pct'].mean():.2f}% ± {df_heldout['loss_rate_pct'].std():.2f}%",
            "Held-Out Mean Latency (ms)": f"{df_heldout['mean_latency_ms'].mean():.2f} ± {df_heldout['mean_latency_ms'].std():.2f}",
            "Held-Out P95 Latency (ms)": f"{df_heldout['p95_latency_ms'].mean():.2f} ± {df_heldout['p95_latency_ms'].std():.2f}",
            "Held-Out Mean Hops": f"{df_heldout['mean_hops'].mean():.2f} ± {df_heldout['mean_hops'].std():.2f}",
            "Held-Out Queuing (ms)": f"{df_heldout['mean_queuing_delay_ms'].mean():.2f} ± {df_heldout['mean_queuing_delay_ms'].std():.2f}",
            # Raw held-out numerical means for plotting/comparisons
            "_heldout_loss_mean": df_heldout['loss_rate_pct'].mean(),
            "_heldout_lat_mean": df_heldout['mean_latency_ms'].mean(),
            "_heldout_hops_mean": df_heldout['mean_hops'].mean(),
            "_heldout_deliv_mean": df_heldout['delivered'].mean(),
            # In-distribution seed 42 reference
            "Seed 42 (In-Dist) Delivered": int(df_indist["delivered"].values[0]) if not df_indist.empty else None,
            "Seed 42 (In-Dist) Loss Rate (%)": f"{df_indist['loss_rate_pct'].values[0]:.2f}%" if not df_indist.empty else None,
            "Seed 42 (In-Dist) Mean Latency (ms)": f"{df_indist['mean_latency_ms'].values[0]:.2f}" if not df_indist.empty else None,
            "Seed 42 (In-Dist) Mean Hops": f"{df_indist['mean_hops'].values[0]:.2f}" if not df_indist.empty else None,
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 115)
    print("SCOPED REWARD-TUNING SUMMARY: 4 HELD-OUT SEEDS (LEAD METRIC) VS. SEED 42 (IN-DISTRIBUTION)")
    print("=" * 115)
    display_cols = [
        "Policy",
        "Held-Out Loss Rate (%)",
        "Held-Out Mean Latency (ms)",
        "Held-Out P95 Latency (ms)",
        "Held-Out Mean Hops",
        "Seed 42 (In-Dist) Loss Rate (%)",
        "Seed 42 (In-Dist) Mean Latency (ms)",
    ]
    print(df_summary[display_cols].to_string(index=False))
    print("=" * 115)

    # Save formatted summary CSV
    summary_csv = output_csv.replace(".csv", "_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"[*] Summary table saved to: {summary_csv}")

    # Plot 4-panel comparison figure
    plot_reward_tuning_comparison(df_summary, df_raw, output_plot)

    return df_raw, df_summary


def plot_reward_tuning_comparison(df_summary: pd.DataFrame, df_raw: pd.DataFrame, output_plot: str):
    """Generates a 4-panel figure comparing the reward tuning candidates across key metrics."""
    os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    policies = df_summary["Policy"].tolist()
    y_pos = np.arange(len(policies))

    palette = [
        "#38bdf8",  # Dynamic Dijkstra: Cyan
        "#f43f5e",  # Static SPF: Rose
        "#a855f7",  # 200k Reference: Purple
        "#64748b",  # Baseline 80k: Slate
        "#3b82f6",  # Candidate A: Blue
        "#f59e0b",  # Candidate B: Amber
        "#10b981",  # Candidate C: Emerald
    ][:len(policies)]

    # 1. Packet Loss Rate (%) (Lower is better)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#1e293b")
    loss_vals = df_summary["_heldout_loss_mean"].tolist()
    bars1 = ax1.barh(y_pos, loss_vals, color=palette, edgecolor="#ffffff", alpha=0.85, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(policies, color="#f8fafc", fontsize=9, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Held-Out Packet Loss Rate (%) [Lower is Better]", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax1.set_title("1. Packet Loss Rate (Mean Across 4 Held-Out Seeds)", color="#f8fafc", fontsize=11, fontweight="bold", loc="left")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.tick_params(colors="#cbd5e1")
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.15, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va="center", color="#f8fafc", fontsize=8.5, fontweight="bold")
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # 2. Mean Latency (ms) (Lower is better)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1e293b")
    lat_vals = df_summary["_heldout_lat_mean"].tolist()
    bars2 = ax2.barh(y_pos, lat_vals, color=palette, edgecolor="#ffffff", alpha=0.85, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(policies))
    ax2.invert_yaxis()
    ax2.set_xlabel("Held-Out Mean Latency (ms) [Lower is Better]", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax2.set_title("2. End-to-End Mean Latency (4 Held-Out Seeds)", color="#f8fafc", fontsize=11, fontweight="bold", loc="left")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.tick_params(colors="#cbd5e1")
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.25, bar.get_y() + bar.get_height()/2, f"{w:.2f}ms", va="center", color="#f8fafc", fontsize=8.5, fontweight="bold")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # 3. Mean Hop Count (Efficiency)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1e293b")
    hops_vals = df_summary["_heldout_hops_mean"].tolist()
    bars3 = ax3.barh(y_pos, hops_vals, color=palette, edgecolor="#ffffff", alpha=0.85, height=0.6)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(policies, color="#f8fafc", fontsize=9, fontweight="bold")
    ax3.invert_yaxis()
    ax3.set_xlabel("Held-Out Mean Hops [Path Efficiency]", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax3.set_title("3. Mean Hop Count Across 4 Held-Out Seeds", color="#f8fafc", fontsize=11, fontweight="bold", loc="left")
    ax3.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax3.tick_params(colors="#cbd5e1")
    for bar in bars3:
        w = bar.get_width()
        ax3.text(w + 0.03, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", color="#f8fafc", fontsize=8.5, fontweight="bold")
    for spine in ax3.spines.values():
        spine.set_color("#334155")

    # 4. In-Distribution (Seed 42) vs. Held-Out (Seeds 100-400) Loss Rate Gap
    ax4 = axes[1, 1]
    ax4.set_facecolor("#1e293b")
    indist_losses = []
    for p in policies:
        v = df_summary.loc[df_summary["Policy"] == p, "Seed 42 (In-Dist) Loss Rate (%)"].values[0]
        indist_losses.append(float(v.replace("%", "")) if v is not None else 0.0)

    x = np.arange(len(policies))
    width = 0.35
    ax4.bar(x - width/2, indist_losses, width, label="Seed 42 (In-Dist)", color="#f59e0b", alpha=0.85)
    ax4.bar(x + width/2, loss_vals, width, label="Held-Out Mean (100-400)", color="#38bdf8", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels([p.split(" (")[0] for p in policies], rotation=25, ha="right", color="#cbd5e1", fontsize=8.5)
    ax4.set_ylabel("Loss Rate (%)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax4.set_title("4. Memorization Audit: In-Distribution vs. Held-Out Loss Rate", color="#f8fafc", fontsize=11, fontweight="bold", loc="left")
    ax4.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax4.tick_params(colors="#cbd5e1")
    ax4.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8.5)
    for spine in ax4.spines.values():
        spine.set_color("#334155")

    fig.suptitle(
        "Scoped Reward-Tuning Experiment: Closing the Packet Loss Gap\n"
        "[Evaluation across 4 Held-Out Seeds (100, 200, 300, 400) with Seed 42 Reference]",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    plt.tight_layout()
    plt.savefig(output_plot, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Comparison figure saved to: {output_plot}")


def main():
    parser = argparse.ArgumentParser(description="Scoped Reward-Tuning Experiment for Packet Loss Gap")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/reward_tuning", help="Directory for candidate checkpoints")
    parser.add_argument("--output-csv", type=str, default="assets/reward_tuning_comparison.csv", help="Output comparison CSV path")
    parser.add_argument("--output-plot", type=str, default="assets/reward_tuning_comparison.png", help="Output comparison PNG plot path")
    parser.add_argument("--force-retrain", action="store_true", help="Force retrain existing checkpoints")
    args = parser.parse_args()

    train_candidates(checkpoint_dir=args.checkpoint_dir, force_retrain=args.force_retrain)
    run_full_evaluation_battery(
        checkpoint_dir=args.checkpoint_dir,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
    )


if __name__ == "__main__":
    main()
