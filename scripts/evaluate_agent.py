"""Evaluation and 3-Way Policy Benchmark Script for Prompt 4.

Benchmarks the trained MaskablePPO RL agent against:
  1. Dynamic Dijkstra (Live Latency, Centralized Global Snapshot)
  2. Static SPF (Physical Fiber Distance, Baseline)

Methodological Rigor Features:
  - Held-out seed evaluation (seed 100 unseen during training)
  - Side-by-side in-distribution (seed 42) vs held-out (seed 100) comparison
  - Explicit confirmation and documentation of deterministic argmax evaluation
  - Survivorship bias investigation: dropped vs. delivered packet topological profile
  - Lead with full comparative metrics table before narrative interpretation

Exports:
  - Tabular 3-way comparisons
  - 4-panel benchmark plot: assets/rl_benchmark_comparison.png
  - Per-packet telemetry CSV: assets/rl_packet_telemetry.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sb3_contrib import MaskablePPO

# Ensure RORL root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.network_sim import NSFNETSimEnv
from src.dijkstra import DijkstraRouter
from src.packet_traffic import PacketTrafficGenerator, RoutingSimulationRunner
from src.metrics import (
    compute_aggregate_metrics,
    compare_three_policies,
    compute_dropped_packet_diagnostics,
)
from src.visualize import plot_three_policy_benchmark


def evaluate_single_run(
    rl_model: MaskablePPO,
    seed: int,
    total_timesteps: int = 40,
    packets_per_step: int = 50,
    spike_timestep: int = 20,
    spike_edge: tuple = (1, 7),
    spike_magnitude: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, NSFNETSimEnv]:
    """Runs a single 3-way evaluation on the specified seed."""
    spikes = {spike_timestep: (spike_edge[0], spike_edge[1], spike_magnitude)}

    # Policy 1: Dynamic Dijkstra
    env_dijkstra = NSFNETSimEnv(seed=seed)
    gen_dijkstra = PacketTrafficGenerator(seed=seed, packets_per_step=packets_per_step)
    router = DijkstraRouter(default_weight_attr="latency")
    runner_dijkstra = RoutingSimulationRunner(
        env=env_dijkstra,
        packet_gen=gen_dijkstra,
        router=router,
        enable_packet_feedback=True,
        seed=seed,
    )
    dijk_pkts_df, dijk_steps_df = runner_dijkstra.run_simulation(
        num_timesteps=total_timesteps,
        routing_mode="dynamic_dijkstra",
        spike_schedule=spikes,
    )

    # Policy 2: Static SPF
    env_static = NSFNETSimEnv(seed=seed)
    gen_static = PacketTrafficGenerator(seed=seed, packets_per_step=packets_per_step)
    runner_static = RoutingSimulationRunner(
        env=env_static,
        packet_gen=gen_static,
        router=router,
        enable_packet_feedback=True,
        seed=seed,
    )
    stat_pkts_df, stat_steps_df = runner_static.run_simulation(
        num_timesteps=total_timesteps,
        routing_mode="static_spf",
        spike_schedule=spikes,
    )

    # Policy 3: MaskablePPO RL Agent
    env_rl = NSFNETSimEnv(seed=seed)
    gen_rl = PacketTrafficGenerator(seed=seed, packets_per_step=packets_per_step)
    runner_rl = RoutingSimulationRunner(
        env=env_rl,
        packet_gen=gen_rl,
        router=router,
        rl_model=rl_model,
        enable_packet_feedback=True,
        seed=seed,
    )
    rl_pkts_df, rl_steps_df = runner_rl.run_simulation(
        num_timesteps=total_timesteps,
        routing_mode="rl_agent",
        spike_schedule=spikes,
    )

    return dijk_pkts_df, dijk_steps_df, stat_pkts_df, stat_steps_df, rl_pkts_df, rl_steps_df, env_rl


def print_survivorship_bias_report(
    rl_pkts_df: pd.DataFrame,
    stat_pkts_df: pd.DataFrame,
    dijk_pkts_df: pd.DataFrame,
    graph: Any,
):
    """Prints diagnostic breakdown of dropped vs delivered packets across policies."""
    rl_diag = compute_dropped_packet_diagnostics(rl_pkts_df, graph)
    stat_diag = compute_dropped_packet_diagnostics(stat_pkts_df, graph)
    dijk_diag = compute_dropped_packet_diagnostics(dijk_pkts_df, graph)

    print("\n" + "=" * 90)
    print("SURVIVORSHIP BIAS INVESTIGATION: DROPPED VS. DELIVERED PACKET PROFILE")
    print("=" * 90)
    print(f"{'Metric / Characteristic':<42} | {'Dynamic Dijkstra':<15} | {'Static SPF':<15} | {'MaskablePPO RL':<15}")
    print("-" * 90)

    rows = [
        ("Delivered Packet Count", dijk_diag['delivered']['count'], stat_diag['delivered']['count'], rl_diag['delivered']['count']),
        ("Dropped Packet Count", dijk_diag['dropped']['count'], stat_diag['dropped']['count'], rl_diag['dropped']['count']),
        ("Packet Loss Rate (%)", f"{dijk_diag['loss_rate_pct']:.2f}%", f"{stat_diag['loss_rate_pct']:.2f}%", f"{rl_diag['loss_rate_pct']:.2f}%"),
        ("Delivered: Mean (s,d) Topo Distance (km)", f"{dijk_diag['delivered']['mean_topological_distance_km']:.1f}", f"{stat_diag['delivered']['mean_topological_distance_km']:.1f}", f"{rl_diag['delivered']['mean_topological_distance_km']:.1f}"),
        ("Dropped:   Mean (s,d) Topo Distance (km)", f"{dijk_diag['dropped']['mean_topological_distance_km']:.1f}", f"{stat_diag['dropped']['mean_topological_distance_km']:.1f}", f"{rl_diag['dropped']['mean_topological_distance_km']:.1f}"),
        ("Delivered: Mean Min Hops", f"{dijk_diag['delivered']['mean_min_hops']:.2f}", f"{stat_diag['delivered']['mean_min_hops']:.2f}", f"{rl_diag['delivered']['mean_min_hops']:.2f}"),
        ("Dropped:   Mean Min Hops", f"{dijk_diag['dropped']['mean_min_hops']:.2f}", f"{stat_diag['dropped']['mean_min_hops']:.2f}", f"{rl_diag['dropped']['mean_min_hops']:.2f}"),
        ("Delivered: Mean Queuing Delay (ms)", f"{dijk_diag['delivered']['mean_queuing_delay_ms']:.2f}", f"{stat_diag['delivered']['mean_queuing_delay_ms']:.2f}", f"{rl_diag['delivered']['mean_queuing_delay_ms']:.2f}"),
        ("Dropped:   Traversed Queuing Delay (ms)", f"{dijk_diag['dropped']['mean_queuing_delay_ms']:.2f}", f"{stat_diag['dropped']['mean_queuing_delay_ms']:.2f}", f"{rl_diag['dropped']['mean_queuing_delay_ms']:.2f}"),
        ("Delivered: Path Traverses Shock Link (%)", f"{dijk_diag['delivered']['crosses_shock_link_pct']:.1f}%", f"{stat_diag['delivered']['crosses_shock_link_pct']:.1f}%", f"{rl_diag['delivered']['crosses_shock_link_pct']:.1f}%"),
        ("Dropped:   Path Traverses Shock Link (%)", f"{dijk_diag['dropped']['crosses_shock_link_pct']:.1f}%", f"{stat_diag['dropped']['crosses_shock_link_pct']:.1f}%", f"{rl_diag['dropped']['crosses_shock_link_pct']:.1f}%"),
    ]

    for label, d_val, s_val, r_val in rows:
        print(f"{label:<42} | {str(d_val):<15} | {str(s_val):<15} | {str(r_val):<15}")

    print("-" * 90)


def evaluate_policies(
    model_path: str = "checkpoints/final_model.zip",
    seed: int = 100,  # Default to held-out seed
    total_timesteps: int = 40,
    packets_per_step: int = 50,
    spike_timestep: int = 20,
    spike_edge: tuple = (1, 7),
    spike_magnitude: float = 5.0,
    eval_both_seeds: bool = False,
    output_plot: str = "assets/rl_benchmark_comparison.png",
    output_csv: str = "assets/rl_packet_telemetry.csv",
):
    print("=" * 90)
    print("3-Way Policy Benchmark: Dynamic Dijkstra vs. Static SPF vs. MaskablePPO RL")
    print("=" * 90)

    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"\n[1] Configuration & Model Loading:")
    print(f"    - Model Checkpoint: {model_path}")
    rl_model = MaskablePPO.load(model_path)
    print("    -> MaskablePPO agent loaded successfully.")
    print("    - Evaluation Inference Mode: DETERMINISTIC GREEDY ARGMAX (model.predict(obs, deterministic=True))")
    print("    - Note on Training vs. Evaluation Gap:")
    print("      During training rollouts, actions are sampled stochastically with an entropy bonus (ent_coef=0.01) to explore alternative routes,")
    print("      yielding ~65%-79% rolling delivery rate. Evaluation disables exploration and selects the deterministic argmax, yielding higher delivery.")

    if eval_both_seeds:
        seeds_to_run = [
            (42, "In-Distribution (Overlaps Training Seed 42)"),
            (100, "Held-Out Unseen Seed (Seed 100)"),
        ]
    else:
        seed_label = "Held-Out Unseen Seed (Seed 100)" if seed != 42 else "In-Distribution (Overlaps Training Seed 42)"
        seeds_to_run = [(seed, seed_label)]

    runs_data = {}

    for s_val, s_desc in seeds_to_run:
        print(f"\n" + "#" * 90)
        print(f"# Executing Benchmark Run on Seed {s_val}: {s_desc}")
        print(f"# Horizon: {total_timesteps} steps, {packets_per_step} pkts/step (2,000 total packets), Traffic Shock at t={spike_timestep} (+5 Gbps on {spike_edge})")
        print("#" * 90)

        d_pkts, d_steps, s_pkts, s_steps, r_pkts, r_steps, env_rl = evaluate_single_run(
            rl_model=rl_model,
            seed=s_val,
            total_timesteps=total_timesteps,
            packets_per_step=packets_per_step,
            spike_timestep=spike_timestep,
            spike_edge=spike_edge,
            spike_magnitude=spike_magnitude,
        )

        runs_data[s_val] = {
            "desc": s_desc,
            "d_pkts": d_pkts, "d_steps": d_steps,
            "s_pkts": s_pkts, "s_steps": s_steps,
            "r_pkts": r_pkts, "r_steps": r_steps,
            "graph": env_rl.graph,
        }

        print(f"\n[Comparative Performance Report — Seed {s_val}]")
        comp_df = compare_three_policies(
            dijkstra_df=d_pkts,
            static_df=s_pkts,
            rl_df=r_pkts,
            total_timesteps=total_timesteps,
        )
        print(comp_df.to_string(index=False))

        # Survivorship bias investigation
        print_survivorship_bias_report(r_pkts, s_pkts, d_pkts, env_rl.graph)

    # If both seeds were run, display side-by-side generalization table
    if eval_both_seeds and len(runs_data) >= 2:
        print("\n" + "=" * 90)
        print("MEMORIZATION VS. GENERALIZATION AUDIT: SEED 42 (TRAIN-OVERLAPPING) VS. SEED 100 (HELD-OUT)")
        print("=" * 90)

        m_dijk_42 = compute_aggregate_metrics(runs_data[42]["d_pkts"], total_timesteps)
        m_stat_42 = compute_aggregate_metrics(runs_data[42]["s_pkts"], total_timesteps)
        m_rl_42 = compute_aggregate_metrics(runs_data[42]["r_pkts"], total_timesteps)

        m_dijk_100 = compute_aggregate_metrics(runs_data[100]["d_pkts"], total_timesteps)
        m_stat_100 = compute_aggregate_metrics(runs_data[100]["s_pkts"], total_timesteps)
        m_rl_100 = compute_aggregate_metrics(runs_data[100]["r_pkts"], total_timesteps)

        audit_keys = [
            ("Packets Delivered", "total_packets_delivered", "{:d}"),
            ("Packet Loss Rate (%)", "packet_loss_rate", "{:.3%}"),
            ("Mean Latency (ms)", "mean_latency_ms", "{:.2f}"),
            ("Median Latency (ms)", "median_latency_ms", "{:.2f}"),
            ("P95 Latency (ms)", "p95_latency_ms", "{:.2f}"),
            ("Max Latency (ms)", "max_latency_ms", "{:.2f}"),
            ("Mean Queuing Delay (ms)", "mean_queuing_delay_ms", "{:.2f}"),
            ("Mean Hops", "mean_hop_count", "{:.2f}"),
            ("Mean Throughput Aggregate (Gbps)", "mean_throughput_aggregate_gbps", "{:.3f}"),
        ]

        audit_rows = []
        for label, k, fmt in audit_keys:
            val_rl_42 = m_rl_42.get(k, 0.0)
            val_rl_100 = m_rl_100.get(k, 0.0)
            gap = ((val_rl_100 - val_rl_42) / val_rl_42 * 100.0) if val_rl_42 != 0 else 0.0

            audit_rows.append({
                "Metric": label,
                "RL (Seed 42, In-Distribution)": fmt.format(val_rl_42) if "{:" in fmt else val_rl_42,
                "RL (Seed 100, Held-Out)": fmt.format(val_rl_100) if "{:" in fmt else val_rl_100,
                "RL Gap (%)": f"{gap:+.2f}%",
                "Dijkstra (Held-Out)": fmt.format(m_dijk_100.get(k, 0.0)) if "{:" in fmt else m_dijk_100.get(k, 0.0),
                "Static SPF (Held-Out)": fmt.format(m_stat_100.get(k, 0.0)) if "{:" in fmt else m_stat_100.get(k, 0.0),
            })

        audit_df = pd.DataFrame(audit_rows)
        print(audit_df.to_string(index=False))

    # Export visualization & CSV for primary seed
    primary_seed = 100 if 100 in runs_data else seed
    p_data = runs_data[primary_seed]
    plot_full_path = os.path.join(PROJECT_ROOT, output_plot) if not os.path.isabs(output_plot) else output_plot
    csv_full_path = os.path.join(PROJECT_ROOT, output_csv) if not os.path.isabs(output_csv) else output_csv

    plot_three_policy_benchmark(
        dijkstra_pkt_df=p_data["d_pkts"],
        static_pkt_df=p_data["s_pkts"],
        rl_pkt_df=p_data["r_pkts"],
        dijkstra_step_df=p_data["d_steps"],
        static_step_df=p_data["s_steps"],
        rl_step_df=p_data["r_steps"],
        save_path=plot_full_path,
        spike_timestep=spike_timestep,
    )
    print(f"\n[*] Saved benchmark plot for Seed {primary_seed} to: {plot_full_path}")

    p_data["r_pkts"].to_csv(csv_full_path, index=False)
    print(f"[*] Exported per-packet telemetry for Seed {primary_seed} to: {csv_full_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MaskablePPO RL Agent vs Dijkstra and Static SPF")
    parser.add_argument("--model", type=str, default="checkpoints/final_model.zip", help="Path to trained model zip")
    parser.add_argument("--seed", type=int, default=100, help="Evaluation seed (default: 100 for held-out)")
    parser.add_argument("--timesteps", type=int, default=40, help="Number of simulation timesteps")
    parser.add_argument("--packets-per-step", type=int, default=50, help="Packets per timestep")
    parser.add_argument("--eval-both-seeds", action="store_true", help="Run both Seed 42 and Seed 100 side-by-side")
    parser.add_argument("--output-plot", type=str, default="assets/rl_benchmark_comparison.png", help="Path for comparison plot")
    parser.add_argument("--output-csv", type=str, default="assets/rl_packet_telemetry.csv", help="Path for per-packet telemetry CSV")

    args = parser.parse_args()
    evaluate_policies(
        model_path=args.model,
        seed=args.seed,
        total_timesteps=args.timesteps,
        packets_per_step=args.packets_per_step,
        eval_both_seeds=args.eval_both_seeds,
        output_plot=args.output_plot,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
