"""Visualization utilities for NSFNET topology and dynamic link-state telemetry."""

from typing import Optional, List, Tuple
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

from .topology import NSFNET_NODES


def plot_nsfnet_topology(
    graph: nx.Graph,
    save_path: str = "assets/nsfnet_topology.png",
    title: str = "NSFNET Benchmark Topology (14 Nodes, 21 Links)",
) -> str:
    """Plots the NSFNET topology with geographical node placement and link annotations.

    Args:
        graph: NetworkX graph.
        save_path: Filepath where the plot image will be saved.
        title: Title of the figure.

    Returns:
        Absolute filepath to the saved image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 8), dpi=200)
    fig.patch.set_facecolor("#0f172a")  # Dark slate background
    ax.set_facecolor("#0f172a")

    # Positions from longitude/latitude
    raw_pos = {n: graph.nodes[n]["pos"] for n in graph.nodes()}
    # Project slightly for better US map proportions
    pos = {n: (p[0], p[1] * 1.25) for n, p in raw_pos.items()}

    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="#38bdf8",  # Sky blue
        width=2.2,
        alpha=0.85,
    )

    # Edge labels: distance (km) & base delay (ms)
    edge_labels = {
        (u, v): f"{int(graph[u][v]['distance_km'])}km\n{graph[u][v]['base_latency_ms']:.1f}ms"
        for u, v in graph.edges()
    }
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=7,
        font_color="#cbd5e1",
        bbox=dict(boxstyle="round,pad=0.2", fc="#1e293b", ec="#475569", alpha=0.9),
    )

    # Draw nodes
    node_colors = ["#f43f5e" if graph.degree(n) == 4 else "#3b82f6" for n in graph.nodes()]
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=850,
        edgecolors="#ffffff",
        linewidths=1.5,
    )

    # Node labels: ID and City Name
    node_labels = {
        n: f"{n}\n{NSFNET_NODES[n]['name']}"
        for n in graph.nodes()
    }
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=node_labels,
        ax=ax,
        font_size=8.5,
        font_weight="bold",
        font_color="#ffffff",
    )

    ax.set_title(
        f"{title}\n14 Nodes, 21 Links (Red: Degree 4, Blue: Degree 2-3)",
        fontsize=14,
        fontweight="bold",
        color="#f8fafc",
        pad=15,
    )

    # Add subtitle annotations
    ax.text(
        0.5, -0.05,
        "Fiber propagation speed: 200 km/ms (~5 µs/km) | Default capacity: 10.0 Gbps per link",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#94a3b8",
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


def plot_dynamic_telemetry(
    df: pd.DataFrame,
    save_path: str = "assets/link_state_telemetry.png",
    selected_edges: Optional[List[str]] = None,
) -> str:
    """Plots 4-panel time-series telemetry verifying that link stats change dynamically over time.

    Subplots:
      1. Congestion [0, 1]
      2. Available Bandwidth [0, 1]
      3. Total Latency (ms)
      4. Packet Loss Rate [0, 1]
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if selected_edges is None:
        # Pick 4 diverse representative links:
        # short link (10, 13) 350km, medium (0, 1) 1300km, long (1, 7) 3000km, central (3, 8) 2400km
        available_edges = df["edge"].unique().tolist()
        candidates = ["(10,13)", "(0,1)", "(1,7)", "(3,8)"]
        selected_edges = [e for e in candidates if e in available_edges]
        if not selected_edges:
            selected_edges = available_edges[:4]

    colors = ["#38bdf8", "#f43f5e", "#10b981", "#f59e0b"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, dpi=200)
    fig.patch.set_facecolor("#0f172a")

    metric_configs = [
        ("congestion", "Congestion Ratio (c)", (0.0, 1.05), "Utilization Ratio [0 - 1]"),
        ("available_bandwidth", "Available Bandwidth Ratio (B_avail)", (0.0, 1.05), "Normalized [0 - 1]"),
        ("latency_ms", "Total Latency (Propagation + Queuing)", None, "Milliseconds (ms)"),
        ("packet_loss_rate", "Packet Loss Rate (RED Buffer Overflow)", (0.0, 0.25), "Loss Probability [0 - 1]"),
    ]

    for idx, (metric, label, ylim, ylabel) in enumerate(metric_configs):
        ax = axes[idx]
        ax.set_facecolor("#1e293b")

        for edge_idx, edge in enumerate(selected_edges):
            sub = df[df["edge"] == edge]
            ax.plot(
                sub["timestep"],
                sub[metric],
                label=f"Link {edge}",
                color=colors[edge_idx % len(colors)],
                linewidth=2.0,
                marker="o",
                markersize=3,
                alpha=0.9,
            )

        if metric == "congestion":
            # Add threshold line for RED packet loss
            ax.axhline(0.70, color="#fbbf24", linestyle="--", linewidth=1.2, alpha=0.8, label="Loss Threshold (c=0.70)")

        if ylim:
            ax.set_ylim(ylim)

        ax.set_ylabel(ylabel, color="#e2e8f0", fontsize=10, fontweight="bold")
        ax.set_title(label, color="#f8fafc", fontsize=11, fontweight="bold", loc="left", pad=8)
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")

        if idx == 0:
            ax.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)

    axes[-1].set_xlabel("Discrete Timestep (t)", color="#f8fafc", fontsize=11, fontweight="bold")
    fig.suptitle(
        "NSFNET Dynamic Link-State Telemetry Verification Across Discrete Timesteps",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


def plot_routing_benchmark(
    dijkstra_pkt_df: pd.DataFrame,
    static_pkt_df: pd.DataFrame,
    dijkstra_step_df: pd.DataFrame,
    static_step_df: pd.DataFrame,
    save_path: str = "assets/dijkstra_routing_benchmark.png",
) -> str:
    """Generates a publication-quality 3-panel benchmark comparison of Dynamic Dijkstra vs Static SPF.

    Panel 1: End-to-End Latency Distribution (KDE / Histogram)
    Panel 2: Packet Loss Rate Time-Series Across Simulation Steps
    Panel 3: Effective Throughput (Gbps) Time-Series

    Args:
        dijkstra_pkt_df: Per-packet dataframe from Dynamic Dijkstra.
        static_pkt_df: Per-packet dataframe from Static SPF.
        dijkstra_step_df: Per-step aggregate dataframe from Dynamic Dijkstra.
        static_step_df: Per-step aggregate dataframe from Static SPF.
        save_path: Destination path for saved PNG image.

    Returns:
        Absolute filepath to saved plot.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), dpi=200)
    fig.patch.set_facecolor("#0f172a")  # Dark slate background

    color_dijk = "#38bdf8"  # Electric cyan / sky blue
    color_stat = "#f43f5e"  # Rose / red

    # -------------------------------------------------------------
    # Panel 1: Latency Distribution Comparison
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor("#1e293b")

    d_deliv = dijkstra_pkt_df[dijkstra_pkt_df["delivered"]]
    s_deliv = static_pkt_df[static_pkt_df["delivered"]]

    d_lat = d_deliv["end_to_end_latency_ms"] if not d_deliv.empty else dijkstra_pkt_df["end_to_end_latency_ms"]
    s_lat = s_deliv["end_to_end_latency_ms"] if not s_deliv.empty else static_pkt_df["end_to_end_latency_ms"]

    bins = np.linspace(min(d_lat.min(), s_lat.min()), max(d_lat.max(), s_lat.max()), 40)
    ax1.hist(
        d_lat,
        bins=bins,
        alpha=0.65,
        color=color_dijk,
        label=f"Dynamic Dijkstra (Mean: {d_lat.mean():.1f} ms | P95: {d_lat.quantile(0.95):.1f} ms)",
        edgecolor="#0284c7",
        density=True,
    )
    ax1.hist(
        s_lat,
        bins=bins,
        alpha=0.55,
        color=color_stat,
        label=f"Static SPF (Mean: {s_lat.mean():.1f} ms | P95: {s_lat.quantile(0.95):.1f} ms)",
        edgecolor="#be123c",
        density=True,
    )

    ax1.axvline(d_lat.mean(), color=color_dijk, linestyle="--", linewidth=1.5, alpha=0.9)
    ax1.axvline(s_lat.mean(), color=color_stat, linestyle="--", linewidth=1.5, alpha=0.9)

    ax1.set_title("1. End-to-End Latency Distribution for Delivered Packets", color="#f8fafc", fontsize=12, fontweight="bold", loc="left", pad=8)
    ax1.set_xlabel("End-to-End Path Latency (ms)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Probability Density", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.tick_params(colors="#cbd5e1")
    ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # -------------------------------------------------------------
    # Panel 2: Packet Loss Rate Time-Series
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor("#1e293b")

    steps_d = dijkstra_step_df["timestep"]
    loss_d = dijkstra_step_df["packet_loss_rate"]
    steps_s = static_step_df["timestep"]
    loss_s = static_step_df["packet_loss_rate"]

    ax2.plot(steps_d, loss_d * 100.0, color=color_dijk, marker="o", markersize=4, linewidth=2.0, label=f"Dynamic Dijkstra (Mean Loss: {loss_d.mean()*100.0:.2f}%)")
    ax2.plot(steps_s, loss_s * 100.0, color=color_stat, marker="s", markersize=4, linewidth=2.0, linestyle="--", label=f"Static SPF (Mean Loss: {loss_s.mean()*100.0:.2f}%)")

    ax2.set_title("2. Packet Loss Rate (%) Across Discrete Timesteps", color="#f8fafc", fontsize=12, fontweight="bold", loc="left", pad=8)
    ax2.set_xlabel("Discrete Timestep (t)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Packet Loss Rate (%)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.tick_params(colors="#cbd5e1")
    ax2.legend(loc="upper left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # -------------------------------------------------------------
    # Panel 3: Throughput per Timestep Time-Series
    # -------------------------------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor("#1e293b")

    thru_d = dijkstra_step_df["throughput_gbps"]
    thru_s = static_step_df["throughput_gbps"]

    ax3.plot(steps_d, thru_d, color=color_dijk, marker="o", markersize=4, linewidth=2.0, label=f"Dynamic Dijkstra (Mean Aggregate: {thru_d.mean():.3f} Gbps)")
    ax3.plot(steps_s, thru_s, color=color_stat, marker="^", markersize=4, linewidth=2.0, linestyle="--", label=f"Static SPF (Mean Aggregate: {thru_s.mean():.3f} Gbps)")

    ax3.set_title("3. Throughput per Timestep (Gbps) Across Timesteps", color="#f8fafc", fontsize=12, fontweight="bold", loc="left", pad=8)
    ax3.set_xlabel("Discrete Timestep (t)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Throughput per Timestep (Gbps)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax3.tick_params(colors="#cbd5e1")
    ax3.legend(loc="lower left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    for spine in ax3.spines.values():
        spine.set_color("#334155")

    fig.suptitle(
        "NSFNET Routing Benchmark: Dynamic Dijkstra (Live Latency) vs. Static SPF (Distance)",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


def plot_three_policy_benchmark(
    dijkstra_pkt_df: pd.DataFrame,
    static_pkt_df: pd.DataFrame,
    rl_pkt_df: pd.DataFrame,
    dijkstra_step_df: pd.DataFrame,
    static_step_df: pd.DataFrame,
    rl_step_df: pd.DataFrame,
    save_path: str = "assets/rl_benchmark_comparison.png",
    spike_timestep: Optional[int] = 20,
) -> str:
    """Generates a publication-quality 4-panel benchmark comparison of:
      1. Dynamic Dijkstra (Live Latency, Centralized Global Snapshot)
      2. Static SPF (Fiber Distance, Baseline)
      3. MaskablePPO RL (Learned Decentralized Per-Hop Decisions)

    Panel 1: End-to-End Latency Distribution for Delivered Packets
    Panel 2: Packet Loss Rate (%) Across Discrete Timesteps
    Panel 3: Throughput per Timestep (Gbps) Across Timesteps
    Panel 4: Mean Hop Count per Timestep (Path Efficiency)
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(13, 16), dpi=200)
    fig.patch.set_facecolor("#0f172a")  # Dark slate background

    color_dijk = "#38bdf8"   # Electric Cyan
    color_stat = "#f43f5e"   # Rose Red
    color_rl = "#10b981"     # Emerald Green

    # -------------------------------------------------------------
    # Panel 1: Latency Distribution Comparison
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor("#1e293b")

    d_deliv = dijkstra_pkt_df[dijkstra_pkt_df["delivered"]]
    s_deliv = static_pkt_df[static_pkt_df["delivered"]]
    r_deliv = rl_pkt_df[rl_pkt_df["delivered"]]

    d_lat = d_deliv["end_to_end_latency_ms"] if not d_deliv.empty else dijkstra_pkt_df["end_to_end_latency_ms"]
    s_lat = s_deliv["end_to_end_latency_ms"] if not s_deliv.empty else static_pkt_df["end_to_end_latency_ms"]
    r_lat = r_deliv["end_to_end_latency_ms"] if not r_deliv.empty else rl_pkt_df["end_to_end_latency_ms"]

    min_val = min(d_lat.min(), s_lat.min(), r_lat.min())
    max_val = max(d_lat.max(), s_lat.max(), r_lat.max())
    bins = np.linspace(min_val, max_val, 45)

    ax1.hist(
        d_lat,
        bins=bins,
        alpha=0.45,
        color=color_dijk,
        label=f"Dynamic Dijkstra (Mean: {d_lat.mean():.1f} ms | P95: {d_lat.quantile(0.95):.1f} ms)",
        edgecolor="#0284c7",
        density=True,
    )
    ax1.hist(
        s_lat,
        bins=bins,
        alpha=0.40,
        color=color_stat,
        label=f"Static SPF (Mean: {s_lat.mean():.1f} ms | P95: {s_lat.quantile(0.95):.1f} ms)",
        edgecolor="#be123c",
        density=True,
    )
    ax1.hist(
        r_lat,
        bins=bins,
        alpha=0.55,
        color=color_rl,
        label=f"MaskablePPO RL (Mean: {r_lat.mean():.1f} ms | P95: {r_lat.quantile(0.95):.1f} ms)",
        edgecolor="#059669",
        density=True,
    )

    ax1.axvline(d_lat.mean(), color=color_dijk, linestyle="--", linewidth=1.5, alpha=0.9)
    ax1.axvline(s_lat.mean(), color=color_stat, linestyle="--", linewidth=1.5, alpha=0.9)
    ax1.axvline(r_lat.mean(), color=color_rl, linestyle="--", linewidth=1.8, alpha=0.95)

    ax1.set_title("1. End-to-End Latency Distribution for Delivered Packets", color="#f8fafc", fontsize=12, fontweight="bold", loc="left", pad=8)
    ax1.set_xlabel("End-to-End Latency (ms)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Probability Density", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.tick_params(colors="#cbd5e1")
    ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # -------------------------------------------------------------
    # Panel 2: Packet Loss Rate Time-Series
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor("#1e293b")

    steps_d = dijkstra_step_df["timestep"]
    loss_d = dijkstra_step_df["packet_loss_rate"]
    steps_s = static_step_df["timestep"]
    loss_s = static_step_df["packet_loss_rate"]
    steps_r = rl_step_df["timestep"]
    loss_r = rl_step_df["packet_loss_rate"]

    ax2.plot(steps_d, loss_d * 100.0, color=color_dijk, marker="o", markersize=3.5, linewidth=2.0, label=f"Dynamic Dijkstra (Mean: {loss_d.mean()*100.0:.2f}%)")
    ax2.plot(steps_s, loss_s * 100.0, color=color_stat, marker="s", markersize=3.5, linewidth=2.0, linestyle="--", label=f"Static SPF (Mean: {loss_s.mean()*100.0:.2f}%)")
    ax2.plot(steps_r, loss_r * 100.0, color=color_rl, marker="^", markersize=4.0, linewidth=2.2, label=f"MaskablePPO RL (Mean: {loss_r.mean()*100.0:.2f}%)")

    if spike_timestep is not None:
        ax2.axvline(spike_timestep, color="#fbbf24", linestyle=":", linewidth=1.5, alpha=0.85, label=f"Traffic Shock (t={spike_timestep})")

    ax2.set_title("2. Packet Loss Rate (%) Across Discrete Timesteps", color="#f8fafc", fontsize=12, fontweight="bold", loc="left", pad=8)
    ax2.set_xlabel("Discrete Timestep (t)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Packet Loss Rate (%)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.tick_params(colors="#cbd5e1")
    ax2.legend(loc="upper left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # -------------------------------------------------------------
    # Panel 3: Throughput per Timestep Time-Series
    # -------------------------------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor("#1e293b")

    thru_d = dijkstra_step_df["throughput_gbps"]
    thru_s = static_step_df["throughput_gbps"]
    thru_r = rl_step_df["throughput_gbps"]

    ax3.plot(steps_d, thru_d, color=color_dijk, marker="o", markersize=3.5, linewidth=2.0, label=f"Dynamic Dijkstra (Mean: {thru_d.mean():.3f} Gbps)")
    ax3.plot(steps_s, thru_s, color=color_stat, marker="s", markersize=3.5, linewidth=2.0, linestyle="--", label=f"Static SPF (Mean: {thru_s.mean():.3f} Gbps)")
    ax3.plot(steps_r, thru_r, color=color_rl, marker="^", markersize=4.0, linewidth=2.2, label=f"MaskablePPO RL (Mean: {thru_r.mean():.3f} Gbps)")

    if spike_timestep is not None:
        ax3.axvline(spike_timestep, color="#fbbf24", linestyle=":", linewidth=1.5, alpha=0.85)

    ax3.set_title("3. Throughput per Timestep (Gbps) Across Simulation Steps", color="#f8fafc", fontsize=12, fontweight="bold", loc="left", pad=8)
    ax3.set_xlabel("Discrete Timestep (t)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Throughput (Gbps)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax3.tick_params(colors="#cbd5e1")
    ax3.legend(loc="lower left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    for spine in ax3.spines.values():
        spine.set_color("#334155")

    # -------------------------------------------------------------
    # Panel 4: Mean Hop Count Across Timesteps
    # -------------------------------------------------------------
    ax4 = axes[3]
    ax4.set_facecolor("#1e293b")

    hops_d = dijkstra_step_df["mean_hops"] if "mean_hops" in dijkstra_step_df.columns else dijkstra_step_df["mean_hop_count"]
    hops_s = static_step_df["mean_hops"] if "mean_hops" in static_step_df.columns else static_step_df["mean_hop_count"]
    hops_r = rl_step_df["mean_hops"] if "mean_hops" in rl_step_df.columns else rl_step_df["mean_hop_count"]

    ax4.plot(steps_d, hops_d, color=color_dijk, marker="o", markersize=3.5, linewidth=2.0, label=f"Dynamic Dijkstra (Mean: {hops_d.mean():.2f} hops)")
    ax4.plot(steps_s, hops_s, color=color_stat, marker="s", markersize=3.5, linewidth=2.0, linestyle="--", label=f"Static SPF (Mean: {hops_s.mean():.2f} hops)")
    ax4.plot(steps_r, hops_r, color=color_rl, marker="^", markersize=4.0, linewidth=2.2, label=f"MaskablePPO RL (Mean: {hops_r.mean():.2f} hops)")

    ax4.set_title("4. Mean Hop Count per Timestep (Routing Efficiency)", color="#f8fafc", fontsize=12, fontweight="bold", loc="left", pad=8)
    ax4.set_xlabel("Discrete Timestep (t)", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Mean Hops", color="#cbd5e1", fontsize=10, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax4.tick_params(colors="#cbd5e1")
    ax4.legend(loc="upper left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    for spine in ax4.spines.values():
        spine.set_color("#334155")

    fig.suptitle(
        "NSFNET 3-Way Policy Benchmark: Dynamic Dijkstra vs. Static SPF vs. MaskablePPO RL\n"
        "[Note: Dijkstra uses global centralized snapshot; RL uses per-hop decentralized local observations]",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)

