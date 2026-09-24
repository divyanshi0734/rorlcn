"""Comprehensive Telemetry Metrics and Evaluation Utilities for Network Routing.

Computes both per-packet and aggregate performance indicators:
  1. End-to-End Latency (Mean, Median, P95, P99, Min, Max, Std)
  2. Packet Loss Rate (Empirical drop ratio & theoretical path drop probability)
  3. Throughput (Effective delivered Gbps, Mbps, and delivered packets/step)
  4. Path Efficiency (Mean hops, queuing delay overhead)
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


def compute_aggregate_metrics(
    packet_df: pd.DataFrame,
    total_timesteps: Optional[int] = None,
) -> Dict[str, Any]:
    """Computes comprehensive aggregate network performance metrics from a per-packet log.

    Args:
        packet_df: pd.DataFrame containing per-packet simulation logs.
        total_timesteps: Optional count of discrete timesteps simulated.

    Returns:
        Dict of standardized aggregate metrics.
    """
    if packet_df.empty:
        return {}

    total_sent = len(packet_df)
    delivered_mask = packet_df["delivered"].astype(bool)
    delivered_df = packet_df[delivered_mask]
    total_delivered = len(delivered_df)
    total_dropped = total_sent - total_delivered

    # 1. Packet Loss Rate
    empirical_loss_rate = total_dropped / total_sent if total_sent > 0 else 0.0
    theoretical_mean_loss_prob = float(packet_df["path_loss_prob"].mean()) if "path_loss_prob" in packet_df.columns else 0.0

    # 2. End-to-End Latency Statistics (evaluated on delivered packets)
    lat_target = delivered_df["end_to_end_latency_ms"] if not delivered_df.empty else packet_df["end_to_end_latency_ms"]
    
    mean_lat = float(lat_target.mean())
    median_lat = float(lat_target.median())
    p95_lat = float(lat_target.quantile(0.95))
    p99_lat = float(lat_target.quantile(0.99))
    min_lat = float(lat_target.min())
    max_lat = float(lat_target.max())
    std_lat = float(lat_target.std()) if len(lat_target) > 1 else 0.0

    # 3. Queuing Delay & Propagation Components
    if "queuing_delay_ms" in packet_df.columns:
        queuing_target = delivered_df["queuing_delay_ms"] if not delivered_df.empty else packet_df["queuing_delay_ms"]
        mean_queue = float(queuing_target.mean())
        max_queue = float(queuing_target.max())
    else:
        mean_queue = 0.0
        max_queue = 0.0

    # 4. Hop Statistics
    hop_target = delivered_df["hop_count"] if not delivered_df.empty else packet_df["hop_count"]
    mean_hops = float(hop_target.mean())
    min_hops = int(hop_target.min())
    max_hops = int(hop_target.max())

    # 5. Throughput Statistics
    # -------------------------------------------------------------------------
    # Note on Throughput Metrics Distinction:
    #   A) Per-Timestep Series: throughput_per_step(t) = sum_{k in Delivered(t)} D_pkt,k [Gbps]
    #      (Delivered traffic volume within individual timestep t; plotted across steps)
    #   B) Whole-Run Aggregate: mean_throughput_aggregate = sum_{all T} throughput_per_step(t) / T [Gbps]
    #      (Time-averaged delivery rate over the entire simulation horizon T)
    # -------------------------------------------------------------------------
    timesteps = total_timesteps if total_timesteps is not None else packet_df["timestep"].nunique()
    timesteps = max(1, timesteps)

    total_delivered_gb = float(delivered_df["demand_gbps"].sum())
    total_offered_gb = float(packet_df["demand_gbps"].sum())

    # Whole-run time-averaged aggregate throughput
    mean_throughput_aggregate_gbps = total_delivered_gb / timesteps
    mean_throughput_aggregate_mbps = mean_throughput_aggregate_gbps * 1000.0
    effective_throughput_gbps = mean_throughput_aggregate_gbps  # Backwards-compatible alias
    effective_throughput_mbps = mean_throughput_aggregate_mbps  # Backwards-compatible alias

    packet_throughput_pkts_per_step = total_delivered / timesteps
    delivery_ratio = total_delivered / total_sent if total_sent > 0 else 0.0

    return {
        "total_packets_sent": total_sent,
        "total_packets_delivered": total_delivered,
        "total_packets_dropped": total_dropped,
        "delivery_ratio": delivery_ratio,
        "packet_loss_rate": empirical_loss_rate,
        "theoretical_mean_loss_prob": theoretical_mean_loss_prob,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "min_latency_ms": min_lat,
        "max_latency_ms": max_lat,
        "std_latency_ms": std_lat,
        "mean_queuing_delay_ms": mean_queue,
        "max_queuing_delay_ms": max_queue,
        "mean_hop_count": mean_hops,
        "min_hops": min_hops,
        "max_hops": max_hops,
        "total_timesteps": timesteps,
        "mean_throughput_aggregate_gbps": mean_throughput_aggregate_gbps,
        "mean_throughput_aggregate_mbps": mean_throughput_aggregate_mbps,
        "effective_throughput_gbps": effective_throughput_gbps,
        "effective_throughput_mbps": effective_throughput_mbps,
        "packet_throughput_pkts_per_step": packet_throughput_pkts_per_step,
        "total_delivered_gb": total_delivered_gb,
        "total_offered_gb": total_offered_gb,
    }


def compute_throughput_per_step(packet_df: pd.DataFrame) -> pd.Series:
    """Computes the per-timestep throughput series: throughput_per_step(t) = sum_{k in Delivered(t)} D_pkt.

    Args:
        packet_df: pd.DataFrame with per-packet transmission logs.

    Returns:
        pd.Series indexed by timestep t containing delivered traffic demand in Gbps.
    """
    if packet_df.empty:
        return pd.Series(dtype=float)
    delivered_df = packet_df[packet_df["delivered"].astype(bool)]
    return delivered_df.groupby("timestep")["demand_gbps"].sum()


def compare_routing_policies(
    dijkstra_df: pd.DataFrame,
    static_df: pd.DataFrame,
    total_timesteps: Optional[int] = None,
) -> pd.DataFrame:
    """Generates comparative tabular report between Dynamic Dijkstra and Static SPF.

    Args:
        dijkstra_df: Per-packet dataframe from Dynamic Dijkstra simulation.
        static_df: Per-packet dataframe from Static SPF simulation.
        total_timesteps: Count of simulated timesteps.

    Returns:
        pd.DataFrame comparing metrics, delta, and percentage improvement.
    """
    m_dijk = compute_aggregate_metrics(dijkstra_df, total_timesteps)
    m_stat = compute_aggregate_metrics(static_df, total_timesteps)

    rows = []
    metric_keys = [
        ("Packets Sent", "total_packets_sent", "{:d}"),
        ("Packets Delivered", "total_packets_delivered", "{:d}"),
        ("Packets Dropped", "total_packets_dropped", "{:d}"),
        ("Packet Loss Rate (%)", "packet_loss_rate", "{:.3%}"),
        ("Mean Latency (ms)", "mean_latency_ms", "{:.2f}"),
        ("Median Latency (ms)", "median_latency_ms", "{:.2f}"),
        ("P95 Latency (ms)", "p95_latency_ms", "{:.2f}"),
        ("Max Latency (ms)", "max_latency_ms", "{:.2f}"),
        ("Mean Queuing Delay (ms)", "mean_queuing_delay_ms", "{:.2f}"),
        ("Mean Hops", "mean_hop_count", "{:.2f}"),
        ("Mean Throughput Aggregate (Gbps)", "mean_throughput_aggregate_gbps", "{:.3f}"),
        ("Mean Throughput Aggregate (Mbps)", "mean_throughput_aggregate_mbps", "{:.1f}"),
        ("Delivered Packets/Step", "packet_throughput_pkts_per_step", "{:.1f}"),
    ]

    for label, key, fmt in metric_keys:
        val_d = m_dijk.get(key, 0.0)
        val_s = m_stat.get(key, 0.0)

        # Improvement calculation (lower is better for latency/loss, higher is better for throughput)
        lower_is_better = "Loss" in label or "Latency" in label or "Delay" in label or "Dropped" in label
        
        if val_s != 0:
            if lower_is_better:
                rel_change = ((val_s - val_d) / val_s) * 100.0  # Positive means reduction (good)
            else:
                rel_change = ((val_d - val_s) / val_s) * 100.0  # Positive means increase (good)
            change_str = f"{rel_change:+.1f}%"
        else:
            change_str = "N/A"

        rows.append({
            "Metric": label,
            "Dynamic Dijkstra": fmt.format(val_d) if "{:" in fmt else val_d,
            "Static SPF": fmt.format(val_s) if "{:" in fmt else val_s,
            "Advantage": change_str,
        })

    return pd.DataFrame(rows)


def compare_three_policies(
    dijkstra_df: pd.DataFrame,
    static_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    total_timesteps: Optional[int] = None,
) -> pd.DataFrame:
    """Generates comparative tabular report comparing Dynamic Dijkstra, Static SPF, and MaskablePPO RL.

    Args:
        dijkstra_df: Per-packet dataframe from Dynamic Dijkstra simulation.
        static_df: Per-packet dataframe from Static SPF simulation.
        rl_df: Per-packet dataframe from MaskablePPO RL agent simulation.
        total_timesteps: Count of simulated timesteps.

    Returns:
        pd.DataFrame comparing metrics across the three policies.
    """
    m_dijk = compute_aggregate_metrics(dijkstra_df, total_timesteps)
    m_stat = compute_aggregate_metrics(static_df, total_timesteps)
    m_rl = compute_aggregate_metrics(rl_df, total_timesteps)

    rows = []
    metric_configs = [
        ("Packets Sent", "total_packets_sent", "{:d}", False),
        ("Packets Delivered", "total_packets_delivered", "{:d}", False),
        ("Packets Dropped", "total_packets_dropped", "{:d}", True),
        ("Packet Loss Rate (%)", "packet_loss_rate", "{:.3%}", True),
        ("Mean Latency (ms)", "mean_latency_ms", "{:.2f}", True),
        ("Median Latency (ms)", "median_latency_ms", "{:.2f}", True),
        ("P95 Latency (ms)", "p95_latency_ms", "{:.2f}", True),
        ("Max Latency (ms)", "max_latency_ms", "{:.2f}", True),
        ("Mean Queuing Delay (ms)", "mean_queuing_delay_ms", "{:.2f}", True),
        ("Mean Hops", "mean_hop_count", "{:.2f}", True),
        ("Mean Throughput Aggregate (Gbps)", "mean_throughput_aggregate_gbps", "{:.3f}", False),
        ("Delivered Packets/Step", "packet_throughput_pkts_per_step", "{:.1f}", False),
    ]

    for label, key, fmt, lower_is_better in metric_configs:
        val_d = m_dijk.get(key, 0.0)
        val_s = m_stat.get(key, 0.0)
        val_r = m_rl.get(key, 0.0)

        # Advantage of RL vs Static
        if val_s != 0:
            if lower_is_better:
                rel_s = ((val_s - val_r) / val_s) * 100.0
            else:
                rel_s = ((val_r - val_s) / val_s) * 100.0
            adv_static = f"{rel_s:+.1f}%"
        else:
            adv_static = "N/A"

        # Advantage of RL vs Dijkstra
        if val_d != 0:
            if lower_is_better:
                rel_d = ((val_d - val_r) / val_d) * 100.0
            else:
                rel_d = ((val_r - val_d) / val_d) * 100.0
            adv_dijk = f"{rel_d:+.1f}%"
        else:
            adv_dijk = "N/A"

        rows.append({
            "Metric": label,
            "Dynamic Dijkstra": fmt.format(val_d) if "{:" in fmt else str(val_d),
            "Static SPF": fmt.format(val_s) if "{:" in fmt else str(val_s),
            "MaskablePPO RL": fmt.format(val_r) if "{:" in fmt else str(val_r),
            "RL vs Static": adv_static,
            "RL vs Dijkstra": adv_dijk,
        })

    return pd.DataFrame(rows)


def compute_dropped_packet_diagnostics(
    packet_df: pd.DataFrame,
    graph: Optional[Any] = None,
    shock_edge: Tuple[int, int] = (1, 7),
) -> Dict[str, Any]:
    """Analyzes characteristics of dropped vs delivered packets to investigate survivorship bias.

    Examines:
      - Topological distance and shortest-path hops between (src, dst)
      - Fraction of packets whose theoretical shortest path traverses the shock edge
      - Bottleneck link congestion encountered along the path
      - Hops traversed before drop
    """
    if packet_df.empty:
        return {}

    delivered_df = packet_df[packet_df["delivered"].astype(bool)]
    dropped_df = packet_df[~packet_df["delivered"].astype(bool)]

    # Compute static topological shortest paths if graph is provided
    sp_cache = {}
    if graph is not None:
        import networkx as nx
        for s in graph.nodes():
            for d in graph.nodes():
                if s != d:
                    try:
                        p_dist = nx.shortest_path(graph, source=s, target=d, weight="distance_km")
                        dist = sum(graph[p_dist[i]][p_dist[i+1]]["distance_km"] for i in range(len(p_dist)-1))
                        p_hops = nx.shortest_path(graph, source=s, target=d)
                        min_hops = len(p_hops) - 1
                        u_shock, v_shock = shock_edge
                        crosses_shock = any(
                            (p_dist[i] == u_shock and p_dist[i+1] == v_shock) or
                            (p_dist[i] == v_shock and p_dist[i+1] == u_shock)
                            for i in range(len(p_dist)-1)
                        )
                        sp_cache[(s, d)] = (dist, min_hops, crosses_shock)
                    except Exception:
                        sp_cache[(s, d)] = (0.0, 0, False)

    def extract_stats(sub_df: pd.DataFrame) -> Dict[str, Any]:
        if sub_df.empty:
            return {
                "count": 0,
                "mean_hops_traversed": 0.0,
                "mean_topological_distance_km": 0.0,
                "mean_min_hops": 0.0,
                "mean_max_congestion": 0.0,
                "mean_queuing_delay_ms": 0.0,
                "crosses_shock_link_pct": 0.0,
            }

        hops_traversed = float(sub_df["hop_count"].mean()) if "hop_count" in sub_df.columns else 0.0
        max_cong = float(sub_df["max_congestion"].mean()) if "max_congestion" in sub_df.columns else 0.0
        queue_delay = float(sub_df["queuing_delay_ms"].mean()) if "queuing_delay_ms" in sub_df.columns else 0.0

        topo_dists = []
        min_hops_list = []
        crosses_shock_list = []

        for _, row in sub_df.iterrows():
            s, d = int(row["src"]), int(row["dst"])
            if (s, d) in sp_cache:
                dist, mh, cs = sp_cache[(s, d)]
                topo_dists.append(dist)
                min_hops_list.append(mh)
                crosses_shock_list.append(1 if cs else 0)

        mean_dist = float(np.mean(topo_dists)) if topo_dists else 0.0
        mean_min_h = float(np.mean(min_hops_list)) if min_hops_list else 0.0
        shock_pct = float(np.mean(crosses_shock_list) * 100.0) if crosses_shock_list else 0.0

        return {
            "count": len(sub_df),
            "mean_hops_traversed": hops_traversed,
            "mean_topological_distance_km": mean_dist,
            "mean_min_hops": mean_min_h,
            "mean_max_congestion": max_cong,
            "mean_queuing_delay_ms": queue_delay,
            "crosses_shock_link_pct": shock_pct,
        }

    return {
        "delivered": extract_stats(delivered_df),
        "dropped": extract_stats(dropped_df),
        "total_packets": len(packet_df),
        "loss_rate_pct": (len(dropped_df) / len(packet_df) * 100.0) if len(packet_df) > 0 else 0.0,
    }
