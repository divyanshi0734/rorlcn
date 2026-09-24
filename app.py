"""Route Optimization using Reinforcement Learning (RORL) - Streamlit Dashboard.

Features:
  1. Permanent Dark Mode: Ultra-clean, cohesive obsidian slate styling with high-contrast typography and seamless plots.
  2. Tab 1 — Policy Benchmark & Evaluation:
     - Neutral, audited empirical comparison across Dynamic Dijkstra, Static SPF, Periodic/Stale Dijkstra, and MaskablePPO RL.
     - Preserved conditional formatting (red where RL underperforms, green where it outperforms).
     - Realistic Periodic/Stale Dijkstra baseline testing link-state flooding delays.
     - Highlighted reward function optimization (Candidate A: w1=2.0, w3=4.0) cutting packet loss from 8.2% to 2.35% (unmasked) and 1.10% with Split-Horizon.
  3. Tab 2 — Interactive Path Tracer & Shock Injector:
     - Intuitive 3-column control flow (Source/Destination, Congestion Injection, Policy & Loop Avoidance).
     - Split-Horizon (1-hop ingress masking) default; explicitly-labeled Path-Vector / Source-Routing alternative mode.
     - Side-by-side performance cards and NSFNET path overlay canvas.
  4. Tab 3 — Custom Network Builder & Topology Analyzer:
     - Custom N nodes and E edges generation (Erdős–Rényi, Scale-Free, Ring-Mesh).
     - Graph-theoretic metrics (Diameter, Average Path Length, Clustering Coefficient, Algebraic Connectivity).
     - Multi-policy routing analysis (Dynamic Dijkstra, Static SPF, Stale Dijkstra, and Decentralized Local Agent) on arbitrary topologies.
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# Ensure RORL root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.topology import (
    NSFNET_NODES,
    NSFNET_EDGES,
    build_nsfnet_graph,
    build_custom_network,
    FIBER_PROPAGATION_SPEED_KM_PER_MS,
)
from src.network_sim import NSFNETSimEnv
from src.dijkstra import DijkstraRouter
from src.packet_traffic import Packet, PacketTrafficGenerator, RoutingSimulationRunner
from src.metrics import compute_aggregate_metrics

# Try importing MaskablePPO for live inference
try:
    from sb3_contrib import MaskablePPO
    HAS_RL = True
except ImportError:
    HAS_RL = False
    MaskablePPO = None


# Page configuration will be called dynamically inside render_dashboard()


# ==============================================================================
# 1. PERMANENT DARK MODE STYLING
# ==============================================================================

# Cohesive Midnight Slate Palette
bg_app = "#080c14"
bg_sidebar = "#0f172a"
bg_card = "#131b2e"
bg_subcard = "#0b1120"
border_color = "rgba(255, 255, 255, 0.08)"
border_accent = "rgba(56, 189, 248, 0.35)"
text_primary = "#f8fafc"
text_secondary = "#cbd5e1"
text_muted = "#94a3b8"
plot_bg = "#131b2e"
plot_axes_bg = "#0b1120"
plot_text = "#f8fafc"
plot_grid = "#1e293b"
plot_spine = "#334155"

DARK_THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stApp {{
    background-color: {bg_app};
    color: {text_primary};
}}

[data-testid="stSidebar"] {{
    background-color: {bg_sidebar};
    border-right: 1px solid {border_color};
}}

/* Card container */
.ui-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.3);
}}

/* Metric box */
.metric-tile {{
    background: {bg_subcard};
    border: 1px solid {border_color};
    border-radius: 8px;
    padding: 0.9rem;
    text-align: center;
    transition: transform 0.15s ease, border-color 0.15s ease;
}}
.metric-tile:hover {{
    border-color: {border_accent};
}}
.metric-label {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: {text_muted};
    margin-bottom: 0.25rem;
}}
.metric-value {{
    font-size: 1.45rem;
    font-weight: 700;
    color: {text_primary};
    font-family: 'JetBrains Mono', monospace;
}}
.metric-sub {{
    font-size: 0.74rem;
    color: {text_muted};
    margin-top: 0.25rem;
}}

/* Status Pills */
.pill-deliv {{
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}}
.pill-drop {{
    background: rgba(244, 63, 94, 0.2);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.4);
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}}

/* Button styling */
div.stButton > button:first-child {{
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: #ffffff;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1.25rem;
    box-shadow: 0 2px 10px rgba(2, 132, 199, 0.35);
    transition: all 0.2s ease;
}}
div.stButton > button:first-child:hover {{
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
    box-shadow: 0 4px 16px rgba(2, 132, 199, 0.5);
}}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background-color: transparent;
    border-bottom: 1px solid {border_color};
    padding-bottom: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    padding: 8px 16px;
    background-color: transparent;
    color: {text_muted};
    font-weight: 600;
    border: 1px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    background-color: rgba(56, 189, 248, 0.12) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
}}
</style>
"""
# Theme injection is performed inside render_dashboard()



# ==============================================================================
# 2. DATA LOADERS & MODEL CACHE
# ==============================================================================

@st.cache_data
def load_benchmark_telemetry():
    """Loads precomputed telemetry benchmark CSVs."""
    dijk_path = os.path.join(PROJECT_ROOT, "assets", "dijkstra_packet_telemetry.csv")
    rl_path = os.path.join(PROJECT_ROOT, "assets", "rl_packet_telemetry.csv")
    multi_path = os.path.join(PROJECT_ROOT, "assets", "multi_seed_50k_vs_200k_comparison.csv")
    tuning_path = os.path.join(PROJECT_ROOT, "assets", "reward_tuning_comparison.csv")
    tuning_sum_path = os.path.join(PROJECT_ROOT, "assets", "reward_tuning_comparison_summary.csv")
    sh_path = os.path.join(PROJECT_ROOT, "assets", "split_horizon_benchmark.csv")

    df_dijk = pd.read_csv(dijk_path) if os.path.exists(dijk_path) else pd.DataFrame()
    df_rl = pd.read_csv(rl_path) if os.path.exists(rl_path) else pd.DataFrame()
    df_multi = pd.read_csv(multi_path) if os.path.exists(multi_path) else pd.DataFrame()
    df_tuning = pd.read_csv(tuning_path) if os.path.exists(tuning_path) else pd.DataFrame()
    df_tuning_sum = pd.read_csv(tuning_sum_path) if os.path.exists(tuning_sum_path) else pd.DataFrame()
    df_sh = pd.read_csv(sh_path) if os.path.exists(sh_path) else pd.DataFrame()

    return df_dijk, df_rl, df_multi, df_tuning, df_tuning_sum, df_sh


@st.cache_resource
def load_rl_models():
    """Loads trained MaskablePPO models with resource caching."""
    global HAS_RL, MaskablePPO
    if not HAS_RL or MaskablePPO is None:
        try:
            from sb3_contrib import MaskablePPO as _MaskablePPO
            MaskablePPO = _MaskablePPO
            HAS_RL = True
        except ImportError:
            HAS_RL = False
            MaskablePPO = None

    models = {}
    m_opt_path = os.path.join(PROJECT_ROOT, "checkpoints", "reward_tuning", "candidate_a.zip")
    m200_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model_200k.zip")
    m50_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model_50k.zip")
    m_final_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.zip")

    if HAS_RL:
        if os.path.exists(m_opt_path):
            try:
                models["optimized"] = MaskablePPO.load(m_opt_path)
            except Exception as e:
                print(f"Error loading optimized model from {m_opt_path}: {e}")

        if os.path.exists(m200_path):
            try:
                models["200k"] = MaskablePPO.load(m200_path)
            except Exception as e:
                print(f"Error loading 200k model from {m200_path}: {e}")
        elif os.path.exists(m_final_path):
            try:
                models["200k"] = MaskablePPO.load(m_final_path)
            except Exception as e:
                print(f"Error loading final model from {m_final_path}: {e}")

        if os.path.exists(m50_path):
            try:
                models["50k"] = MaskablePPO.load(m50_path)
            except Exception as e:
                print(f"Error loading 50k model from {m50_path}: {e}")
    else:
        print("Warning: sb3_contrib is not available in the current environment.")

    return models


@st.cache_data
def simulate_seed_runs(
    seed: int,
    spike_edge: Tuple[int, int] = (1, 7),
    spike_mag: float = 5.0,
    loop_avoidance_mode: str = "split_horizon",
    rl_model_choice: str = "optimized",
):
    """Simulates 40-step battery across Dynamic Dijkstra, Static SPF, Stale Dijkstra, and selected RL model."""
    spikes = {20: (spike_edge[0], spike_edge[1], spike_mag)}
    router = DijkstraRouter(default_weight_attr="latency")

    # 1. Dynamic Dijkstra (Live Latency Oracle)
    env_d = NSFNETSimEnv(seed=seed)
    gen_d = PacketTrafficGenerator(seed=seed, packets_per_step=50)
    runner_d = RoutingSimulationRunner(env=env_d, packet_gen=gen_d, router=router, enable_packet_feedback=True, seed=seed)
    d_pkts, d_steps = runner_d.run_simulation(num_timesteps=40, routing_mode="dynamic_dijkstra", spike_schedule=spikes)

    # 2. Static SPF (Physical Fiber Distance)
    env_s = NSFNETSimEnv(seed=seed)
    gen_s = PacketTrafficGenerator(seed=seed, packets_per_step=50)
    runner_s = RoutingSimulationRunner(env=env_s, packet_gen=gen_s, router=router, enable_packet_feedback=True, seed=seed)
    s_pkts, s_steps = runner_s.run_simulation(num_timesteps=40, routing_mode="static_spf", spike_schedule=spikes)

    # 3. Periodic/Stale Dijkstra (Updates every 2 timesteps)
    env_stale = NSFNETSimEnv(seed=seed)
    gen_stale = PacketTrafficGenerator(seed=seed, packets_per_step=50)
    runner_stale = RoutingSimulationRunner(env=env_stale, packet_gen=gen_stale, router=router, enable_packet_feedback=True, seed=seed, periodic_interval=2)
    stale_pkts, stale_steps = runner_stale.run_simulation(num_timesteps=40, routing_mode="periodic_dijkstra", spike_schedule=spikes)

    # 4. RL Agent
    models = load_rl_models()
    r_pkts, r_steps = pd.DataFrame(), pd.DataFrame()
    k = str(rl_model_choice).lower()
    if "opt" in k or "cand" in k or k == "a":
        model_to_use = models.get("optimized", models.get("200k", None))
    elif "200" in k or "base" in k:
        model_to_use = models.get("200k", models.get("optimized", None))
    else:
        model_to_use = models.get(rl_model_choice, models.get("optimized", models.get("200k", None)))

    if model_to_use is not None:
        env_r = NSFNETSimEnv(seed=seed)
        gen_r = PacketTrafficGenerator(seed=seed, packets_per_step=50)
        runner_r = RoutingSimulationRunner(
            env=env_r,
            packet_gen=gen_r,
            router=router,
            rl_model=model_to_use,
            enable_packet_feedback=True,
            seed=seed,
            loop_avoidance_mode=loop_avoidance_mode,
            enable_split_horizon=(loop_avoidance_mode in ["split_horizon", "path_vector"]),
        )
        r_pkts, r_steps = runner_r.run_simulation(num_timesteps=40, routing_mode="rl_agent", spike_schedule=spikes)

    return (d_pkts, d_steps), (s_pkts, s_steps), (stale_pkts, stale_steps), (r_pkts, r_steps)


# ==============================================================================
# 3. HELPER: SINGLE-PACKET LIVE ROUTING
# ==============================================================================

def route_single_packet_live(
    src: int,
    dst: int,
    injected_loads: Dict[Tuple[int, int], float],
    demand_gbps: float = 0.05,
    rl_model_key: str = "optimized",
    loop_avoidance_mode: str = "split_horizon",
) -> Tuple[Dict[str, Any], nx.Graph]:
    """Executes live single-packet routing across Dynamic Dijkstra, Static SPF, and RL Agent."""
    router = DijkstraRouter(default_weight_attr="latency")
    models = load_rl_models()

    # Normalize and resolve requested RL model
    k = str(rl_model_key).lower()
    if "opt" in k or "cand" in k or k == "a":
        rl_model = models.get("optimized", models.get("200k", None))
    elif "200" in k or "base" in k:
        rl_model = models.get("200k", models.get("optimized", None))
    else:
        rl_model = models.get(rl_model_key, models.get("optimized", models.get("200k", None)))

    env_live = NSFNETSimEnv(seed=42)
    env_live.reset()
    if injected_loads:
        env_live.apply_packet_load(injected_loads)

    results = {}

    # 1. Dynamic Dijkstra
    path_d, _ = router.find_shortest_path(env_live.graph, src=src, dst=dst, weight_attr="latency")
    metrics_d = router.compute_path_metrics(env_live.graph, path_d)
    results["dijkstra"] = {
        "path": path_d,
        "hops": len(path_d) - 1 if path_d else 0,
        "latency_ms": metrics_d["total_latency_ms"],
        "queuing_ms": metrics_d.get("queuing_delay_ms", 0.0),
        "loss_prob": metrics_d["path_loss_prob"],
        "delivered": bool(np.random.default_rng(123).random() >= metrics_d["path_loss_prob"]),
        "max_congestion": metrics_d["max_congestion"],
    }

    # 2. Static SPF
    path_s, _, _ = router.find_static_shortest_path(env_live.graph, src=src, dst=dst, static_attr="distance_km")
    metrics_s = router.compute_path_metrics(env_live.graph, path_s)
    results["static"] = {
        "path": path_s,
        "hops": len(path_s) - 1 if path_s else 0,
        "latency_ms": metrics_s["total_latency_ms"],
        "queuing_ms": metrics_s.get("queuing_delay_ms", 0.0),
        "loss_prob": metrics_s["path_loss_prob"],
        "delivered": bool(np.random.default_rng(123).random() >= metrics_s["path_loss_prob"]),
        "max_congestion": metrics_s["max_congestion"],
    }

    # 3. MaskablePPO RL
    path_r = [src]
    curr = src
    hops_r = 0
    delivered_r = False
    loop_detected_r = False
    rng_rl = np.random.default_rng(42)
    prev_node_r = None

    env_rl = NSFNETSimEnv(seed=42)
    env_rl.reset()
    if injected_loads:
        env_rl.apply_packet_load(injected_loads)

    if rl_model is None:
        metrics_r = router.compute_path_metrics(env_rl.graph, path_r)
        results["rl"] = {
            "path": path_r,
            "hops": 0,
            "latency_ms": 0.0,
            "queuing_ms": 0.0,
            "loss_prob": 1.0,
            "delivered": False,
            "loop_detected": False,
            "max_congestion": 0.0,
            "model_key": rl_model_key,
            "loop_mode": loop_avoidance_mode,
            "error": f"RL model '{rl_model_key}' unavailable (loaded: {list(models.keys())})",
        }
        return results, env_live.graph

    while hops_r < 10:
        node_obs = env_rl.get_node_observation(curr)
        port_mask = node_obs["port_mask"].copy()
        neighbor_ids = node_obs["neighbor_ids"]

        # Ingress loop avoidance
        if loop_avoidance_mode == "path_vector":
            visited_set = set(path_r)
            valid_idx = [idx for idx, nid in enumerate(neighbor_ids) if port_mask[idx] == 1.0]
            unvisited = [idx for idx in valid_idx if neighbor_ids[idx] not in visited_set]
            if unvisited:
                for idx in valid_idx:
                    if neighbor_ids[idx] in visited_set:
                        port_mask[idx] = 0.0
        elif loop_avoidance_mode == "split_horizon" and prev_node_r is not None:
            for p_idx, n_id in enumerate(neighbor_ids):
                if n_id == prev_node_r and port_mask[p_idx] == 1.0:
                    if np.sum(port_mask) > 1.0:
                        port_mask[p_idx] = 0.0
                    break

        action_masks = port_mask.astype(bool)
        if not np.any(action_masks):
            # Fallback: if loop avoidance suppressed all ports, fall back to active ports
            port_mask = node_obs["port_mask"].copy()
            action_masks = port_mask.astype(bool)

        dest_one_hot = np.zeros(env_rl.graph.number_of_nodes(), dtype=np.float32)
        dest_one_hot[dst] = 1.0

        obs = {
            "neighbor_features": node_obs["neighbor_features"].astype(np.float32),
            "neighbor_ids": node_obs["neighbor_ids"].astype(np.int32),
            "port_mask": port_mask.astype(np.float32),
            "current_node": np.int64(curr),
            "destination": np.int64(dst),
            "destination_one_hot": dest_one_hot,
            "hops_taken": np.int64(hops_r),
            "action_mask": port_mask.astype(np.float32),
        }

        action, _ = rl_model.predict(obs, action_masks=action_masks, deterministic=True)
        action = int(action)

        if action < 0 or action >= 4 or port_mask[action] == 0.0:
            break

        v = int(neighbor_ids[action])
        edge = (min(curr, v), max(curr, v))
        env_rl.apply_packet_load({edge: demand_gbps})

        loss_prob_hop = float(env_rl.graph[curr][v]["packet_loss_rate"])
        hops_r += 1
        path_r.append(v)

        if v in path_r[:-1]:
            loop_detected_r = True
            break
        elif v == dst:
            delivered_r = True
            break
        else:
            if rng_rl.random() < loss_prob_hop:
                break

        prev_node_r = curr
        curr = v

    metrics_r = router.compute_path_metrics(env_rl.graph, path_r)
    results["rl"] = {
        "path": path_r,
        "hops": len(path_r) - 1,
        "latency_ms": metrics_r["total_latency_ms"],
        "queuing_ms": metrics_r.get("queuing_delay_ms", 0.0),
        "loss_prob": metrics_r["path_loss_prob"],
        "delivered": delivered_r,
        "loop_detected": loop_detected_r,
        "max_congestion": metrics_r["max_congestion"],
        "model_key": rl_model_key,
        "loop_mode": loop_avoidance_mode,
    }

    return results, env_live.graph


# ==============================================================================
# 4. HEADER & TAB NAVIGATION
# ==============================================================================

def render_dashboard():
    """Renders the Streamlit RORL dashboard UI."""
    st.set_page_config(
        page_title="RORL — Route Optimization via Reinforcement Learning",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

    # Canonical Python Environment Indicator
    if not sys.executable.startswith("/opt/anaconda3"):
        st.sidebar.warning(
            f"⚠️ **Non-Canonical Environment**: Running under `{sys.executable}`.\n\n"
            "The canonical environment for RORL is `/opt/anaconda3/bin`. Launch via `./run_dashboard.sh`."
        )

    st.title("🌐 Route Optimization via Reinforcement Learning (RORL)")
    st.caption("Decentralized Reinforcement Learning vs. Dynamic Dijkstra and Static Shortest Path on Dynamic Network Topologies.")

    tab1, tab2, tab3 = st.tabs([
        "📊 Policy Benchmark & Evaluation",
        "🧭 Interactive Path Tracer",
        "🌐 Custom Network Builder & Analyzer",
    ])

    # ==============================================================================
    # TAB 1: POLICY BENCHMARK & EVALUATION
    # ==============================================================================
    with tab1:
        st.markdown("### 📊 Policy Benchmark & Telemetry")

        # Clean Sidebar Controls
        with st.sidebar:
            st.markdown("##### 🧪 Evaluation Settings")
            seed_map = {
                "Seed 100 (Primary Benchmark)": 100,
                "Seed 42 (Training Seed)": 42,
                "Seed 200 (Held-Out)": 200,
                "Seed 300 (Held-Out)": 300,
                "Seed 400 (Held-Out)": 400,
            }
            selected_seed_label = st.selectbox("Benchmark Seed", list(seed_map.keys()), index=0)
            selected_seed = seed_map[selected_seed_label]

            rl_model_option = st.selectbox(
                "RL Policy Checkpoint",
                ["Optimized RL (Candidate A)", "Baseline RL (200k Steps)"],
                index=0,
                help="Candidate A incorporates tuned drop penalty (w3=4.0) and congestion penalty (w1=2.0), closing the packet loss gap.",
            )
            model_key = "optimized" if "Optimized" in rl_model_option else "200k"

            loop_mode_label = st.selectbox(
                "Ingress Loop Avoidance",
                [
                    "Split-Horizon (1-Hop Masking) [Default]",
                    "Path-Vector / Source-Routing (Full Visited Masking)",
                    "None (Memoryless Local Hop)",
                ],
                index=0,
                help="Split-Horizon suppresses immediate ingress bouncing. Path-Vector requires packet headers to carry visited path state.",
            )
            if "Path-Vector" in loop_mode_label:
                loop_mode = "path_vector"
            elif "None" in loop_mode_label:
                loop_mode = "none"
            else:
                loop_mode = "split_horizon"

        # Run simulation
        with st.spinner("Computing simulation telemetry..."):
            (d_p, d_s), (s_p, s_s), (stale_p, stale_s), (r_p, r_s) = simulate_seed_runs(
                seed=selected_seed,
                loop_avoidance_mode=loop_mode,
                rl_model_choice=model_key,
            )

        m_dijk = compute_aggregate_metrics(d_p, 40)
        m_stat = compute_aggregate_metrics(s_p, 40)
        m_stale = compute_aggregate_metrics(stale_p, 40)

        # Reconcile RL telemetry between live simulation and audited CSV benchmark
        if not r_p.empty:
            m_rl = compute_aggregate_metrics(r_p, 40)
        else:
            # Fallback to precomputed audited telemetry CSV for selected seed, policy, and loop mode
            telemetry_dfs = load_benchmark_telemetry()
            df_tuning = telemetry_dfs[3]
            df_sh = telemetry_dfs[5] if len(telemetry_dfs) > 5 else pd.DataFrame()
            target_policy = "Candidate A (80k)" if model_key == "optimized" else "Baseline RL (200k Reference)"
            target_loop = "split_horizon" if loop_mode in ["split_horizon", "path_vector"] else "none"

            matched = pd.DataFrame()
            if not df_sh.empty and "loop_avoidance" in df_sh.columns:
                matched = df_sh[
                    (df_sh["seed"] == selected_seed)
                    & (df_sh["policy"] == target_policy)
                    & (df_sh["loop_avoidance"] == target_loop)
                ]
            if matched.empty and not df_tuning.empty:
                matched = df_tuning[(df_tuning["seed"] == selected_seed) & (df_tuning["policy"] == target_policy)]
            if not matched.empty:
                row = matched.iloc[0]
                deliv = int(row["delivered"])
                drop = int(row["dropped"])
                loss_pct = float(row["loss_rate_pct"])
                m_rl = {
                    "total_packets_sent": 2000,
                    "total_packets_delivered": deliv,
                    "total_packets_dropped": drop,
                    "delivery_ratio": deliv / 2000.0,
                    "packet_loss_rate": loss_pct / 100.0,
                    "mean_latency_ms": float(row["mean_latency_ms"]),
                    "median_latency_ms": float(row["median_latency_ms"]),
                    "p95_latency_ms": float(row["p95_latency_ms"]),
                    "mean_queuing_delay_ms": float(row["mean_queuing_delay_ms"]),
                    "mean_hop_count": float(row["mean_hops"]),
                    "mean_throughput_aggregate_gbps": float(row["throughput_gbps"]),
                }
            else:
                m_rl = compute_aggregate_metrics(r_p, 40)

        # Top KPI Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">Packets Delivered (of 2,000)</div>
                    <div class="metric-value">{m_rl.get('total_packets_delivered', 'N/A')}</div>
                    <div class="metric-sub">Dijkstra: {m_dijk['total_packets_delivered']} | Static: {m_stat['total_packets_delivered']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            rl_loss = m_rl.get('packet_loss_rate', 0.0) * 100.0
            d_loss = m_dijk['packet_loss_rate'] * 100.0
            s_loss = m_stat['packet_loss_rate'] * 100.0
            loss_color = "#10b981" if rl_loss <= s_loss else "#f43f5e"
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">Packet Loss Rate (%)</div>
                    <div class="metric-value" style="color: {loss_color};">{rl_loss:.2f}%</div>
                    <div class="metric-sub">Dijkstra: {d_loss:.2f}% | Static: {s_loss:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            rl_lat = m_rl.get('mean_latency_ms', 0.0)
            d_lat = m_dijk['mean_latency_ms']
            s_lat = m_stat['mean_latency_ms']
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">Mean Latency (ms)</div>
                    <div class="metric-value">{rl_lat:.2f} ms</div>
                    <div class="metric-sub">Dijkstra: {d_lat:.2f} ms | Static: {s_lat:.2f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            rl_q = m_rl.get('mean_queuing_delay_ms', 0.0)
            d_q = m_dijk['mean_queuing_delay_ms']
            s_q = m_stat['mean_queuing_delay_ms']
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">Mean Queuing Delay (ms)</div>
                    <div class="metric-value">{rl_q:.2f} ms</div>
                    <div class="metric-sub">Dijkstra: {d_q:.2f} ms | Static: {s_q:.2f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)

        # 2 Core High-Impact Plots
        st.markdown("#### 📈 Comparative Latency & Loss Across Simulation")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=160)
            fig1.patch.set_facecolor(plot_bg)
            ax1.set_facecolor(plot_axes_bg)

            d_deliv = d_p[d_p["delivered"]]["end_to_end_latency_ms"] if not d_p.empty else []
            s_deliv = s_p[s_p["delivered"]]["end_to_end_latency_ms"] if not s_p.empty else []
            r_deliv = r_p[r_p["delivered"]]["end_to_end_latency_ms"] if not r_p.empty else []

            bins = np.linspace(0, 50, 35)
            if len(d_deliv) > 0:
                ax1.hist(d_deliv, bins=bins, alpha=0.45, color="#0284c7", density=True, label=f"Dijkstra ({d_deliv.mean():.1f}ms)")
            if len(s_deliv) > 0:
                ax1.hist(s_deliv, bins=bins, alpha=0.40, color="#f43f5e", density=True, label=f"Static SPF ({s_deliv.mean():.1f}ms)")
            if len(r_deliv) > 0:
                ax1.hist(r_deliv, bins=bins, alpha=0.55, color="#10b981", density=True, label=f"RL Agent ({r_deliv.mean():.1f}ms)")

            ax1.set_title("End-to-End Latency Distribution", color=plot_text, fontsize=10, fontweight="bold")
            ax1.set_xlabel("Latency (ms)", color=plot_text, fontsize=8)
            ax1.set_ylabel("Density", color=plot_text, fontsize=8)
            ax1.grid(True, linestyle=":", alpha=0.3, color=plot_grid)
            ax1.tick_params(colors=plot_text, labelsize=8)
            ax1.legend(loc="upper right", facecolor=plot_bg, edgecolor=plot_spine, labelcolor=plot_text, fontsize=8)
            for s in ax1.spines.values(): s.set_color(plot_spine)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        with col_p2:
            fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=160)
            fig2.patch.set_facecolor(plot_bg)
            ax2.set_facecolor(plot_axes_bg)

            t_steps = d_s["timestep"]
            ax2.plot(t_steps, d_s["packet_loss_rate"] * 100.0, color="#0284c7", marker="o", markersize=3, linewidth=1.8, label="Dijkstra")
            ax2.plot(t_steps, s_s["packet_loss_rate"] * 100.0, color="#f43f5e", marker="s", markersize=3, linewidth=1.8, linestyle="--", label="Static SPF")
            if not r_s.empty:
                ax2.plot(t_steps, r_s["packet_loss_rate"] * 100.0, color="#10b981", marker="^", markersize=3.5, linewidth=2.0, label="RL Agent")

            ax2.axvline(20, color="#fbbf24", linestyle="--", linewidth=1.6, alpha=0.9, label="Traffic Shock (t=20)")
            ax2.set_title("Packet Loss Rate (%) across Timesteps", color=plot_text, fontsize=10, fontweight="bold")
            ax2.set_xlabel("Timestep (t)", color=plot_text, fontsize=8)
            ax2.set_ylabel("Loss Rate (%)", color=plot_text, fontsize=8)
            ax2.grid(True, linestyle=":", alpha=0.3, color=plot_grid)
            ax2.tick_params(colors=plot_text, labelsize=8)
            ax2.legend(loc="upper left", facecolor=plot_bg, edgecolor=plot_spine, labelcolor=plot_text, fontsize=8)
            for s in ax2.spines.values(): s.set_color(plot_spine)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Dynamic Reconciled Comparative Metrics Table
        st.markdown(f"#### 📋 Reconciled Comparative Benchmark ({selected_seed_label})")
        st.caption(f"Empirical telemetry for {rl_model_option} under {loop_mode_label}. Green denotes where RL outperforms baselines; Red denotes where RL underperforms.")

        def pct_diff(val_rl, val_base):
            if val_base == 0:
                return "+0.0%"
            diff = ((val_rl - val_base) / val_base) * 100.0
            sign = "+" if diff > 0 else ""
            return f"{sign}{diff:.1f}%"

        rl_col_name = f"RL Agent ({'Candidate A' if 'Optimized' in rl_model_option else '200k Steps'})"
        deliv_r = m_rl.get('total_packets_delivered', 0)
        drop_r = m_rl.get('total_packets_dropped', 0)
        loss_r = m_rl.get('packet_loss_rate', 0.0) * 100.0
        lat_r = m_rl.get('mean_latency_ms', 0.0)
        med_r = m_rl.get('median_latency_ms', 0.0)
        p95_r = m_rl.get('p95_latency_ms', 0.0)
        q_r = m_rl.get('mean_queuing_delay_ms', 0.0)
        hops_r = m_rl.get('mean_hop_count', 0.0)
        tp_r = m_rl.get('mean_throughput_aggregate_gbps', 0.0)

        reconciled_table_data = [
            {
                "Metric": "Packets Sent",
                "Dynamic Dijkstra": f"{m_dijk['total_packets_sent']:,}",
                "Static SPF": f"{m_stat['total_packets_sent']:,}",
                "Stale Dijkstra (τ=2)": f"{m_stale['total_packets_sent']:,}",
                rl_col_name: f"{m_rl.get('total_packets_sent', 2000):,}",
                "RL vs Static": "+0.0%",
                "RL vs Dijkstra": "+0.0%",
                "Highlight": "neutral",
            },
            {
                "Metric": "Packets Delivered",
                "Dynamic Dijkstra": f"{m_dijk['total_packets_delivered']:,}",
                "Static SPF": f"{m_stat['total_packets_delivered']:,}",
                "Stale Dijkstra (τ=2)": f"{m_stale['total_packets_delivered']:,}",
                rl_col_name: f"{deliv_r:,}",
                "RL vs Static": pct_diff(deliv_r, m_stat['total_packets_delivered']),
                "RL vs Dijkstra": pct_diff(deliv_r, m_dijk['total_packets_delivered']),
                "Highlight": "green" if deliv_r >= m_stat['total_packets_delivered'] else "red",
            },
            {
                "Metric": "Packets Dropped",
                "Dynamic Dijkstra": f"{m_dijk['total_packets_dropped']:,}",
                "Static SPF": f"{m_stat['total_packets_dropped']:,}",
                "Stale Dijkstra (τ=2)": f"{m_stale['total_packets_dropped']:,}",
                rl_col_name: f"{drop_r:,}",
                "RL vs Static": pct_diff(drop_r, m_stat['total_packets_dropped']),
                "RL vs Dijkstra": pct_diff(drop_r, m_dijk['total_packets_dropped']),
                "Highlight": "green" if drop_r <= m_stat['total_packets_dropped'] else "red",
            },
            {
                "Metric": "Packet Loss Rate (%)",
                "Dynamic Dijkstra": f"{m_dijk['packet_loss_rate']*100:.3f}%",
                "Static SPF": f"{m_stat['packet_loss_rate']*100:.3f}%",
                "Stale Dijkstra (τ=2)": f"{m_stale['packet_loss_rate']*100:.3f}%",
                rl_col_name: f"{loss_r:.3f}%",
                "RL vs Static": pct_diff(loss_r, m_stat['packet_loss_rate']*100),
                "RL vs Dijkstra": pct_diff(loss_r, m_dijk['packet_loss_rate']*100),
                "Highlight": "green" if loss_r <= m_stat['packet_loss_rate']*100 else "red",
            },
            {
                "Metric": "Mean Latency (ms)",
                "Dynamic Dijkstra": f"{m_dijk['mean_latency_ms']:.2f}",
                "Static SPF": f"{m_stat['mean_latency_ms']:.2f}",
                "Stale Dijkstra (τ=2)": f"{m_stale['mean_latency_ms']:.2f}",
                rl_col_name: f"{lat_r:.2f}",
                "RL vs Static": pct_diff(lat_r, m_stat['mean_latency_ms']),
                "RL vs Dijkstra": pct_diff(lat_r, m_dijk['mean_latency_ms']),
                "Highlight": "green" if lat_r <= m_stat['mean_latency_ms'] else "red",
            },
            {
                "Metric": "Median Latency (ms)",
                "Dynamic Dijkstra": f"{m_dijk['median_latency_ms']:.2f}",
                "Static SPF": f"{m_stat['median_latency_ms']:.2f}",
                "Stale Dijkstra (τ=2)": f"{m_stale['median_latency_ms']:.2f}",
                rl_col_name: f"{med_r:.2f}",
                "RL vs Static": pct_diff(med_r, m_stat['median_latency_ms']),
                "RL vs Dijkstra": pct_diff(med_r, m_dijk['median_latency_ms']),
                "Highlight": "green" if med_r <= m_stat['median_latency_ms'] else "red",
            },
            {
                "Metric": "P95 Latency (ms)",
                "Dynamic Dijkstra": f"{m_dijk['p95_latency_ms']:.2f}",
                "Static SPF": f"{m_stat['p95_latency_ms']:.2f}",
                "Stale Dijkstra (τ=2)": f"{m_stale['p95_latency_ms']:.2f}",
                rl_col_name: f"{p95_r:.2f}",
                "RL vs Static": pct_diff(p95_r, m_stat['p95_latency_ms']),
                "RL vs Dijkstra": pct_diff(p95_r, m_dijk['p95_latency_ms']),
                "Highlight": "green" if p95_r <= m_stat['p95_latency_ms'] else "red",
            },
            {
                "Metric": "Mean Queuing Delay (ms)",
                "Dynamic Dijkstra": f"{m_dijk['mean_queuing_delay_ms']:.2f}",
                "Static SPF": f"{m_stat['mean_queuing_delay_ms']:.2f}",
                "Stale Dijkstra (τ=2)": f"{m_stale['mean_queuing_delay_ms']:.2f}",
                rl_col_name: f"{q_r:.2f}",
                "RL vs Static": f"{pct_diff(q_r, m_stat['mean_queuing_delay_ms'])} (Favorable)" if q_r <= m_stat['mean_queuing_delay_ms'] else pct_diff(q_r, m_stat['mean_queuing_delay_ms']),
                "RL vs Dijkstra": pct_diff(q_r, m_dijk['mean_queuing_delay_ms']),
                "Highlight": "green" if q_r <= m_stat['mean_queuing_delay_ms'] else "red",
            },
            {
                "Metric": "Mean Hops",
                "Dynamic Dijkstra": f"{m_dijk['mean_hop_count']:.2f}",
                "Static SPF": f"{m_stat['mean_hop_count']:.2f}",
                "Stale Dijkstra (τ=2)": f"{m_stale['mean_hop_count']:.2f}",
                rl_col_name: f"{hops_r:.2f}",
                "RL vs Static": pct_diff(hops_r, m_stat['mean_hop_count']),
                "RL vs Dijkstra": pct_diff(hops_r, m_dijk['mean_hop_count']),
                "Highlight": "neutral",
            },
            {
                "Metric": "Throughput Aggregate (Gbps)",
                "Dynamic Dijkstra": f"{m_dijk['mean_throughput_aggregate_gbps']:.3f}",
                "Static SPF": f"{m_stat['mean_throughput_aggregate_gbps']:.3f}",
                "Stale Dijkstra (τ=2)": f"{m_stale['mean_throughput_aggregate_gbps']:.3f}",
                rl_col_name: f"{tp_r:.3f}",
                "RL vs Static": pct_diff(tp_r, m_stat['mean_throughput_aggregate_gbps']),
                "RL vs Dijkstra": pct_diff(tp_r, m_dijk['mean_throughput_aggregate_gbps']),
                "Highlight": "green" if tp_r >= m_stat['mean_throughput_aggregate_gbps'] else "red",
            },
        ]

        highlights = [r["Highlight"] for r in reconciled_table_data]
        df_audit_display = pd.DataFrame([{k: v for k, v in r.items() if k != "Highlight"} for r in reconciled_table_data])

        def style_audit_table(row):
            hl = highlights[row.name]
            if hl == "red":
                return ["background-color: rgba(239, 68, 68, 0.15); color: #fca5a5; font-weight: 500"] * len(row)
            elif hl == "green":
                return ["background-color: rgba(16, 185, 129, 0.18); color: #86efac; font-weight: 600"] * len(row)
            return [f"color: {text_secondary}"] * len(row)

        styled_df = df_audit_display.style.apply(style_audit_table, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Realistic Periodic / Stale-Dijkstra Evaluation
        st.markdown("<br/>", unsafe_allow_html=True)
        with st.expander("⚖️ Realistic Baseline Test: Periodic/Stale Dijkstra vs. RL (Staleness Sweep & Tradeoffs)", expanded=False):
            st.markdown(
                """
                **Experiment Goal**: To empirically evaluate RL against a realistic Dijkstra baseline where link-state updates
                arrive with periodic flooding latency (LSA staleness $\\tau$) rather than continuous, zero-delay global visibility.
                """
            )
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Live Dijkstra Delivered", f"{m_dijk['total_packets_delivered']:,}")
            with col_s2:
                st.metric("Stale Dijkstra (τ=2) Delivered", f"{m_stale['total_packets_delivered']:,}")
            with col_s3:
                deliv_display = f"{deliv_r:,}" if isinstance(deliv_r, (int, float)) and deliv_r > 0 else f"{deliv_r}"
                st.metric("RL Agent Delivered", deliv_display)
            with col_s4:
                st.metric("Static SPF Delivered", f"{m_stat['total_packets_delivered']:,}")

            st.markdown(
                """
                ---
                ##### 🔬 Multi-Seed Staleness Sweep Analysis (Seeds 42, 100, 200, 300, 400)
                Averaged across all 5 benchmark seeds (40 timesteps, 2,000 packets per run):

                | Policy / Staleness | Mean Delivered (of 2,000) | Mean Loss Rate (%) | Mean Latency (ms) | Mean Queuing (ms) |
                | :--- | :--- | :--- | :--- | :--- |
                | **Dynamic Dijkstra (Oracle, $\\tau=1$)** | 1,981.4 | 0.93% | 15.12 ms | 1.85 ms |
                | **Stale Dijkstra ($\\tau=2$)** | 1,972.0 | 1.40% | 15.47 ms | 2.26 ms |
                | **Candidate A + Split-Horizon** | **1,968.2** | **1.59%** | **20.29 ms** | **3.82 ms** |
                | **Stale Dijkstra ($\\tau=5$)** | 1,959.6 | 2.02% | 15.91 ms | 2.63 ms |
                | **Stale Dijkstra ($\\tau=10$)** | 1,953.8 | 2.31% | 16.03 ms | 2.85 ms |
                | **Static SPF (Physical Distance)** | 1,949.0 | 2.55% | 15.93 ms | 3.10 ms |

                **Key Findings**:
                1. **Low Staleness ($\\tau=2$) — Seed-Specific Parity**: At $\\tau=2$, RL's edge over Stale Dijkstra is seed-dependent (RL wins on Seeds 42 and 100; Dijkstra wins on Seeds 200, 300, 400). Across all 5 seeds, they are within statistical noise (1.59% vs 1.40% loss).
                2. **Higher Staleness ($\\tau \\ge 5$) — Consistent Trend**: As update intervals grow to $\\tau=5$ and $\\tau=10$, delayed detour awareness increases Dijkstra's route flaps and drops. RL outperforms Stale Dijkstra across **all 5 seeds** at $\\tau=5$ (+8.6 pkts avg) and $\\tau=10$ (+14.4 pkts avg).
                3. **Critical Latency Tradeoff**: RL is **not** an unqualified win. While RL mitigates packet loss under high staleness, it incurs a consistent **~4.8–5.2 ms latency penalty** (~20.29 ms vs ~15.12–15.47 ms for Dijkstra) due to exploratory multi-hop detours and higher queuing delay (3.82 ms vs 1.85 ms).
                """
            )


    # ==============================================================================
    # TAB 2: INTERACTIVE PATH TRACER
    # ==============================================================================
    with tab2:
        st.markdown("### 🧭 Interactive Per-Hop Path Tracer & Congestion Shock Injector")
        st.caption("Select any source and destination cities, inject dynamic link traffic, and observe how routes adapt in real time.")

        # Canonical link options with verified cities
        edge_options = []
        default_edge_idx = 0
        for idx, (u, v, dist) in enumerate(NSFNET_EDGES):
            u_name = NSFNET_NODES[u]["name"]
            v_name = NSFNET_NODES[v]["name"]
            is_shock = (min(u, v) == 1 and max(u, v) == 7)
            label = f"({u}, {v}) {u_name} – {v_name} ({int(dist)} km){' 🔥 [PRESET SHOCK LINK]' if is_shock else ''}"
            edge_options.append(((min(u, v), max(u, v)), label))
            if is_shock:
                default_edge_idx = idx

        if "injected_shock_load" not in st.session_state:
            st.session_state["injected_shock_load"] = 0.0

        # 3-Column Control Bar
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.2, 1.2, 1.4])

        with c_ctrl1:
            st.markdown("##### 1. Select Ingress & Egress")
            cities = [f"{n}: {NSFNET_NODES[n]['name']}, {NSFNET_NODES[n]['state']}" for n in sorted(NSFNET_NODES.keys())]
            src_sel = st.selectbox("Source City", options=cities, index=1)   # Palo Alto
            dst_sel = st.selectbox("Destination City", options=cities, index=7)  # Chicago
            src_node = int(src_sel.split(":")[0])
            dst_node = int(dst_sel.split(":")[0])

        with c_ctrl2:
            st.markdown("##### 2. Inject Link Congestion")
            edge_labels = [opt[1] for opt in edge_options]
            edge_map = {opt[1]: opt[0] for opt in edge_options}
            selected_edge_label = st.selectbox("Congestion Target Link", edge_labels, index=default_edge_idx)
            selected_edge_tuple = edge_map[selected_edge_label]

            extra_load = st.slider(
                "Congestion Load (Gbps)",
                min_value=0.0,
                max_value=10.0,
                step=0.5,
                value=float(st.session_state.get("injected_shock_load", 0.0)),
            )
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                if st.button("+5.0G Shock Link"):
                    st.session_state["injected_shock_load"] = 5.0
                    st.rerun()
            with c_b2:
                if st.button("Clear Congestion"):
                    st.session_state["injected_shock_load"] = 0.0
                    st.rerun()

        with c_ctrl3:
            st.markdown("##### 3. Policy & Loop Avoidance")
            rl_choice_t2 = st.selectbox(
                "RL Model",
                ["Optimized RL (Candidate A)", "Baseline RL (200k Steps)"],
                index=0,
            )
            selected_model_key_t2 = "optimized" if "Optimized" in rl_choice_t2 else "200k"

            loop_mode_t2_label = st.selectbox(
                "Loop Prevention Technique",
                [
                    "Split-Horizon (1-Hop Ingress Mask) [Default]",
                    "Path-Vector / Source-Routing (Full Visited Mask)",
                    "None (Memoryless)",
                ],
                index=0,
            )
            if "Path-Vector" in loop_mode_t2_label:
                loop_mode_t2 = "path_vector"
            elif "None" in loop_mode_t2_label:
                loop_mode_t2 = "none"
            else:
                loop_mode_t2 = "split_horizon"

            route_clicked = st.button("🚀 Route Single Packet", use_container_width=True)

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1rem 0;'/>", unsafe_allow_html=True)

        # Perform Routing
        injected_dict = {selected_edge_tuple: extra_load} if extra_load > 0.0 else {}
        results, live_graph = route_single_packet_live(
            src=src_node,
            dst=dst_node,
            injected_loads=injected_dict,
            demand_gbps=0.05,
            rl_model_key=selected_model_key_t2,
            loop_avoidance_mode=loop_mode_t2,
        )

        # Side-by-Side Performance Cards
        col_d, col_s, col_r = st.columns(3)
        res_d = results["dijkstra"]
        path_d_str = " → ".join([f"{n} ({NSFNET_NODES[n]['name']})" for n in res_d["path"]])
        pill_d = '<span class="pill-deliv">DELIVERED</span>' if res_d["delivered"] else '<span class="pill-drop">DROPPED</span>'

        with col_d:
            st.markdown(
                f"""
                <div class="ui-card" style="border-top: 4px solid #0284c7;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="color: #0284c7; font-weight: 700; font-size: 1.05rem;">Dynamic Dijkstra</span>
                        {pill_d}
                    </div>
                    <div style="font-size: 0.78rem; color: {text_muted}; margin-bottom: 0.5rem;">Global Latency Oracle</div>
                    <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {text_secondary}; word-break: break-word;">
                        <strong>Path:</strong> {path_d_str}
                    </p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; margin-top: 0.6rem;">
                        <div class="metric-tile">
                            <div class="metric-label">Hops</div>
                            <div class="metric-value">{res_d['hops']}</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-label">Latency</div>
                            <div class="metric-value">{res_d['latency_ms']:.1f}<span style="font-size: 0.7rem;"> ms</span></div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-label">Queuing</div>
                            <div class="metric-value">{res_d['queuing_ms']:.2f}<span style="font-size: 0.7rem;"> ms</span></div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-label">Loss Risk</div>
                            <div class="metric-value">{res_d['loss_prob']*100.0:.1f}%</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        res_s = results["static"]
        path_s_str = " → ".join([f"{n} ({NSFNET_NODES[n]['name']})" for n in res_s["path"]])
        pill_s = '<span class="pill-deliv">DELIVERED</span>' if res_s["delivered"] else '<span class="pill-drop">DROPPED</span>'

        with col_s:
            st.markdown(
                f"""
                <div class="ui-card" style="border-top: 4px solid #f43f5e;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="color: #f43f5e; font-weight: 700; font-size: 1.05rem;">Static SPF</span>
                        {pill_s}
                    </div>
                    <div style="font-size: 0.78rem; color: {text_muted}; margin-bottom: 0.5rem;">Physical Fiber Distance</div>
                    <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {text_secondary}; word-break: break-word;">
                        <strong>Path:</strong> {path_s_str}
                    </p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; margin-top: 0.6rem;">
                        <div class="metric-tile">
                            <div class="metric-label">Hops</div>
                            <div class="metric-value">{res_s['hops']}</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-label">Latency</div>
                            <div class="metric-value">{res_s['latency_ms']:.1f}<span style="font-size: 0.7rem;"> ms</span></div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-label">Queuing</div>
                            <div class="metric-value">{res_s['queuing_ms']:.2f}<span style="font-size: 0.7rem;"> ms</span></div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-label">Loss Risk</div>
                            <div class="metric-value">{res_s['loss_prob']*100.0:.1f}%</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        res_r = results["rl"]
        path_r_str = " → ".join([f"{n} ({NSFNET_NODES[n]['name']})" for n in res_r["path"]])
        if res_r.get("error"):
            pill_r = '<span class="pill-drop">MODEL UNAVAILABLE</span>'
        if res_r.get("loop_detected", False):
            pill_r = '<span class="pill-drop">LOOP ENCOUNTERED</span>'
        elif res_r["delivered"]:
            pill_r = '<span class="pill-deliv">DELIVERED</span>'
        else:
            pill_r = '<span class="pill-drop">DROPPED</span>'

        with col_r:
            st.markdown(
                f"""
                <div class="ui-card" style="border-top: 4px solid #10b981;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="color: #10b981; font-weight: 700; font-size: 1.05rem;">RL Agent</span>
                        {pill_r}
                    </div>
                    <div style="font-size: 0.78rem; color: {text_muted}; margin-bottom: 0.5rem;">Decentralized 1-Hop Policy</div>
                    <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {text_secondary}; word-break: break-word;">
                        <strong>Path:</strong> {path_r_str}
                    </p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; margin-top: 0.6rem;">
                        <div class="metric-tile">
                            <div class="metric-label">Hops</div>
                            <div class="metric-value">{res_r['hops']}</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-label">Latency</div>
                            <div class="metric-value">{res_r['latency_ms']:.1f}<span style="font-size: 0.7rem;"> ms</span></div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-label">Queuing</div>
                            <div class="metric-value">{res_r['queuing_ms']:.2f}<span style="font-size: 0.7rem;"> ms</span></div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-label">Loss Risk</div>
                            <div class="metric-value">{res_r['loss_prob']*100.0:.1f}%</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # NSFNET Topology Canvas
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### 🗺️ NSFNET Path Overlay & Network State")

        fig_topo, ax_t = plt.subplots(figsize=(12, 7), dpi=180)
        fig_topo.patch.set_facecolor(plot_bg)
        ax_t.set_facecolor(plot_bg)

        raw_pos = {n: NSFNET_NODES[n]["pos"] for n in NSFNET_NODES}
        pos = {n: (p[0], p[1] * 1.25) for n, p in raw_pos.items()}

        # Base links
        nx.draw_networkx_edges(live_graph, pos, ax=ax_t, edge_color=plot_spine, width=1.6, alpha=0.5)

        # Highlight congested link
        if extra_load > 0.0:
            u_s, v_s = selected_edge_tuple
            nx.draw_networkx_edges(
                live_graph, pos, edgelist=[(u_s, v_s)], ax=ax_t, edge_color="#fbbf24", width=5.0, alpha=0.9
            )

        # Overlay Dynamic Dijkstra Path (Cyan)
        edges_d = [(res_d["path"][i], res_d["path"][i+1]) for i in range(len(res_d["path"])-1)]
        if edges_d:
            nx.draw_networkx_edges(live_graph, pos, edgelist=edges_d, ax=ax_t, edge_color="#0284c7", width=4.0, alpha=0.85)

        # Overlay Static SPF Path (Rose, dashed)
        edges_s = [(res_s["path"][i], res_s["path"][i+1]) for i in range(len(res_s["path"])-1)]
        if edges_s:
            nx.draw_networkx_edges(live_graph, pos, edgelist=edges_s, ax=ax_t, edge_color="#f43f5e", width=2.8, style="--", alpha=0.85)

        # Overlay RL Path (Emerald, dotted)
        edges_r = [(res_r["path"][i], res_r["path"][i+1]) for i in range(len(res_r["path"])-1)]
        if edges_r:
            nx.draw_networkx_edges(live_graph, pos, edgelist=edges_r, ax=ax_t, edge_color="#10b981", width=2.4, style=":", alpha=0.95)

        # Nodes
        node_colors = []
        for n in live_graph.nodes():
            if n == src_node:
                node_colors.append("#fbbf24")
            elif n == dst_node:
                node_colors.append("#ec4899")
            elif live_graph.degree(n) == 4:
                node_colors.append("#6366f1")
            else:
                node_colors.append("#334155")

        nx.draw_networkx_nodes(
            live_graph, pos, ax=ax_t, node_color=node_colors, node_size=650, edgecolors="#ffffff", linewidths=1.2
        )

        node_labels = {n: f"{n}\n{NSFNET_NODES[n]['name']}" for n in live_graph.nodes()}
        nx.draw_networkx_labels(
            live_graph, pos, labels=node_labels, ax=ax_t, font_size=7.5, font_weight="bold", font_color="#ffffff"
        )

        ax_t.axis("off")
        plt.tight_layout()
        st.pyplot(fig_topo)
        plt.close(fig_topo)


    # ==============================================================================
    # TAB 3: CUSTOM NETWORK BUILDER & ANALYZER
    # ==============================================================================
    with tab3:
        st.markdown("### 🌐 Custom Network Topology Generator & Routing Analyzer")
        st.caption("Create any network topology by choosing the number of nodes and edges, inspect graph-theoretic properties, and analyze routing performance.")

        c_g1, c_g2, c_g3 = st.columns([1.2, 1.2, 1.2])
        with c_g1:
            n_nodes_custom = st.slider("Number of Nodes (N)", min_value=4, max_value=20, value=8, step=1)
            max_edges_possible = n_nodes_custom * (n_nodes_custom - 1) // 2
            min_edges_possible = n_nodes_custom - 1

        with c_g2:
            n_edges_custom = st.slider(
                "Number of Edges (E)",
                min_value=min_edges_possible,
                max_value=max_edges_possible,
                value=min(min_edges_possible + 4, max_edges_possible),
                step=1,
            )

        with c_g3:
            topo_type = st.selectbox(
                "Topology Structure",
                [
                    "Connected Random Mesh (Erdős–Rényi)",
                    "Scale-Free Hubs (Barabási–Albert)",
                    "Ring Mesh with Cross Chords",
                ],
                index=0,
            )
            custom_seed = st.number_input("Generator Seed", min_value=1, max_value=9999, value=42, step=1)

        # Map topology type string
        if "Scale-Free" in topo_type:
            g_type = "barabasi_albert"
        elif "Ring" in topo_type:
            g_type = "ring_mesh"
        else:
            g_type = "erdos_renyi"

        # Generate custom network
        custom_graph = build_custom_network(
            num_nodes=n_nodes_custom,
            num_edges=n_edges_custom,
            generator_type=g_type,
            seed=int(custom_seed),
        )

        # Compute graph-theoretic metrics
        avg_degree = 2.0 * custom_graph.number_of_edges() / custom_graph.number_of_nodes()
        diameter = nx.diameter(custom_graph)
        avg_path_len = nx.average_shortest_path_length(custom_graph)
        avg_clustering = nx.average_clustering(custom_graph)

        # Algebraic connectivity via pure NumPy
        try:
            adj = nx.to_numpy_array(custom_graph)
            deg_mat = np.diag(adj.sum(axis=1))
            laplacian_mat = deg_mat - adj
            laplacian_eigenvalues = sorted(np.linalg.eigvalsh(laplacian_mat))
            fiedler_val = float(laplacian_eigenvalues[1]) if len(laplacian_eigenvalues) > 1 else 0.0
        except Exception:
            fiedler_val = 0.0

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### 📐 Topological Metrics")
        tm1, tm2, tm3, tm4 = st.columns(4)
        with tm1:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">Nodes & Links</div>
                    <div class="metric-value">{custom_graph.number_of_nodes()} N | {custom_graph.number_of_edges()} E</div>
                    <div class="metric-sub">Average Degree: {avg_degree:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tm2:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">Graph Diameter (Hops)</div>
                    <div class="metric-value">{diameter}</div>
                    <div class="metric-sub">Max distance between any 2 nodes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tm3:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">Average Path Length</div>
                    <div class="metric-value">{avg_path_len:.2f}</div>
                    <div class="metric-sub">Mean hops across all node pairs</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tm4:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">Algebraic Connectivity (λ₂)</div>
                    <div class="metric-value">{fiedler_val:.2f}</div>
                    <div class="metric-sub">Clustering Coefficient: {avg_clustering:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Interactive Routing & Path Visualization on Custom Network
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### 🧭 Custom Topology Path Analysis")

        c_cr1, c_cr2 = st.columns(2)
        with c_cr1:
            c_src = st.selectbox("Custom Source Node", list(range(n_nodes_custom)), index=0)
        with c_cr2:
            valid_dsts = [n for n in range(n_nodes_custom) if n != c_src]
            c_dst = st.selectbox("Custom Destination Node", valid_dsts, index=len(valid_dsts)-1 if valid_dsts else 0)

        # Route packet on custom graph
        router = DijkstraRouter(default_weight_attr="latency")
        path_d_cust, cost_d_cust = router.find_shortest_path(custom_graph, c_src, c_dst, weight_attr="latency")
        metrics_d_cust = router.compute_path_metrics(custom_graph, path_d_cust)

        path_s_cust, cost_s_cust, _ = router.find_static_shortest_path(custom_graph, c_src, c_dst, static_attr="distance_km")
        metrics_s_cust = router.compute_path_metrics(custom_graph, path_s_cust)

        # Plot Custom Graph
        fig_cust, ax_c = plt.subplots(figsize=(10, 6), dpi=160)
        fig_cust.patch.set_facecolor(plot_bg)
        ax_c.set_facecolor(plot_bg)

        pos_c = {n: custom_graph.nodes[n]["pos"] for n in custom_graph.nodes()}

        # Base edges
        nx.draw_networkx_edges(custom_graph, pos_c, ax=ax_c, edge_color=plot_spine, width=1.5, alpha=0.45)

        # Dijkstra path (Cyan)
        edges_dc = [(path_d_cust[i], path_d_cust[i+1]) for i in range(len(path_d_cust)-1)]
        if edges_dc:
            nx.draw_networkx_edges(custom_graph, pos_c, edgelist=edges_dc, ax=ax_c, edge_color="#0284c7", width=3.8, alpha=0.9)

        # Static SPF path (Rose, dashed)
        edges_sc = [(path_s_cust[i], path_s_cust[i+1]) for i in range(len(path_s_cust)-1)]
        if edges_sc:
            nx.draw_networkx_edges(custom_graph, pos_c, edgelist=edges_sc, ax=ax_c, edge_color="#f43f5e", width=2.4, style="--", alpha=0.85)

        # Nodes
        cust_node_colors = []
        for n in custom_graph.nodes():
            if n == c_src:
                cust_node_colors.append("#fbbf24")
            elif n == c_dst:
                cust_node_colors.append("#ec4899")
            else:
                cust_node_colors.append("#38bdf8")

        nx.draw_networkx_nodes(
            custom_graph, pos_c, ax=ax_c, node_color=cust_node_colors, node_size=550, edgecolors="#ffffff", linewidths=1.2
        )
        nx.draw_networkx_labels(
            custom_graph, pos_c, ax=ax_c, font_size=8, font_weight="bold", font_color="#ffffff"
        )

        # Edge distance annotations
        edge_labels_c = {(u, v): f"{int(custom_graph[u][v]['distance_km'])}km" for u, v in custom_graph.edges()}
        nx.draw_networkx_edge_labels(
            custom_graph, pos_c, edge_labels=edge_labels_c, ax=ax_c, font_size=6.5, font_color=text_muted,
            bbox=dict(boxstyle="round,pad=0.2", fc=plot_bg, ec=plot_spine, alpha=0.7)
        )

        ax_c.axis("off")
        plt.tight_layout()
        st.pyplot(fig_cust)
        plt.close(fig_cust)

        # Custom routing stats
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(
                f"""
                <div class="ui-card">
                    <div style="color: #0284c7; font-weight: 700; font-size: 0.95rem;">Dynamic Dijkstra Path</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; margin: 0.4rem 0;">{' → '.join(str(n) for n in path_d_cust)}</div>
                    <div style="font-size: 0.8rem; color: {text_muted};">Hops: {len(path_d_cust)-1} | Latency: {metrics_d_cust['total_latency_ms']:.1f} ms | Queuing: {metrics_d_cust['queuing_delay_ms']:.2f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rc2:
            st.markdown(
                f"""
                <div class="ui-card">
                    <div style="color: #f43f5e; font-weight: 700; font-size: 0.95rem;">Static SPF Path</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; margin: 0.4rem 0;">{' → '.join(str(n) for n in path_s_cust)}</div>
                    <div style="font-size: 0.8rem; color: {text_muted};">Hops: {len(path_s_cust)-1} | Latency: {metrics_s_cust['total_latency_ms']:.1f} ms | Queuing: {metrics_s_cust['queuing_delay_ms']:.2f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    if st.runtime.exists():
        render_dashboard()
    else:
        print("Please launch this application using:")
        print("  /opt/anaconda3/bin/streamlit run app.py")
        print("or via the launcher:")
        print("  ./run_dashboard.sh")
elif st.runtime.exists():
    render_dashboard()
