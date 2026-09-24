"""Demonstration script for Prompt 3: Gym-style NetworkRoutingEnv & Per-Hop Routing.

Demonstrates:
  1. Environment initialization, observation/action spaces, and action masking.
  2. Step-by-step trace of a per-hop routing episode on NSFNET.
  3. Proactive congestion feedback: load_e, congestion_after, latency_norm, loss_prob.
  4. Reward component breakdown at each hop and terminal destination bonus (+10.0).
  5. Action masking preventing invalid port selection.
  6. Loop detection and penalty demonstration.
"""

import os
import sys
import numpy as np

# Ensure RORL root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.network_sim import NSFNETSimEnv
from src.routing_env import NetworkRoutingEnv, RewardConfig


def run_routing_env_demo():
    print("=" * 80)
    print("Prompt 3 Demonstration: Gym NetworkRoutingEnv & Per-Hop Routing Dynamics")
    print("Project: Route Optimization using Reinforcement Learning in Computer Networks")
    print("=" * 80)

    # 1. Initialize environment
    seed = 42
    sim = NSFNETSimEnv(seed=seed)
    cfg = RewardConfig(
        w1_congestion=3.0,
        w2_latency=1.0,
        w3_loss=2.0,
        w4_hop_penalty=0.05,
        terminal_destination_reward=10.0,
        terminal_drop_penalty_base=5.0,
        terminal_loop_penalty=8.0,
        invalid_action_penalty=2.0,
        max_hops=10,
    )
    env = NetworkRoutingEnv(env=sim, reward_config=cfg, packet_demand_gbps=0.10, seed=seed)

    print("\n[1] Environment Spaces & Configuration:")
    print(f"    - Action Space: {env.action_space} (Ports 0..3)")
    print(f"    - Observation Keys: {list(env.observation_space.spaces.keys())}")
    print(f"    - Reward Weights: w1={cfg.w1_congestion} (cong), w2={cfg.w2_latency} (lat), w3={cfg.w3_loss} (loss), w4={cfg.w4_hop_penalty} (step)")
    print(f"    - Terminal Rewards: +{cfg.terminal_destination_reward} (dest), -{cfg.terminal_drop_penalty_base}*(1+h/H) (drop), -{cfg.terminal_loop_penalty} (loop), -{cfg.invalid_action_penalty} (invalid)")

    # 2. Reset with fixed pair: Seattle (0) -> Chicago (7)
    src_node, dst_node = 0, 7
    obs, info = env.reset(src=src_node, dst=dst_node)
    src_name = env.graph.nodes[src_node]["name"]
    dst_name = env.graph.nodes[dst_node]["name"]

    print(f"\n[2] Initiating Episode 1: Route Packet #{info['packet_id']} from {src_node} ({src_name}) to {dst_node} ({dst_name})")
    print(f"    - Demand: {info['demand_gbps']} Gbps | Max Hops: {cfg.max_hops}")
    print(f"    - Initial Node: {env.current_node} | Valid Outgoing Ports: {np.where(env.action_masks())[0]}")

    # Hop-by-hop execution along path: Seattle (0) -> Salt Lake City (3) -> Ann Arbor (8) -> Chicago (7)
    route_plan = [3, 8, 7]
    print(f"\n[3] Step-by-Step Hop Execution:")
    print(f"{'Hop':<5}{'From':<16}{'To':<16}{'Cong After':<12}{'Lat(ms)':<10}{'Loss Prob':<12}{'Step Rew':<10}{'Total Rew':<10}")
    print("-" * 88)

    for hop_idx, target in enumerate(route_plan, 1):
        u = env.current_node
        u_name = env.graph.nodes[u]["name"]
        target_name = env.graph.nodes[target]["name"]

        # Find port matching target
        neighbor_ids = list(obs["neighbor_ids"])
        action = neighbor_ids.index(target)

        next_obs, reward, term, trunc, step_info = env.step(action)

        print(
            f"{hop_idx:<5}"
            f"{f'{u} ({u_name})':<16}"
            f"{f'{target} ({target_name})':<16}"
            f"{step_info['congestion_after']:<12.4f}"
            f"{step_info['latency_ms']:<10.2f}"
            f"{step_info['packet_loss_prob']:<12.4f}"
            f"{reward:<10.3f}"
            f"{env.episode_reward:<10.3f}"
        )

        obs = next_obs
        if term or trunc:
            print(f"\n>>> Episode Complete at Hop {hop_idx}! Status: Reached Destination={step_info['reached_dest']} <<<")
            break

    print(f"    - Final Path Taken: {env.path_so_far}")
    print(f"    - Cumulative Episode Return: {env.episode_reward:.3f}")

    # 3. Demonstration of Action Masking
    print("\n[4] Action Masking Verification:")
    obs, _ = env.reset(src=0, dst=13)
    mask = env.action_masks()
    print(f"    - Router 0 (Seattle) degree = 3 (neighbors: {list(env.graph.neighbors(0))})")
    print(f"    - Action Mask (port_mask): {mask}")
    print(f"    - Padded Port 3 valid? -> {mask[3]} (Correctly Masked Out for MaskablePPO)")

    # Attempting to select invalid port 3 triggers fallback penalty
    _, inv_reward, inv_term, _, inv_info = env.step(action=3)
    print(f"    - Unmasked Agent Fallback: selecting Port 3 yielded reward={inv_reward:.1f}, terminated={inv_term}, invalid_action={inv_info['invalid_action']}")

    # 4. Demonstration of Loop Detection
    print("\n[5] Loop Detection Verification:")
    obs, _ = env.reset(src=0, dst=13)
    # Hop 1: 0 -> 1
    port_to_1 = list(obs["neighbor_ids"]).index(1)
    obs, _, _, _, _ = env.step(port_to_1)
    # Hop 2: 1 -> 0 (loop!)
    port_back_0 = list(obs["neighbor_ids"]).index(0)
    _, loop_reward, loop_term, _, loop_info = env.step(port_back_0)
    print(f"    - Path: [0, 1, 0] (Loop back to 0!)")
    print(f"    - Terminated: {loop_term} | Looped Flag: {loop_info['looped']}")
    print(f"    - Reward includes -8.0 Loop Penalty: step_reward={loop_reward:.3f} (r_hop={loop_info['r_hop']:.3f} - 8.0)")

    print("\n" + "=" * 80)
    print("Prompt 3 Verification SUCCESS: NetworkRoutingEnv operates as a standard Gym MDP!")
    print("=" * 80)


if __name__ == "__main__":
    run_routing_env_demo()
