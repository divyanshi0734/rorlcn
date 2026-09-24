"""Gym-style Reinforcement Learning Environment for Per-Hop Packet Routing.

Wraps the NSFNET discrete-timestep simulation fabric and dynamic link-state engine.
Trains a centralized RL agent to make per-hop sequential packet routing decisions,
utilizing proactive closed-loop congestion feedback and a calibrated multi-objective reward.

================================================================================
ARCHITECTURAL ASYMMETRY NOTE (Prompt 2 vs. Prompt 3):
--------------------------------------------------------------------------------
1. Dynamic Dijkstra (Prompt 2):
   - Centralized full-path snapshot computation.
   - Evaluates a static snapshot of the global network state at decision time tau
     and assigns a fixed end-to-end route P = [s, v1, ..., d].
2. RL Agent in NetworkRoutingEnv (Prompt 3):
   - Decentralized / localized per-hop sequential decision making.
   - At current router u with destination d, selects only the immediate next hop v.
   - Proactive closed-loop congestion feedback: each hop immediately applies the
     packet's own demand to the link (apply_packet_load), penalizing the agent for
     pushing links toward saturation and enabling proactive congestion avoidance.
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import networkx as nx

# Gymnasium / Gym compatibility layer with standalone fallback
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = True
    except ImportError:
        HAS_GYMNASIUM = False

        # Lightweight standalone Space classes matching Gym/Gymnasium API
        class Space:
            pass

        class Discrete(Space):
            def __init__(self, n: int, seed: Optional[int] = None):
                self.n = n
                self.rng = np.random.default_rng(seed)
                self.shape = ()
                self.dtype = np.int64

            def sample(self, mask: Optional[np.ndarray] = None) -> int:
                if mask is not None:
                    valid = np.where(mask)[0]
                    if len(valid) > 0:
                        return int(self.rng.choice(valid))
                return int(self.rng.integers(0, self.n))

            def contains(self, x: Any) -> bool:
                return isinstance(x, (int, np.integer)) and 0 <= x < self.n

        class Box(Space):
            def __init__(self, low: Any, high: Any, shape: Tuple[int, ...], dtype: Any = np.float32):
                self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.array(low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.array(high, dtype=dtype)
                self.shape = shape
                self.dtype = dtype

            def contains(self, x: Any) -> bool:
                arr = np.asarray(x, dtype=self.dtype)
                return arr.shape == self.shape and np.all(arr >= self.low) and np.all(arr <= self.high)

        class DictSpace(Space):
            def __init__(self, spaces_dict: Dict[str, Space]):
                self.spaces = spaces_dict

            def __getitem__(self, key: str) -> Space:
                return self.spaces[key]

        class spaces:
            Discrete = Discrete
            Box = Box
            Dict = DictSpace

        class gym:
            class Env:
                pass

from .network_sim import NSFNETSimEnv
from .packet_traffic import Packet, PacketTrafficGenerator


@dataclass
class RewardConfig:
    """Hyperparameter configuration for per-hop routing reward function.

    Per-hop reward:
      r_hop = -w1 * congestion_after(e) - w2 * latency_norm(e) - w3 * packet_loss_prob(e) - w4

    Terminal rewards:
      +10.0 on destination reached (v == d)
      -5.0 * (1 + hops / max_hops) on packet dropped (Bernoulli trial fails)
      -8.0 on loop detected (v in path_so_far)
      -2.0 on invalid / padded port selected (fallback for unmasked agents)
    """
    w1_congestion: float = 3.0
    w2_latency: float = 1.0
    w3_loss: float = 2.0
    w4_hop_penalty: float = 0.05
    w5_progress: float = 0.0

    terminal_destination_reward: float = 10.0
    terminal_drop_penalty_base: float = 5.0
    terminal_loop_penalty: float = 8.0
    invalid_action_penalty: float = 2.0

    latency_norm_scale_ms: float = 100.0
    max_hops: int = 10
    enable_split_horizon: bool = False



class NetworkRoutingEnv(gym.Env):
    """Gym-style Reinforcement Learning Environment for Per-Hop Packet Routing.

    Observation Space (Dict):
      - 'neighbor_features': np.ndarray of shape (4, 4) with [latency_norm, congestion, avail_bw, loss_rate]
      - 'neighbor_ids': np.ndarray of shape (4,), padded with -1
      - 'port_mask': np.ndarray of shape (4,), 1.0 for valid neighbors, 0.0 for padded
      - 'current_node': int current router node ID
      - 'destination': int destination router node ID
      - 'destination_one_hot': np.ndarray of shape (num_nodes,) one-hot destination encoding
      - 'hops_taken': int hops traversed so far this episode
      - 'action_mask': np.ndarray of shape (4,), boolean/float mask for valid ports

    Action Space:
      Discrete(4): Outgoing port index (0, 1, 2, or 3).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        env: Optional[NSFNETSimEnv] = None,
        reward_config: Optional[RewardConfig] = None,
        max_hops: int = 10,
        packet_demand_gbps: float = 0.05,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 1. Underlying physical simulation environment
        self.env = env if env is not None else NSFNETSimEnv(seed=seed)
        self.graph = self.env.graph
        self.num_nodes = self.graph.number_of_nodes()
        self.d_max = 4  # Max degree for NSFNET

        # 2. Reward and operational configuration
        self.reward_config = reward_config if reward_config is not None else RewardConfig(max_hops=max_hops)
        self.max_hops = self.reward_config.max_hops
        self.packet_demand_gbps = packet_demand_gbps

        # 3. Seeded traffic generator for packet sampling
        self.packet_gen = PacketTrafficGenerator(
            nodes=list(self.graph.nodes()),
            packets_per_step=1,
            packet_demand_gbps=packet_demand_gbps,
            seed=seed,
        )

        # 4. Action and Observation Spaces
        self.action_space = spaces.Discrete(self.d_max)

        self.observation_space = spaces.Dict({
            "neighbor_features": spaces.Box(low=0.0, high=1.0, shape=(self.d_max, 4), dtype=np.float32),
            "neighbor_ids": spaces.Box(low=-1, high=self.num_nodes - 1, shape=(self.d_max,), dtype=np.int32),
            "port_mask": spaces.Box(low=0.0, high=1.0, shape=(self.d_max,), dtype=np.float32),
            "current_node": spaces.Discrete(self.num_nodes),
            "destination": spaces.Discrete(self.num_nodes),
            "destination_one_hot": spaces.Box(low=0.0, high=1.0, shape=(self.num_nodes,), dtype=np.float32),
            "hops_taken": spaces.Discrete(self.max_hops + 1),
            "action_mask": spaces.Box(low=0.0, high=1.0, shape=(self.d_max,), dtype=np.float32),
        })

        # Episode dynamic state
        self.current_packet: Optional[Packet] = None
        self.current_node: int = 0
        self.destination: int = 0
        self.hops_taken: int = 0
        self.path_so_far: List[int] = []
        self.episode_reward: float = 0.0
        self.done: bool = False

    def action_masks(self) -> np.ndarray:
        """Returns boolean mask of valid actions for MaskablePPO / sb3-contrib.

        True indicates a valid active outgoing port; False indicates a padded/invalid port.
        """
        obs = self.env.get_node_observation(self.current_node)
        port_mask = obs["port_mask"].copy()
        if self.reward_config.enable_split_horizon and len(self.path_so_far) >= 2:
            prev_node = self.path_so_far[-2]
            neighbor_ids = obs["neighbor_ids"]
            for p_idx, n_id in enumerate(neighbor_ids):
                if n_id == prev_node and port_mask[p_idx] == 1.0:
                    if np.sum(port_mask) > 1.0:
                        port_mask[p_idx] = 0.0
                    break
        return port_mask.astype(bool)


    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        src: Optional[int] = None,
        dst: Optional[int] = None,
        demand_gbps: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the environment for a new packet routing episode.

        Args:
            seed: Optional seed to reseed environment RNG.
            options: Optional dictionary for custom reset options.
            src: Optional fixed source node ID.
            dst: Optional fixed destination node ID (src != dst).
            demand_gbps: Optional traffic demand override.

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.packet_gen.reset(seed=seed)

        # Determine source and destination
        if src is not None and dst is not None:
            if src == dst:
                raise ValueError(f"Source and destination must be distinct, got src={src}, dst={dst}")
            s_node, d_node = int(src), int(dst)
        else:
            pkts = self.packet_gen.generate_packets(timestep=self.env.timestep, count=1)
            s_node, d_node = pkts[0].src, pkts[0].dst

        pkt_demand = demand_gbps if demand_gbps is not None else self.packet_demand_gbps
        self.current_packet = Packet(
            packet_id=self.packet_gen.packet_counter,
            timestep=self.env.timestep,
            src=s_node,
            dst=d_node,
            demand_gbps=pkt_demand,
        )

        self.current_node = s_node
        self.destination = d_node
        self.hops_taken = 0
        self.path_so_far = [s_node]
        self.episode_reward = 0.0
        self.done = False

        obs = self._get_observation()
        info = {
            "packet_id": self.current_packet.packet_id,
            "src": self.current_node,
            "dst": self.destination,
            "demand_gbps": pkt_demand,
            "path_so_far": list(self.path_so_far),
            "hops_taken": 0,
        }
        return obs, info

    def _get_observation(self) -> Dict[str, Any]:
        """Constructs the augmented observation dictionary at current_node."""
        node_obs = self.env.get_node_observation(self.current_node)

        # One-hot destination encoding
        dest_one_hot = np.zeros(self.num_nodes, dtype=np.float32)
        dest_one_hot[self.destination] = 1.0

        return {
            "neighbor_features": node_obs["neighbor_features"],
            "neighbor_ids": node_obs["neighbor_ids"],
            "port_mask": node_obs["port_mask"],
            "current_node": int(self.current_node),
            "destination": int(self.destination),
            "destination_one_hot": dest_one_hot,
            "hops_taken": int(self.hops_taken),
            "action_mask": node_obs["port_mask"].copy(),
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advances the packet by one hop based on the selected outgoing port index.

        Args:
            action: Outgoing port index in {0, 1, 2, 3}.

        Returns:
            Tuple of:
              - next_state: Observation dict for the new node.
              - reward: Computed per-hop and terminal reward.
              - terminated: Boolean flag indicating natural episode completion.
              - truncated: Boolean flag indicating max-hops timeout.
              - info: Diagnostic and telemetry telemetry dict.
        """
        if self.done:
            raise RuntimeError("Cannot step an environment that has already finished. Call reset() first.")

        u = self.current_node
        node_obs = self.env.get_node_observation(u)
        port_mask = node_obs["port_mask"]
        neighbor_ids = node_obs["neighbor_ids"]

        cfg = self.reward_config

        # ---------------------------------------------------------------------
        # 1. Action Validation & Invalid Port Handling
        # ---------------------------------------------------------------------
        if action < 0 or action >= self.d_max or port_mask[action] == 0.0:
            # Invalid action selected (fallback penalty for unmasked agents)
            reward = -float(cfg.invalid_action_penalty)
            self.done = True
            terminated = True
            truncated = False
            info = {
                "invalid_action": True,
                "current_node": u,
                "action": action,
                "hops_taken": self.hops_taken,
                "path_so_far": list(self.path_so_far),
                "reached_dest": False,
                "dropped": False,
                "looped": False,
            }
            return self._get_observation(), reward, terminated, truncated, info

        v = int(neighbor_ids[action])
        edge = (min(u, v), max(u, v))

        # Record pre-load state for telemetry
        congestion_before = float(self.graph[u][v]["congestion"])

        # ---------------------------------------------------------------------
        # 2. Proactive Closed-Loop Packet Load Feedback
        # ---------------------------------------------------------------------
        # Critically: add this packet's demand to the link BEFORE calculating reward.
        # This penalizes the agent for pushing a link toward saturation.
        self.env.apply_packet_load({edge: self.current_packet.demand_gbps})

        # Read post-load link metrics
        edge_data = self.graph[u][v]
        congestion_after = float(edge_data["congestion"])
        latency_raw_ms = float(edge_data["latency"])
        latency_norm = float(np.clip(latency_raw_ms / cfg.latency_norm_scale_ms, 0.0, 1.0))
        loss_prob = float(edge_data["packet_loss_rate"])

        # ---------------------------------------------------------------------
        # 3. Per-Hop Step Reward Calculation
        # ---------------------------------------------------------------------
        # Progress reward: potential-based distance shaping to incentivize advancing toward destination
        progress_reward = 0.0
        if cfg.w5_progress > 0.0:
            try:
                hops_before = nx.shortest_path_length(self.graph, u, self.destination)
                hops_after = nx.shortest_path_length(self.graph, v, self.destination)
                progress_reward = cfg.w5_progress * float(hops_before - hops_after)
            except Exception:
                progress_reward = 0.0

        # r_hop = -w1 * congestion_after - w2 * latency_norm - w3 * loss_prob - w4 + progress_reward
        r_hop = (
            - cfg.w1_congestion * congestion_after
            - cfg.w2_latency * latency_norm
            - cfg.w3_loss * loss_prob
            - cfg.w4_hop_penalty
            + progress_reward
        )


        self.hops_taken += 1
        self.path_so_far.append(v)

        reward = r_hop
        terminated = False
        truncated = False
        looped = False
        reached_dest = False
        dropped = False

        # ---------------------------------------------------------------------
        # 4. Terminal Condition Evaluations (Strict Ordering)
        # ---------------------------------------------------------------------
        # Check A: Loop Detection (v already visited in this episode's path)
        if v in self.path_so_far[:-1]:
            looped = True
            terminated = True
            reward -= float(cfg.terminal_loop_penalty)

        # Check B: Destination Arrival (v == d) - Short-circuits packet drop check!
        elif v == self.destination:
            reached_dest = True
            terminated = True
            reward += float(cfg.terminal_destination_reward)

        # Check C: Intermediate Packet Loss (Bernoulli drop trial along link e)
        # Evaluated strictly if packet has not yet reached destination
        else:
            # Drop trial: Bernoulli random test against non-linear loss rate
            if self.rng.random() < loss_prob:
                dropped = True
                terminated = True
                drop_penalty = cfg.terminal_drop_penalty_base * (1.0 + self.hops_taken / cfg.max_hops)
                reward -= float(drop_penalty)

            # Check D: Max Hops Truncation
            elif self.hops_taken >= cfg.max_hops:
                truncated = True
                # Timeout without destination delivery

        self.done = terminated or truncated
        self.current_node = v
        self.episode_reward += reward

        info = {
            "current_node": v,
            "destination": self.destination,
            "hops_taken": self.hops_taken,
            "path_so_far": list(self.path_so_far),
            "congestion_before": congestion_before,
            "congestion_after": congestion_after,
            "latency_ms": latency_raw_ms,
            "latency_norm": latency_norm,
            "packet_loss_prob": loss_prob,
            "r_hop": r_hop,
            "reached_dest": reached_dest,
            "dropped": dropped,
            "looped": looped,
            "truncated": truncated,
            "invalid_action": False,
        }

        next_obs = self._get_observation()
        return next_obs, float(reward), terminated, truncated, info

    def render(self) -> str:
        """Renders diagnostic ASCII view of current packet routing progress."""
        u_name = self.graph.nodes[self.current_node]["name"]
        d_name = self.graph.nodes[self.destination]["name"]
        output = (
            f"[Packet #{self.current_packet.packet_id}] "
            f"Current: {self.current_node} ({u_name}) -> Dest: {self.destination} ({d_name}) | "
            f"Hops: {self.hops_taken}/{self.max_hops} | "
            f"Path: {self.path_so_far} | "
            f"Reward: {self.episode_reward:.3f} | Done: {self.done}"
        )
        print(output)
        return output
