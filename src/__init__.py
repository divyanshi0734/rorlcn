"""Route Optimization using Reinforcement Learning in Computer Networks (RORL).
Package containing topology, dynamic link-state simulation, and telemetry.
"""

from .topology import build_nsfnet_graph, NSFNET_NODES, NSFNET_EDGES
from .dynamic_link_state import DynamicLinkStateEngine
from .traffic_generator import TrafficGenerator
from .network_sim import NSFNETSimEnv
from .dijkstra import DijkstraRouter
from .packet_traffic import Packet, PacketTrafficGenerator, RoutingSimulationRunner
from .metrics import (
    compute_aggregate_metrics,
    compare_routing_policies,
    compare_three_policies,
    compute_dropped_packet_diagnostics,
)
from .routing_env import NetworkRoutingEnv, RewardConfig
try:
    from .train_agent import PPOConfig, train_routing_agent, RoutingExperimentCallback
except ImportError:
    PPOConfig = None
    train_routing_agent = None
    RoutingExperimentCallback = None


__all__ = [
    "build_nsfnet_graph",
    "NSFNET_NODES",
    "NSFNET_EDGES",
    "DynamicLinkStateEngine",
    "TrafficGenerator",
    "NSFNETSimEnv",
    "DijkstraRouter",
    "Packet",
    "PacketTrafficGenerator",
    "RoutingSimulationRunner",
    "compute_aggregate_metrics",
    "compare_routing_policies",
    "compare_three_policies",
    "compute_dropped_packet_diagnostics",
    "NetworkRoutingEnv",
    "RewardConfig",
    "PPOConfig",
    "train_routing_agent",
    "RoutingExperimentCallback",
]
