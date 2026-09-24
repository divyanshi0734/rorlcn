"""NSFNET 14-node, 21-link network topology specification and builder.

Provides canonical nodes, geographical coordinates, physical fiber distances,
base propagation latencies, and NetworkX graph generation.
"""

from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import numpy as np

# Canonical NSFNET Nodes with geographical coordinates (lon, lat) and site names
NSFNET_NODES: Dict[int, Dict[str, Any]] = {
    0: {"name": "Seattle", "state": "WA", "pos": (-122.33, 47.60)},
    1: {"name": "Palo Alto", "state": "CA", "pos": (-122.14, 37.44)},
    2: {"name": "San Diego", "state": "CA", "pos": (-117.16, 32.71)},
    3: {"name": "Salt Lake City", "state": "UT", "pos": (-111.89, 40.76)},
    4: {"name": "Boulder", "state": "CO", "pos": (-105.27, 40.01)},
    5: {"name": "Lincoln", "state": "NE", "pos": (-96.70, 40.81)},
    6: {"name": "Houston", "state": "TX", "pos": (-95.36, 29.76)},
    7: {"name": "Chicago", "state": "IL", "pos": (-87.62, 41.87)},
    8: {"name": "Ann Arbor", "state": "MI", "pos": (-83.74, 42.28)},
    9: {"name": "Atlanta", "state": "GA", "pos": (-84.38, 33.74)},
    10: {"name": "Pittsburgh", "state": "PA", "pos": (-79.99, 40.44)},
    11: {"name": "Ithaca", "state": "NY", "pos": (-76.50, 42.44)},
    12: {"name": "Princeton", "state": "NJ", "pos": (-74.65, 40.35)},
    13: {"name": "Washington DC", "state": "DC", "pos": (-77.03, 38.90)},
}

# Canonical 21 links with physical fiber lengths in kilometers (km)
# Propagation speed in optical silica fiber: ~200 km/ms (~5 us/km)
NSFNET_EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 1300.0),   # Seattle - Palo Alto
    (0, 2, 1900.0),   # Seattle - San Diego
    (0, 3, 1100.0),   # Seattle - Salt Lake City
    (1, 2, 750.0),    # Palo Alto - San Diego
    (1, 7, 3000.0),   # Palo Alto - Chicago
    (2, 5, 2100.0),   # San Diego - Lincoln
    (3, 4, 600.0),    # Salt Lake City - Boulder
    (3, 8, 2400.0),   # Salt Lake City - Ann Arbor
    (4, 5, 750.0),    # Boulder - Lincoln
    (4, 9, 2200.0),   # Boulder - Atlanta
    (5, 6, 1200.0),   # Lincoln - Houston
    (5, 10, 1500.0),  # Lincoln - Pittsburgh
    (6, 11, 2300.0),  # Houston - Ithaca
    (7, 8, 380.0),    # Chicago - Ann Arbor
    (7, 12, 1150.0),  # Chicago - Princeton
    (8, 9, 1000.0),   # Ann Arbor - Atlanta
    (8, 12, 900.0),   # Ann Arbor - Princeton
    (9, 13, 900.0),   # Atlanta - Washington DC
    (10, 11, 400.0),  # Pittsburgh - Ithaca
    (10, 13, 350.0),  # Pittsburgh - Washington DC
    (11, 12, 350.0),  # Ithaca - Princeton
]

FIBER_PROPAGATION_SPEED_KM_PER_MS = 200.0  # km/ms (~200,000 km/s in silica glass)
DEFAULT_CAPACITY_GBPS = 10.0               # Default link capacity in Gbps


def build_nsfnet_graph(
    custom_edges: Optional[List[Tuple[int, int, float]]] = None,
    default_capacity_gbps: float = DEFAULT_CAPACITY_GBPS,
) -> nx.Graph:
    """Constructs the NSFNET 14-node, 21-link NetworkX graph.

    Args:
        custom_edges: Optional custom list of (u, v, distance_km) tuples.
        default_capacity_gbps: Nominal link capacity in Gigabits per second (Gbps).

    Returns:
        nx.Graph: Connected undirected graph with node and edge attributes.
    """
    g = nx.Graph()

    # Add 14 nodes with metadata
    for node_id, data in NSFNET_NODES.items():
        g.add_node(
            node_id,
            name=data["name"],
            state=data["state"],
            pos=data["pos"],
        )

    edges_to_add = custom_edges if custom_edges is not None else NSFNET_EDGES

    for edge_id, (u, v, dist_km) in enumerate(edges_to_add):
        # Base physical propagation latency in milliseconds (ms)
        base_prop_latency_ms = dist_km / FIBER_PROPAGATION_SPEED_KM_PER_MS
        
        g.add_edge(
            u,
            v,
            edge_id=edge_id,
            distance_km=dist_km,
            capacity_gbps=default_capacity_gbps,
            base_latency_ms=base_prop_latency_ms,
            # Dynamic attributes initialized to zero-load state:
            traffic_load_gbps=0.0,
            congestion=0.0,
            available_bandwidth=1.0,
            available_bandwidth_gbps=default_capacity_gbps,
            latency=base_prop_latency_ms,
            packet_loss_rate=0.0,
        )

    # Sanity checks
    assert g.number_of_nodes() == 14, f"Expected 14 nodes, got {g.number_of_nodes()}"
    assert g.number_of_edges() == len(edges_to_add), f"Expected {len(edges_to_add)} edges, got {g.number_of_edges()}"
    assert nx.is_connected(g), "NSFNET topology graph must be connected"

    return g


def get_canonical_edge_mapping(g: nx.Graph) -> Dict[Tuple[int, int], int]:
    """Returns mapping from undirected edge tuple (min(u, v), max(u, v)) to edge index [0..E-1]."""
    mapping = {}
    for idx, (u, v) in enumerate(sorted(g.edges())):
        mapping[(min(u, v), max(u, v))] = idx
    return mapping


def build_custom_network(
    num_nodes: int = 8,
    num_edges: int = 12,
    generator_type: str = "erdos_renyi",
    default_capacity_gbps: float = DEFAULT_CAPACITY_GBPS,
    seed: Optional[int] = 42,
) -> nx.Graph:
    """Constructs a connected custom network topology with realistic physical fiber metrics.

    Guarantees:
      - Always produces a connected graph (is_connected == True).
      - Exact node count `num_nodes`.
      - Clamped edge count `num_edges` in [N - 1, N*(N-1)//2].
      - 2D layout coordinates for plotting.
      - Full initialization of dynamic routing attributes (latency, congestion, capacity).

    Args:
        num_nodes: Number of router nodes (>= 3).
        num_edges: Desired number of bidirectional links.
        generator_type: 'erdos_renyi', 'barabasi_albert', or 'ring_mesh'.
        default_capacity_gbps: Nominal link bandwidth (default 10.0 Gbps).
        seed: Random seed for deterministic reproducibility.

    Returns:
        nx.Graph with populated node and edge attributes.
    """
    rng = np.random.default_rng(seed)
    num_nodes = max(3, int(num_nodes))
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_edges = max(num_nodes - 1, min(int(num_edges), max_possible_edges))

    g = nx.Graph()
    for i in range(num_nodes):
        g.add_node(i, name=f"Node {i}", state="Custom")

    if generator_type == "barabasi_albert" and num_nodes > 3 and num_edges >= num_nodes:
        # m edges per new node
        m = max(1, min(num_edges // num_nodes, num_nodes - 1))
        base_g = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
        # Add remaining edges to match num_edges
        all_possible = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes) if not base_g.has_edge(u, v)]
        rng.shuffle(all_possible)
        edges_to_add = list(base_g.edges()) + all_possible[: max(0, num_edges - base_g.number_of_edges())]
    elif generator_type == "ring_mesh":
        # Create a ring first
        edges_to_add = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        remaining = num_edges - len(edges_to_add)
        if remaining > 0:
            all_possible = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes) if (u, v) not in edges_to_add and (v, u) not in edges_to_add]
            rng.shuffle(all_possible)
            edges_to_add.extend(all_possible[:remaining])
    else:
        # Default connected Erdos-Renyi: start with random spanning tree
        nodes_list = list(range(num_nodes))
        rng.shuffle(nodes_list)
        edges_to_add = []
        for i in range(1, num_nodes):
            parent = rng.choice(nodes_list[:i])
            edges_to_add.append((min(nodes_list[i], parent), max(nodes_list[i], parent)))

        # Add remaining random edges
        remaining = num_edges - len(edges_to_add)
        if remaining > 0:
            existing = set((min(u, v), max(u, v)) for u, v in edges_to_add)
            all_possible = [
                (u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)
                if (u, v) not in existing
            ]
            rng.shuffle(all_possible)
            edges_to_add.extend(all_possible[:remaining])

    # Assign positions using spring layout
    temp_g = nx.Graph()
    temp_g.add_nodes_from(range(num_nodes))
    temp_g.add_edges_from(edges_to_add)
    pos_dict = nx.spring_layout(temp_g, seed=seed)

    for n in g.nodes():
        g.nodes[n]["pos"] = (float(pos_dict[n][0]), float(pos_dict[n][1]))

    # Add edges with physical fiber distances and dynamic routing properties
    for edge_id, (u, v) in enumerate(edges_to_add):
        p_u = np.array(g.nodes[u]["pos"])
        p_v = np.array(g.nodes[v]["pos"])
        eucl_dist = float(np.linalg.norm(p_u - p_v))
        # Scale Euclidean distance to realistic optical distance [400 km .. 2400 km]
        dist_km = round(float(400.0 + eucl_dist * 1500.0), 1)
        base_prop_ms = dist_km / FIBER_PROPAGATION_SPEED_KM_PER_MS

        g.add_edge(
            u,
            v,
            edge_id=edge_id,
            distance_km=dist_km,
            capacity_gbps=default_capacity_gbps,
            base_latency_ms=base_prop_ms,
            traffic_load_gbps=0.0,
            congestion=0.0,
            available_bandwidth=1.0,
            available_bandwidth_gbps=default_capacity_gbps,
            latency=base_prop_ms,
            packet_loss_rate=0.0,
        )

    return g

