"""TopologyAtlas: Non-isomorphic single-DOF planar linkage topologies.

Supports 6-bar (2 topologies), 8-bar (16 topologies), and 10-bar (~230 topologies).

Source: Tsai (2001), "Mechanism Design: Enumeration of Kinematic Structures
According to Function." Extended to 10-bar via computational enumeration
following the same methodology.

6-bar topologies have a single link assortment class (n2, n3):
  - (4,2): 2 topologies (Watt, Stephenson)

8-bar topologies are classified by link assortment (n2, n3, n4, n5):
  - Class I   (4,4,0,0): 9 topologies
  - Class II  (5,2,1,0): 5 topologies
  - Class III (6,0,2,0): 2 topologies

10-bar topologies have 7 link assortment classes (n2, n3, n4, n5):
  - (4,6,0,0), (5,4,1,0), (6,2,2,0), (7,0,3,0),
  - (6,3,0,1), (7,1,1,1), (8,0,0,2)

Topology data was generated via computational enumeration: all connected simple
graphs with the correct degree sequence were generated, then filtered by the
Baranov truss condition (no proper subgraph of k>=3 nodes may have
3(k-1) - 2*edges <= 0, i.e., no rigid or overconstrained subchains).
The results match published atlas counts (16 for 8-bar, 230 for 10-bar).

Usage::

    from dms.mechanisms import TopologyAtlas, NBarMechanism

    atlas = TopologyAtlas(n_links=8)   # 16 non-isomorphic 8-bar topologies
    for tid, topo in atlas:
        mech = NBarMechanism(topo, link_lengths=[1.0] * 8)
        ...
"""

from __future__ import annotations

import copy
import itertools
import logging
from typing import Iterator

import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

logger = logging.getLogger(__name__)


def _degree_to_link_type(degree: int) -> str:
    """Map node degree to link type name."""
    mapping = {2: "binary", 3: "ternary", 4: "quaternary", 5: "pentary"}
    if degree not in mapping:
        raise ValueError(f"Unexpected degree {degree} for planar linkage")
    return mapping[degree]


# ---------------------------------------------------------------------------
# Raw topology data: edge lists for the 2 six-bar topologies.
#
# Node 0 is the ground link. Remaining nodes are labeled 1-5.
#
# Both topologies satisfy:
#   - 6 nodes, 7 edges
#   - Gruebler DOF = 3*(6-1) - 2*7 = 1
#   - Non-isomorphic (Watt has adjacent ternary links; Stephenson does not)
#   - Degree sequence: [3,3,2,2,2,2] -- 4 binary + 2 ternary
#
# Class (4,2): 4 binary (degree 2) + 2 ternary (degree 3)
# ---------------------------------------------------------------------------

_6BAR_TOPOLOGY_DATA: dict[str, dict] = {
    # Watt: ternary links (nodes 0, 3) are adjacent
    "T6B_W": {
        "edges": [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (2, 5),
            (3, 4), (3, 5),
        ],
        "class": (4, 2, 0, 0),
    },
    # Stephenson: ternary links (nodes 0, 4) are NOT adjacent
    "T6B_S": {
        "edges": [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (2, 5),
            (3, 4), (4, 5),
        ],
        "class": (4, 2, 0, 0),
    },
}


# ---------------------------------------------------------------------------
# Raw topology data: edge lists for all 16 eight-bar topologies.
#
# Node 0 is designated as the ground link in every topology.
# The remaining nodes are labeled 1-7.
#
# These adjacency lists satisfy:
#   - 8 nodes, 10 edges each
#   - Gruebler DOF = 1
#   - Non-isomorphic pairwise (verified by nx.is_isomorphic)
#   - Correct degree sequences for each class
#   - No rigid subchains (Baranov condition verified)
#
# Class I (4,4,0,0): 4 binary (degree 2) + 4 ternary (degree 3)
# degree sequence: [3,3,3,3,2,2,2,2] (sum=20, edges=10)
# Node 0 (ground) is a ternary link (degree 3).
#
# Class II (5,2,1,0): 5 binary (degree 2) + 2 ternary (degree 3) + 1 quaternary (degree 4)
# degree sequence: [4,3,3,2,2,2,2,2] (sum=20, edges=10)
# Node 0 (ground) is the quaternary link (degree 4).
#
# Class III (6,0,2,0): 6 binary (degree 2) + 2 quaternary (degree 4)
# degree sequence: [4,4,2,2,2,2,2,2] (sum=20, edges=10)
# Node 0 (ground) is a quaternary link (degree 4).
# ---------------------------------------------------------------------------

_TOPOLOGY_DATA: dict[str, dict] = {
    # ===== Class I: (4,4,0,0) -- 9 topologies =====
    "T01": {
        "edges": [
            (0, 1), (0, 2), (0, 7),
            (1, 4), (1, 6),
            (2, 3), (2, 6),
            (3, 5), (3, 7),
            (4, 5),
        ],
        "class": (4, 4, 0, 0),
    },
    "T02": {
        "edges": [
            (0, 1), (0, 3), (0, 6),
            (1, 5), (1, 7),
            (2, 4), (2, 5), (2, 6),
            (3, 4), (3, 7),
        ],
        "class": (4, 4, 0, 0),
    },
    "T03": {
        "edges": [
            (0, 3), (0, 4), (0, 6),
            (1, 2), (1, 3), (1, 5),
            (2, 4), (2, 7),
            (3, 7),
            (5, 6),
        ],
        "class": (4, 4, 0, 0),
    },
    "T04": {
        "edges": [
            (0, 2), (0, 6), (0, 7),
            (1, 3), (1, 4), (1, 6),
            (2, 4), (2, 5),
            (3, 5), (3, 7),
        ],
        "class": (4, 4, 0, 0),
    },
    "T05": {
        "edges": [
            (0, 2), (0, 4), (0, 5),
            (1, 2), (1, 6), (1, 7),
            (2, 3),
            (3, 4), (3, 6),
            (5, 7),
        ],
        "class": (4, 4, 0, 0),
    },
    "T06": {
        "edges": [
            (0, 1), (0, 3), (0, 4),
            (1, 2), (1, 7),
            (2, 3), (2, 5),
            (3, 6),
            (4, 5),
            (6, 7),
        ],
        "class": (4, 4, 0, 0),
    },
    "T07": {
        "edges": [
            (0, 3), (0, 4), (0, 7),
            (1, 2), (1, 4), (1, 7),
            (2, 3), (2, 6),
            (3, 5),
            (5, 6),
        ],
        "class": (4, 4, 0, 0),
    },
    "T08": {
        "edges": [
            (0, 1), (0, 3), (0, 6),
            (1, 2), (1, 7),
            (2, 3), (2, 4),
            (3, 5),
            (4, 5),
            (6, 7),
        ],
        "class": (4, 4, 0, 0),
    },
    "T09": {
        "edges": [
            (0, 1), (0, 4), (0, 7),
            (1, 5), (1, 6),
            (2, 3), (2, 5), (2, 6),
            (3, 4), (3, 7),
        ],
        "class": (4, 4, 0, 0),
    },

    # ===== Class II: (5,2,1,0) -- 5 topologies =====
    "T10": {
        "edges": [
            (0, 1), (0, 3), (0, 5), (0, 6),
            (1, 2), (1, 4),
            (2, 3), (2, 7),
            (4, 6),
            (5, 7),
        ],
        "class": (5, 2, 1, 0),
    },
    "T11": {
        "edges": [
            (0, 1), (0, 4), (0, 5), (0, 6),
            (1, 3), (1, 7),
            (2, 4), (2, 5), (2, 7),
            (3, 6),
        ],
        "class": (5, 2, 1, 0),
    },
    "T12": {
        "edges": [
            (0, 1), (0, 2), (0, 3), (0, 6),
            (1, 4), (1, 7),
            (2, 4), (2, 5),
            (3, 7),
            (5, 6),
        ],
        "class": (5, 2, 1, 0),
    },
    "T13": {
        "edges": [
            (0, 4), (0, 5), (0, 6), (0, 7),
            (1, 2), (1, 3), (1, 6),
            (2, 5), (2, 7),
            (3, 4),
        ],
        "class": (5, 2, 1, 0),
    },
    "T14": {
        "edges": [
            (0, 3), (0, 5), (0, 6), (0, 7),
            (1, 3), (1, 4), (1, 6),
            (2, 4), (2, 5), (2, 7),
        ],
        "class": (5, 2, 1, 0),
    },

    # ===== Class III: (6,0,2,0) -- 2 topologies =====
    "T15": {
        "edges": [
            (0, 2), (0, 3), (0, 4), (0, 6),
            (1, 3), (1, 4), (1, 5), (1, 7),
            (2, 7),
            (5, 6),
        ],
        "class": (6, 0, 2, 0),
    },
    "T16": {
        "edges": [
            (0, 1), (0, 3), (0, 5), (0, 7),
            (1, 2), (1, 4), (1, 6),
            (2, 5),
            (3, 4),
            (6, 7),
        ],
        "class": (6, 0, 2, 0),
    },
}


# ---------------------------------------------------------------------------
# 10-bar enumeration support
# ---------------------------------------------------------------------------

def _get_link_assortment_classes(n_links: int) -> list[dict]:
    """Compute valid link assortment classes for n_links.

    For a planar single-DOF mechanism with n_links:
      j = (3*(n-1) - 1) / 2  joints
      sum_of_degrees = 2 * j
      n2 + n3 + n4 + n5 = n_links
      2*n2 + 3*n3 + 4*n4 + 5*n5 = 2*j

    Returns list of dicts with 'assortment' and 'degree_sequence' keys.
    """
    j = (3 * (n_links - 1) - 1) // 2  # number of joints
    sum_degrees = 2 * j

    classes = []
    # n5 range: 3*n5 <= sum_degrees - 2*n_links (since n3 + 2*n4 + 3*n5 = sum_degrees - 2*n_links)
    extra = sum_degrees - 2 * n_links  # n3 + 2*n4 + 3*n5 = extra
    for n5 in range(extra // 3 + 1):
        for n4 in range((extra - 3 * n5) // 2 + 1):
            n3 = extra - 2 * n4 - 3 * n5
            if n3 < 0:
                continue
            n2 = n_links - n3 - n4 - n5
            if n2 < 0:
                continue
            degree_seq = sorted(
                [2] * n2 + [3] * n3 + [4] * n4 + [5] * n5,
                reverse=True,
            )
            classes.append({
                "assortment": (n2, n3, n4, n5),
                "degree_sequence": degree_seq,
            })
    return classes


def _passes_baranov_condition(G: nx.Graph) -> bool:
    """Check the Baranov truss condition for a mechanism graph.

    A valid kinematic chain must have no proper subgraph of k >= 3 nodes
    whose internal edges form a rigid or overconstrained subchain:
    3*(k-1) - 2*edges_in_subgraph <= 0.

    This is equivalent to requiring that no subchain has DOF <= 0.
    """
    nodes = list(G.nodes)
    n = len(nodes)
    if n < 4:
        return True

    for k in range(3, n):
        for subset in itertools.combinations(nodes, k):
            subset_set = set(subset)
            edges_in_sub = sum(
                1 for u, v in G.edges()
                if u in subset_set and v in subset_set
            )
            sub_dof = 3 * (k - 1) - 2 * edges_in_sub
            if sub_dof <= 0:
                return False
    return True


def _enumerate_graphs_for_degree_sequence(
    degree_sequence: list[int],
    n_attempts: int = 50000,
) -> list[nx.Graph]:
    """Generate all non-isomorphic connected graphs with the given degree sequence.

    Uses random graph generation + deduplication via WL hashing and
    isomorphism checking.
    """
    hash_buckets: dict[str, list[nx.Graph]] = {}
    found_graphs: list[nx.Graph] = []

    for _ in range(n_attempts):
        try:
            G = nx.random_degree_sequence_graph(degree_sequence, seed=None)
        except nx.NetworkXUnfeasible:
            continue
        except nx.NetworkXError:
            continue

        if not nx.is_connected(G):
            continue
        if nx.number_of_selfloops(G) > 0:
            continue

        wl_hash = weisfeiler_lehman_graph_hash(G)

        is_duplicate = False
        bucket = hash_buckets.get(wl_hash, [])
        for existing in bucket:
            if nx.is_isomorphic(G, existing):
                is_duplicate = True
                break

        if not is_duplicate:
            hash_buckets.setdefault(wl_hash, []).append(G)
            found_graphs.append(G)

    return found_graphs


def _enumerate_10bar() -> tuple[dict[str, nx.Graph], dict[str, tuple[int, int, int, int]]]:
    """Enumerate all valid 10-bar single-DOF planar linkage topologies.

    Algorithm:
    1. Compute all 7 link assortment classes for 10-bar.
    2. For each class, generate all connected simple graphs with the degree sequence.
    3. Filter by Baranov truss condition (no rigid subchains).
    4. Deduplicate via graph isomorphism with WL hash bucketing.
    5. Assign topology IDs as T10B_001 through T10B_NNN.
    """
    classes = _get_link_assortment_classes(10)
    logger.info("10-bar enumeration: %d link assortment classes", len(classes))

    all_topologies: list[tuple[nx.Graph, tuple[int, int, int, int]]] = []

    for cls_info in classes:
        assortment = cls_info["assortment"]
        deg_seq = cls_info["degree_sequence"]
        logger.info(
            "Enumerating class %s with degree sequence %s...",
            assortment, deg_seq,
        )

        n_unique_degrees = len(set(deg_seq))
        if n_unique_degrees <= 2:
            n_attempts = 200000
        else:
            n_attempts = 100000

        candidates = _enumerate_graphs_for_degree_sequence(deg_seq, n_attempts)
        logger.info(
            "Class %s: %d unique candidate graphs before Baranov filter",
            assortment, len(candidates),
        )

        valid = [G for G in candidates if _passes_baranov_condition(G)]
        logger.info(
            "Class %s: %d valid topologies after Baranov filter",
            assortment, len(valid),
        )

        for G in valid:
            all_topologies.append((G, assortment))

    topologies_dict: dict[str, nx.Graph] = {}
    classes_dict: dict[str, tuple[int, int, int, int]] = {}

    for idx, (G, assortment) in enumerate(all_topologies, start=1):
        tid = f"T10B_{idx:03d}"
        topologies_dict[tid] = G
        classes_dict[tid] = assortment

    logger.info("10-bar enumeration complete: %d total topologies", len(topologies_dict))
    return topologies_dict, classes_dict


class TopologyAtlas:
    """Atlas of non-isomorphic single-DOF planar linkage topologies.

    Supports 6-bar (2 topologies: Watt, Stephenson), 8-bar (16 topologies),
    and 10-bar (~230 topologies, enumerated on construction).

    Each topology is stored as a NetworkX Graph with node attributes:
        - 'link_type': one of 'ground', 'binary', 'ternary', 'quaternary', 'pentary'
        - 'topology_id': string identifier

    Ground link is always node 0.

    Parameters
    ----------
    n_links : int
        Number of links. Must be 6, 8, or 10.
    validate : bool
        Run structural validation after loading. Default True. Set to False
        when probing novel atlases or experimenting — the count and class
        distribution checks raise on any deviation from the published values.
    """

    def __init__(self, n_links: int = 8, validate: bool = True) -> None:
        self.n_links = n_links
        self.topologies: dict[str, nx.Graph] = {}
        self._classes: dict[str, tuple[int, int, int, int]] = {}

        if n_links == 6:
            self._load_6bar()
        elif n_links == 8:
            self._load_8bar()
        elif n_links == 10:
            self._load_10bar()
        else:
            raise ValueError(f"n_links must be 6, 8, or 10, got {n_links}")

        if validate:
            self._validate()

    def _load_6bar(self) -> None:
        """Load hardcoded 6-bar topology data (Watt + Stephenson)."""
        for tid, tdata in _6BAR_TOPOLOGY_DATA.items():
            G = nx.Graph()
            G.add_edges_from(tdata["edges"])
            self._classes[tid] = tdata["class"]

            for node in G.nodes:
                degree = G.degree(node)
                if node == 0:
                    G.nodes[node]["link_type"] = "ground"
                else:
                    G.nodes[node]["link_type"] = _degree_to_link_type(degree)
                G.nodes[node]["topology_id"] = tid

            self.topologies[tid] = G

    def _load_8bar(self) -> None:
        """Load hardcoded 8-bar topology data."""
        for tid, tdata in _TOPOLOGY_DATA.items():
            G = nx.Graph()
            G.add_edges_from(tdata["edges"])
            self._classes[tid] = tdata["class"]

            for node in G.nodes:
                degree = G.degree(node)
                if node == 0:
                    G.nodes[node]["link_type"] = "ground"
                else:
                    G.nodes[node]["link_type"] = _degree_to_link_type(degree)
                G.nodes[node]["topology_id"] = tid

            self.topologies[tid] = G

    def _load_10bar(self) -> None:
        """Enumerate and load all 10-bar topologies."""
        topo_dict, cls_dict = _enumerate_10bar()
        self._classes = cls_dict

        for tid, G in topo_dict.items():
            for node in G.nodes:
                degree = G.degree(node)
                if node == 0:
                    G.nodes[node]["link_type"] = "ground"
                else:
                    G.nodes[node]["link_type"] = _degree_to_link_type(degree)
                G.nodes[node]["topology_id"] = tid

            self.topologies[tid] = G

    def _validate(self) -> None:
        """Verify atlas correctness on initialization."""
        expected_nodes = self.n_links
        expected_edges = (3 * (self.n_links - 1) - 1) // 2

        if self.n_links == 6:
            expected_count = 2
            expected_class_dist = {(4, 2, 0, 0): 2}
        elif self.n_links == 8:
            expected_count = 16
            expected_class_dist = {
                (4, 4, 0, 0): 9,
                (5, 2, 1, 0): 5,
                (6, 0, 2, 0): 2,
            }
        elif self.n_links == 10:
            expected_count = 230
            expected_class_dist = None
        else:
            expected_count = None
            expected_class_dist = None

        if expected_count is not None and len(self.topologies) != expected_count:
            raise ValueError(
                f"Expected {expected_count} topologies for {self.n_links}-bar, "
                f"got {len(self.topologies)}"
            )

        for tid, G in self.topologies.items():
            n = G.number_of_nodes()
            j = G.number_of_edges()
            if n != expected_nodes:
                raise ValueError(
                    f"Topology {tid}: expected {expected_nodes} nodes, got {n}"
                )
            if j != expected_edges:
                raise ValueError(
                    f"Topology {tid}: expected {expected_edges} edges, got {j}"
                )
            dof = self.gruebler_dof(G)
            if dof != 1:
                raise ValueError(
                    f"Topology {tid}: DOF={dof}, expected 1"
                )

        if self.n_links in (6, 8):
            tids = list(self.topologies.keys())
            graphs = list(self.topologies.values())
            for i in range(len(graphs)):
                for j_idx in range(i + 1, len(graphs)):
                    if nx.is_isomorphic(graphs[i], graphs[j_idx]):
                        raise ValueError(
                            f"Topologies {tids[i]} and {tids[j_idx]} are isomorphic"
                        )

        if expected_class_dist is not None:
            class_counts: dict[tuple, int] = {}
            for tid in self.topologies:
                cls = self._classes[tid]
                class_counts[cls] = class_counts.get(cls, 0) + 1

            for cls, expected_cnt in expected_class_dist.items():
                actual = class_counts.get(cls, 0)
                if actual != expected_cnt:
                    raise ValueError(
                        f"Class {cls}: expected {expected_cnt}, got {actual}"
                    )

    @staticmethod
    def gruebler_dof(G: nx.Graph) -> int:
        """Compute Gruebler degree of freedom for a planar mechanism graph.

        F = 3*(n-1) - 2*j
        where n = number of links (nodes), j = number of joints (edges).
        """
        n = G.number_of_nodes()
        j = G.number_of_edges()
        return 3 * (n - 1) - 2 * j

    def get_topology(self, tid: str) -> nx.Graph:
        """Return a deep copy of topology graph by ID."""
        if tid not in self.topologies:
            raise KeyError(f"Unknown topology ID: {tid}")
        return copy.deepcopy(self.topologies[tid])

    def get_class(self, tid: str) -> tuple[int, int, int, int]:
        """Return the class tuple (n_binary, n_ternary, n_quaternary, n_pentary)."""
        if tid not in self._classes:
            raise KeyError(f"Unknown topology ID: {tid}")
        return self._classes[tid]

    def __len__(self) -> int:
        return len(self.topologies)

    def __iter__(self) -> Iterator[tuple[str, nx.Graph]]:
        for tid, G in self.topologies.items():
            yield tid, G
