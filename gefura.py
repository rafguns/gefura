"""Gefura measures for 'bridges' between node groups in a network

These measures are adaptations of betweenness centrality where only shortest
paths between nodes from different groups are taken into account. The gefura
measures thus gauge the extent to which a node bridges between groups (old
Greek γεφυρα = bridge).

This module implements global and local gefura. Both directed and
undirected, as well as weighted and unweighted networks are supported.
Overlapping groups are currently only supported for global gefura.

"""
from collections import defaultdict, deque
from itertools import combinations
from typing import Iterable, Literal, Optional, Union

import networkx as nx
from networkx.algorithms.centrality.betweenness import (
    _single_source_dijkstra_path_basic,
)

__version__ = "0.2"
__all__ = ["global_gefura", "local_gefura"]

Node = Union[str, int]


def _groups_per_node(groups: Iterable[set[Node]]) -> dict[Node, set[int]]:
    """Make mapping from a node to its group(s)"""
    d = defaultdict(set)
    for i, group in enumerate(groups):
        for n in group:
            d[n].add(i)
    return d


def _single_source_shortest_path_basic(G, s, max_path_length=None):
    """Adapted from networkx.algorithms.centrality.betweenness

    Includes optional maximum path length

    """
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)
    dist = {}
    sigma[s] = 1.0
    dist[s] = 0
    Q = deque([s])
    while Q:  # use BFS to find shortest paths
        v = Q.popleft()
        S.append(v)
        distv = dist[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in dist:
                dist[w] = distv + 1
                if max_path_length is None or dist[w] <= max_path_length:
                    Q.append(w)
            if dist[w] == distv + 1:  # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma, dist


def global_gefura(
    G: nx.Graph,
    groups: Iterable[set[Node]],
    *,
    weight: Optional[str] = None,
    normalized: bool = True,
    max_path_length: Optional[int] = None,
) -> dict[Node, float]:
    """Determine global gefura measure of each node

    This function handles both weighted and unweighted networks, directed and
    undirected, and connected and unconnected. Node groups may overlap.

    Arguments
    ---------
    G : a networkx.Graph
        the network

    groups : a list or iterable of sets
        Each set represents a group and contains 1 to N nodes

    weight : None or a string
        If None, the network is treated as unweighted. If a string, this is
        the edge data key corresponding to the edge weight

    normalized : True|False
        Whether or not to normalize the output to [0, 1].

    max_path_length : int or None
        Maximum number of edges to traverse (unweighted only)

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)
    >>> groups = [{0, 2}, {1}, {3, 4}]
    >>> global_gefura(G, groups)
    {0: 0.0, 1: 0.5, 2: 0.8, 3: 0.6, 4: 0.0}

    """
    gamma = dict.fromkeys(G, 0)
    group_of = _groups_per_node(groups)
    if set(group_of) != set(G):
        msg = "Nodes in G and nodes in groups should be the same!"
        raise ValueError(msg)

    for s in G:
        if weight is None:
            S, P, sigma, _ = _single_source_shortest_path_basic(G, s, max_path_length)
        else:
            S, P, sigma, _ = _single_source_dijkstra_path_basic(G, s, weight)

        # Accumulation
        delta = dict.fromkeys(G, 0)
        s_groups = group_of[s]
        while S:
            w = S.pop()
            w_groups = group_of[w]
            # We count one path i times, if it functions as a path between i
            # different pairs of groups
            i = len(s_groups) * len(w_groups) - len(s_groups & w_groups)
            deltaw = delta[w]
            coeff = (i + deltaw) / sigma[w]
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                gamma[w] += deltaw

    return rescale_global(gamma, G, groups, normalized=normalized)


def _local_gefura(
    G: nx.Graph,
    groups: Iterable[set[Node]],
    *,
    weight: Optional[str] = None,
    normalized: bool = True,
    max_path_length: Optional[int] = None,
) -> dict[Node, float]:
    gamma = dict.fromkeys(G, 0)
    group_of = _groups_per_node(groups)
    if set(group_of) != set(G):
        msg = "Nodes in G and nodes in groups should be the same!"
        raise ValueError(msg)

    for s in G:
        if weight is None:
            S, P, sigma, _ = _single_source_shortest_path_basic(
                G, s, max_path_length=max_path_length
            )
        else:
            S, P, sigma, _ = _single_source_dijkstra_path_basic(G, s, weight)

        # Accumulation
        delta = dict.fromkeys(G, 0)
        s_groups = group_of[s]
        while S:
            w = S.pop()
            w_groups = group_of[w]
            # We count one path i times, if it functions as a path between i
            # different pairs of groups
            i = len(s_groups) * len(w_groups) - len(s_groups & w_groups)
            deltaw, sigmaw = delta[w], sigma[w]
            coeff = (i + deltaw) / sigmaw
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s and i == 0:
                gamma[w] += delta[w]

    return rescale_local(gamma, G, groups, normalized=normalized)


def local_gefura(
    G: nx.Graph,
    groups: Iterable[set[Node]],
    *,
    weight: Optional[str] = None,
    normalized: bool = True,
    direction: Literal["in", "out", "all"] = "out",
    max_path_length: Optional[int] = None,
) -> dict[Node, float]:
    """Determine local gefura measure of each node

    This function handles both weighted and unweighted networks, directed and
    undirected, and connected and unconnected.

    Arguments
    ---------
    G : a networkx.Graph
        the network

    groups : a list or iterable of sets
        Each set represents a group and contains 1 to N nodes

    weight : None or a string
        If None, the network is treated as unweighted. If a string, this is
        the edge data key corresponding to the edge weight

    normalized : True|False
        Whether or not to normalize the output to [0, 1].

    direction : 'in'|'out'|'all'
        Only applicable in the case of a directed network, otherwise this
        parameter is ignored.
        - 'all' is based on all paths between the own group and the rest
        - 'in' is based on all paths from elsewhere to the own group
        - 'out' is based on all paths from the own group to elsewhere

    max_path_length : int or None
        Maximum number of edges to traverse (unweighted only)

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)
    >>> groups = [{0, 2}, {1}, {3, 4}]
    >>> local_gefura(G, groups)
    {0: 0.0, 1: 0, 2: 0.6666666666666666, 3: 1.0, 4: 0.0}

    """
    kwargs = {
        "weight": weight,
        "normalized": normalized,
        "max_path_length": max_path_length,
    }
    if not G.is_directed() or direction == "out":
        return _local_gefura(G, groups, **kwargs)

    if direction not in ("in", "all"):
        msg = "Direction should be either 'in', 'out' or 'all'."
        raise ValueError(msg)

    # The algorithm follows the 'out' direction in directed graphs.
    # For 'in', we reverse the direction prior to applying the algorithm.
    gamma_in = _local_gefura(G.reverse(copy=False), groups, **kwargs)
    if direction == "in":
        return gamma_in

    # 'all' is the sum of 'in' and 'out'
    gamma_out = _local_gefura(G, groups, **kwargs)
    norm = 2 if normalized else 1  # Count both A -> gamma and gamma -> A

    return {k: (gamma_in[k] + gamma_out[k]) / norm for k in gamma_in}


def rescale_global(
    gamma: dict[Node, float],
    G: nx.Graph,
    groups: Iterable[set[Node]],
    *,
    normalized: bool,
) -> dict[Node, float]:
    # Since all shortest paths are counted twice if undirected, we divide by 2.
    # Only do this in the unnormalized case. If normalized, we need to account
    # for both group A -> B and B -> A.
    base_factor = 1 if G.is_directed() and not normalized else 2

    for s in G:
        if normalized:
            # All combinations of 2 groups
            group_combinations = list(combinations(groups, 2))
            ss = {s}
            factor = (
                sum(
                    len(A - ss) * len(B - ss) - len(A & B - ss)
                    for A, B in group_combinations
                )
                * base_factor
            )
        else:
            factor = base_factor
        try:
            gamma[s] /= factor
        except ZeroDivisionError:
            gamma[s] = 0

    return gamma


def rescale_local(
    gamma: dict[Node, float],
    G: nx.Graph,
    groups: Iterable[set[Node]],
    *,
    normalized: bool,
) -> dict[Node, float]:
    if normalized:
        for s in G:
            own_group_size = next(len(group) for group in groups if s in group)
            factor = (own_group_size - 1) * (len(G) - own_group_size)
            try:
                gamma[s] /= factor
            except ZeroDivisionError:
                gamma[s] = 0
    return gamma
