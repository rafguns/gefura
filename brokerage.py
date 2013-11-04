"""Brokerage measures for 'bridges' between node groups in a network

These measures are adaptations of betweenness centrality where only shortest
paths between nodes from different groups are taken into account. The brokerage
measures are also known as 'Q-measures' (not to be mistaken with e.g.
modularity Q).

This module implements global and local brokerage. Both directed and
undirected, as well as weighted and unweighted networks are supported.

"""
from __future__ import division

from collections import defaultdict
from itertools import combinations
from networkx.algorithms.centrality.betweenness import \
    _single_source_shortest_path_basic, _single_source_dijkstra_path_basic


__all__ = ["global_brokerage", "local_brokerage"]


def global_brokerage(G, groups, weight=None, normalized=True):
    """Determine global brokerage measure of each node

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

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)
    >>> groups = [{0, 2}, {1}, {3, 4}]
    >>> brokerage.global_brokerage(G, groups)
    {0: 0.0, 1: 0.5, 2: 0.8, 3: 0.6, 4: 0.0}

    """
    BG = dict.fromkeys(G, 0)
    # Make mapping node -> groups.
    group_of = defaultdict(set)
    for i, group in enumerate(groups):
        for n in group:
            group_of[n].add(i)

    for s in G:
        if weight is None:
            S, P, sigma = _single_source_shortest_path_basic(G, s)
        else:
            S, P, sigma = _single_source_dijkstra_path_basic(G, s, weight)

        # Accumulation
        delta = dict.fromkeys(G, 0)
        s_groups = group_of[s]
        while S:
            w = S.pop()
            w_groups = group_of[w]
            # We count one path i times, if it functions as a path between i
            # different pairs of groups
            i = len(s_groups) * len(w_groups) - len(s_groups & w_groups)
            for v in P[w]:
                sigmas = sigma[v] / sigma[w]
                delta[v] += sigmas * (i + delta[w])
            if w != s:
                BG[w] += delta[w]

    # Since all shortest paths are counted twice if undirected, we divide by 2.
    if not G.is_directed():
        for s in G:
            BG[s] /= 2

    # Normalize
    if normalized:
        # All combinations of 2 groups
        group_combinations = list(combinations(groups, 2))
        for s in G:
            factor = sum(len(A - {s}) * len(B - {s}) - len(A & B - {s})
                         for A, B in group_combinations)
            if G.is_directed():
                # Count both A -> B and B -> A
                factor *= 2
            try:
                BG[s] /= factor
            except ZeroDivisionError:
                BG[s] = 0

    return BG


def _local_brokerage(G, groups, weight=None, normalized=True):
    BL = dict.fromkeys(G, 0)
    # Make mapping node -> group.
    # This assumes that groups are disjoint.
    group_of = {n: group for group in groups for n in group}

    for s in G:
        if weight is None:
            S, P, sigma = _single_source_shortest_path_basic(G, s)
        else:
            S, P, sigma = _single_source_dijkstra_path_basic(G, s, weight)

        # Accumulation
        delta = dict.fromkeys(G, 0)
        while S:
            w = S.pop()
            different_groups = group_of[s] != group_of[w]
            for v in P[w]:
                sigmas = sigma[v] / sigma[w]
                i = 1 if different_groups else 0
                delta[v] += sigmas * (i + delta[w])
            if w != s and not different_groups:
                BL[w] += delta[w]

    # Normalize
    if normalized:
        # All combinations of 2 groups
        for s in G:
            own_group_size = len(group_of[s])
            factor = (own_group_size - 1) * (len(G) - own_group_size)

            try:
                BL[s] /= factor
            except ZeroDivisionError:
                BL[s] = 0

    return BL


def local_brokerage(G, groups, weight=None, normalized=True, direction='out'):
    """Determine local brokerage measure of each node

    This function handles both weighted and unweighted networks, directed and
    undirected, and connected and unconnected. It supposes, however, that node
    groups are disjoint, i.e. that groups cannot overlap.

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

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)
    >>> groups = [{0, 2}, {1}, {3, 4}]
    >>> brokerage.local_brokerage(G, groups)
    {0: 0.0, 1: 0, 2: 0.6666666666666666, 3: 1.0, 4: 0.0}

    """
    if not G.is_directed() or direction == 'out':
        return _local_brokerage(G, groups, weight, normalized)

    if direction not in ('in', 'all'):
        raise ValueError("direction should be either 'in', 'out' or 'all'.")

    H = G.reverse()
    BL_in = _local_brokerage(H, groups, weight, normalized)
    if direction == 'in':
        return BL_in
    else:
        # 'all' is the sum of 'in' and 'out'
        BL_out = _local_brokerage(G, groups, weight, normalized)
        norm = 2 if normalized else 1  # Count both A -> B and B -> A
        return {k: (BL_in[k] + BL_out[k]) / norm for k in BL_in}
