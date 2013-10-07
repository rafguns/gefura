"""Brokerage measures for 'bridges' between node groups in a network

These measures are adaptations of betweenness centrality where only shortest
paths between nodes from different groups are taken into account. They were
previously known as `Q-measures'. Because that term is overloaded (e.g., Q
modularity), we now simply refer to them as brokerage measures.

For the moment, this module only implements global brokerage. Local and
external brokerage may be added later on.

"""
from __future__ import division

from itertools import combinations
from networkx.algorithms.centrality.betweenness import \
    _single_source_shortest_path_basic, _single_source_dijkstra_path_basic


__all__ = ["global_brokerage"]


def global_brokerage(G, groups, weight=None, normalized=True):
    """Determine global brokerage measure of each node

    Currently this function handles both weighted and unweighted networks,
    directed and undirected, and connected and unconnected. It supposes,
    however, that node groups are disjoint, i.e. that groups cannot overlap.

    Arguments
    ---------
    G : a networkx.Graph

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
            i = 1 if group_of[s] != group_of[w] else 0
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
            factor = sum(len(A - {s}) * len(BG - {s})
                         for A, BG in group_combinations)
            if G.is_directed():
                # Count both A -> BG and BG -> A
                factor *= 2
            try:
                BG[s] /= factor
            except ZeroDivisionError:
                BG[s] = 0

    return BG
