# -*- coding: utf-8 -*-
"""Gefura measures for 'bridges' between node groups in a network

These measures are adaptations of betweenness centrality where only shortest
paths between nodes from different groups are taken into account. The gefura
measures thus gauge the extent to which a node bridges between groups (old
Greek γεφυρα = bridge).

This module implements global and local gefura. Both directed and
undirected, as well as weighted and unweighted networks are supported.

"""
from itertools import combinations
from networkx.algorithms.centrality.betweenness import (
    _single_source_shortest_path_basic,
    _single_source_dijkstra_path_basic,
)


__all__ = ["global_gefura", "local_gefura"]


def global_gefura(G, groups, weight=None, normalized=True):
    """Determine global gefura measure of each node

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

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)
    >>> groups = [{0, 2}, {1}, {3, 4}]
    >>> global_gefura(G, groups)
    {0: 0.0, 1: 0.5, 2: 0.8, 3: 0.6, 4: 0.0}

    """
    gamma = dict.fromkeys(G, 0)
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
            deltaw, sigmaw = delta[w], sigma[w]
            coeff = (
                (1 + deltaw) / sigmaw if group_of[s] != group_of[w] else deltaw / sigmaw
            )
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                gamma[w] += deltaw

    gamma = rescale_global(gamma, G, groups, normalized)

    return gamma


def _local_gefura(G, groups, weight=None, normalized=True):
    gamma = dict.fromkeys(G, 0)
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
            deltaw, sigmaw = delta[w], sigma[w]
            coeff = (1 + deltaw) / sigmaw if different_groups else deltaw / sigmaw
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s and not different_groups:
                gamma[w] += deltaw

    gamma = rescale_local(gamma, G, group_of, normalized)
    return gamma


def local_gefura(G, groups, weight=None, normalized=True, direction="out"):
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

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)
    >>> groups = [{0, 2}, {1}, {3, 4}]
    >>> local_gefura(G, groups)
    {0: 0.0, 1: 0, 2: 0.6666666666666666, 3: 1.0, 4: 0.0}

    """
    if not G.is_directed() or direction == "out":
        return _local_gefura(G, groups, weight, normalized)

    if direction not in ("in", "all"):
        raise ValueError("direction should be either 'in', 'out' or 'all'.")

    gamma_in = _local_gefura(G.reverse(copy=False), groups, weight, normalized)
    if direction == "in":
        return gamma_in
    else:
        # 'all' is the sum of 'in' and 'out'
        gamma_out = _local_gefura(G, groups, weight, normalized)
        norm = 2 if normalized else 1  # Count both A -> gamma and gamma -> A
        print(gamma_in, gamma_out)
        return {k: (gamma_in[k] + gamma_out[k]) / norm for k in gamma_in}


def rescale_global(gamma, G, groups, normalized):
    # Since all shortest paths are counted twice if undirected, we divide by 2.
    # Only do this in the unnormalized case. If normalized, we need to account
    # for both group A -> B and B -> A.
    base_factor = 1 if G.is_directed() and not normalized else 2

    for s in G:
        if normalized:
            # All combinations of 2 groups
            group_combinations = list(combinations(groups, 2))
            factor = (
                sum(len(A - {s}) * len(B - {s}) for A, B in group_combinations)
                * base_factor
            )
        else:
            factor = base_factor
        try:
            gamma[s] /= factor
        except ZeroDivisionError:
            gamma[s] = 0

    return gamma


def rescale_local(gamma, G, group_of, normalized):
    if normalized:
        for s in G:
            own_group_size = len(group_of[s])
            factor = (own_group_size - 1) * (len(G) - own_group_size)
            try:
                gamma[s] /= factor
            except ZeroDivisionError:
                gamma[s] = 0
    return gamma
