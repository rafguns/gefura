import itertools

import networkx as nx
import pytest

from gefura import global_gefura, local_gefura

# Format: (list of edges, node to gefura value dict, kwargs, ID)
# We follow the convention that nodes that start with the same character
# belong to the same group.
global_gefura_data = [
    pytest.param(
        [
            ("a1", "a2"),
            ("a1", "a3"),
            ("a2", "a3"),
            ("a3", "b1"),
            ("a2", "b2"),
            ("b1", "b2"),
            ("b2", "c1"),
            ("b1", "c3"),
            ("b2", "c2"),
            ("c2", "c3"),
        ],
        {
            "a1": 0,
            "a2": 13 / 48,
            "a3": 17 / 96,
            "b1": 29 / 90,
            "b2": 5 / 9,
            "c1": 0,
            "c2": 5 / 96,
            "c3": 5 / 96,
        },
        {},
        id="3 groups",
    ),
    pytest.param(
        [
            ("a1", "b1"),
            ("b1", "b2"),
            ("b2", "c1"),
            ("c1", "c2"),
            ("c2", "b3"),
            ("b3", "a2"),
        ],
        {
            "a1": 0,
            "a2": 0,
            "b1": 4 / 12,
            "b2": 6 / 12,
            "b3": 4 / 12,
            "c1": 6 / 11,
            "c2": 5 / 11,
        },
        {},
        id="line graph, 3 groups",
    ),
    pytest.param(
        [("b1", "a1"), ("a1", "a2"), ("a1", "b2"), ("a2", "a3"), ("a3", "b2")],
        {"a1": 2.5, "a2": 0.5, "a3": 0.5, "b1": 0, "b2": 0.5},
        {"normalized": False},
        id="2 groups, unnormalized",
    ),
    pytest.param(
        [("a1", "a2"), ("a2", "b1"), ("b1", "b2"), ("b2", "b3")],
        {"a1": 0, "a2": 1, "b1": 1, "b2": 0.5, "b3": 0},
        {},
        id="line graph, 2 groups",
    ),
]


def group_nodes_by_first_char(G):
    return [set(grp[1]) for grp in itertools.groupby(sorted(G), key=lambda x: x[0])]


@pytest.mark.parametrize(("edges", "expected", "kwargs"), global_gefura_data)
def test_global_gefura(edges, expected, kwargs):
    G = nx.Graph()
    G.add_edges_from(edges)
    groups = group_nodes_by_first_char(G)

    assert global_gefura(G, groups, **kwargs) == pytest.approx(expected)


@pytest.fixture()
def graph_with_singleton_groups():
    G = nx.Graph()
    G.add_edge(1, 2)

    return G, [{1}, {2}], {1: 0.0, 2: 0.0}


def test_singleton_groups_global(graph_with_singleton_groups):
    G, groups, expected = graph_with_singleton_groups

    assert global_gefura(G, groups, normalized=False) == pytest.approx(expected)
    # Normalization should not throw ZeroDivisionError
    assert global_gefura(G, groups) == pytest.approx(expected)


def test_singleton_groups_local(graph_with_singleton_groups):
    G, groups, expected = graph_with_singleton_groups

    assert local_gefura(G, groups, normalized=False) == pytest.approx(expected)
    # Normalization should not throw ZeroDivisionError
    assert local_gefura(G, groups) == pytest.approx(expected)


@pytest.fixture()
def digraph():
    edges = [
        ("a1", "a2"),
        ("a1", "b2"),
        ("a2", "a1"),
        ("a2", "b1"),
        ("b1", "a1"),
        ("b1", "b2"),
    ]
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def test_digraph_global(digraph):
    groups = group_nodes_by_first_char(digraph)

    known_vals_unnormalized = {"a1": 1.5, "a2": 1, "b1": 0.5, "b2": 0}
    known_vals_normalized = {"a1": 0.375, "a2": 0.25, "b1": 0.125, "b2": 0}

    gamma = global_gefura(digraph, groups, normalized=False)
    assert gamma == pytest.approx(known_vals_unnormalized)
    assert global_gefura(digraph, groups) == pytest.approx(known_vals_normalized)


def test_digraph_local(digraph):
    groups = group_nodes_by_first_char(digraph)

    known_vals_unnormalized_out = {"a1": 0.5, "a2": 1, "b1": 0, "b2": 0}
    known_vals_unnormalized_in = {"a1": 1, "a2": 0, "b1": 0.5, "b2": 0}
    known_vals_unnormalized_all = {"a1": 1.5, "a2": 1, "b1": 0.5, "b2": 0}
    known_vals_normalized_out = {"a1": 0.25, "a2": 0.5, "b1": 0, "b2": 0}
    known_vals_normalized_in = {"a1": 0.5, "a2": 0, "b1": 0.25, "b2": 0}
    known_vals_normalized_all = {"a1": 0.375, "a2": 0.25, "b1": 0.125, "b2": 0}

    gamma_out = local_gefura(digraph, groups, normalized=False)
    assert gamma_out == pytest.approx(known_vals_unnormalized_out)
    gamma_in = local_gefura(digraph, groups, normalized=False, direction="in")
    assert gamma_in == pytest.approx(known_vals_unnormalized_in)
    gamma_all = local_gefura(digraph, groups, normalized=False, direction="all")
    assert gamma_all == pytest.approx(known_vals_unnormalized_all)

    gamma_out = local_gefura(digraph, groups)
    assert gamma_out == pytest.approx(known_vals_normalized_out)
    gamma_in = local_gefura(digraph, groups, direction="in")
    assert gamma_in == pytest.approx(known_vals_normalized_in)
    gamma_all = local_gefura(digraph, groups, direction="all")
    assert gamma_all == pytest.approx(known_vals_normalized_all)


@pytest.fixture()
def weighted_graph():
    edges = [
        ("a1", "a2", 1),
        ("a2", "b2", 3),
        ("a1", "b1", 1),
        ("b1", "b3", 2),
        ("b2", "b3", 1),
    ]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G


def test_weighted_graph_global(weighted_graph):
    groups = group_nodes_by_first_char(weighted_graph)

    known_vals = {"a1": 0.5, "a2": 1 / 6, "b1": 0.5, "b2": 0.125, "b3": 0.125}
    gamma = global_gefura(weighted_graph, groups, weight="weight")
    assert gamma == pytest.approx(known_vals)


def test_global_ignore_weights(weighted_graph):
    groups = group_nodes_by_first_char(weighted_graph)

    known_vals = {"a1": 1 / 3, "a2": 1 / 3, "b1": 0.25, "b2": 0.25, "b3": 0}
    gamma = global_gefura(weighted_graph, groups)
    assert gamma == pytest.approx(known_vals)


def test_weighted_grlocal(weighted_graph):
    groups = group_nodes_by_first_char(weighted_graph)

    known_vals = {"a1": 1.5, "a2": 0.5, "b1": 2, "b2": 0.5, "b3": 0.5}
    gamma = local_gefura(weighted_graph, groups, weight="weight", normalized=False)
    assert gamma == pytest.approx(known_vals)


@pytest.fixture()
def small_graph_3_groups():
    edges = [
        ("a1", "b1"),
        ("a2", "b1"),
        ("b1", "b2"),
        ("b2", "c1"),
        ("b2", "c2"),
        ("a1", "c1"),
    ]
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def test_local_normalized(small_graph_3_groups):
    groups = group_nodes_by_first_char(small_graph_3_groups)

    known_gamma = {
        "a1": 0.125,
        "a2": 0,
        "b1": 0.375,
        "b2": 0.375,
        "c1": 0.125,
        "c2": 0,
    }
    gamma = local_gefura(small_graph_3_groups, groups)
    assert gamma == pytest.approx(known_gamma)


def test_unnormalized(small_graph_3_groups):
    groups = group_nodes_by_first_char(small_graph_3_groups)

    known_gamma = {"a1": 0.5, "a2": 0, "b1": 1.5, "b2": 1.5, "c1": 0.5, "c2": 0}
    gamma = local_gefura(small_graph_3_groups, groups, normalized=False)
    assert gamma == pytest.approx(known_gamma)


def test_local_line_graph():
    edges = [
        ("a3", "a2"),
        ("a2", "c1"),
        ("c1", "b1"),
        ("b1", "a1"),
        ("a1", "b2"),
        ("b2", "b3"),
    ]
    groups = [{"a1", "a2", "a3"}, {"b1", "b2", "b3"}, {"c1"}]
    known_vals = {"a1": 4, "a2": 4, "a3": 0, "b1": 6, "b2": 4, "b3": 0, "c1": 0}
    G = nx.Graph()
    G.add_edges_from(edges)

    assert local_gefura(G, groups, normalized=False) == pytest.approx(known_vals)


wrong_node_set_msg = "Nodes in G and nodes in groups.*"


def test_wrong_node_set_global():
    with pytest.raises(ValueError, match=wrong_node_set_msg):
        global_gefura(nx.Graph(), [{1}])


def test_wrong_node_set_local():
    with pytest.raises(ValueError, match=wrong_node_set_msg):
        local_gefura(nx.Graph(), [{1}])


def test_local_directed_wrong_direction_value():
    with pytest.raises(ValueError, match="Direction should be.*"):
        local_gefura(nx.DiGraph(), [], direction="foobar")


def test_local_directed():
    edges = [
        ("a1", "a2"),
        ("a1", "b1"),
        ("a2", "a1"),
        ("a2", "a3"),
        ("b1", "b2"),
        ("b1", "c1"),
        ("c1", "a1"),
        ("c2", "c1"),
    ]
    groups = [{"a1", "a2", "a3"}, {"b1", "b2"}, {"c1", "c2"}]
    known_out = {"a1": 0.375, "a2": 0, "a3": 0, "b1": 0, "b2": 0, "c1": 1, "c2": 0}
    known_in = {"a1": 0.75, "a2": 0.375, "a3": 0, "b1": 0.8, "b2": 0, "c1": 0, "c2": 0}
    known_all = {
        "a1": 9 / 16,
        "a2": 3 / 16,
        "a3": 0,
        "b1": 0.4,
        "b2": 0,
        "c1": 0.5,
        "c2": 0,
    }
    G = nx.DiGraph()
    G.add_edges_from(edges)

    for d, vals in (("out", known_out), ("in", known_in), ("all", known_all)):
        assert local_gefura(G, groups, direction=d) == pytest.approx(vals)


def test_overlap_simple():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    groups = [{1, 2}, {2, 3}]
    known = {1: 0, 2: 1, 3: 0}
    assert global_gefura(G, groups) == pytest.approx(known)


@pytest.fixture()
def overlapping_line_graph():
    edges = [(1, 2), (2, 3), (3, 4)]
    groups = [{1, 2, 3}, {2, 3, 4}, {4}]
    G = nx.Graph()
    G.add_edges_from(edges)
    return G, groups


def test_overlapping_line_graph_unnormalized(overlapping_line_graph):
    G, groups = overlapping_line_graph

    known = {1: 0, 2: 3, 3: 5, 4: 0}
    assert global_gefura(G, groups, normalized=False) == pytest.approx(known)


def test_overlapping_line_graph_normalized(overlapping_line_graph):
    G, groups = overlapping_line_graph

    known = {1: 0, 2: 0.5, 3: 5 / 6, 4: 0}
    assert global_gefura(G, groups) == pytest.approx(known)


@pytest.fixture()
def overlapping_graph():
    edges = [
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 6),
        (3, 5),
        (4, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 8),
    ]
    groups = [{1, 2, 3}, {2, 3, 4, 5, 6}, {4, 7, 8}]
    G = nx.Graph()
    G.add_edges_from(edges)
    return G, groups


def test_overlapping_graph_unnormalized(overlapping_graph):
    G, groups = overlapping_graph
    known = {
        1: 0,
        2: 19 / 6,
        3: 20 / 3,
        4: 0,
        5: 53 / 3,
        6: 26 / 3,
        7: 5 / 3,
        8: 7 / 6,
    }
    assert global_gefura(G, groups, normalized=False) == pytest.approx(known)


def test_overlapping_graph_normalized(overlapping_graph):
    G, groups = overlapping_graph
    known = {
        1: 0,
        2: 19 / 144,
        3: 5 / 18,
        4: 0,
        5: 53 / 90,
        6: 26 / 90,
        7: 5 / 84,
        8: 7 / 168,
    }
    assert global_gefura(G, groups) == pytest.approx(known)
