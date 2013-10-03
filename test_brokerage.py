import networkx as nx

from brokerage import global_brokerage
from nose.tools import assert_almost_equal


def assert_dict_almost_equal(d1, d2):
    for k in d1:
        assert_almost_equal(d1[k], d2[k])


def test_3_groups():
    edges = [('a1', 'a2'), ('a1', 'a3'), ('a2', 'a3'), ('a3', 'b1'),
             ('a2', 'b2'), ('b1', 'b2'), ('b2', 'c1'), ('b1', 'c3'),
             ('b2', 'c2'), ('c2', 'c3')]
    known_vals = {'a1': 0.,
                  'a2': 1. / 16 * (4 + 1. / 3),
                  'a3': 1. / 16 * (2.5 + 1. / 3),
                  'b1': 1. / 15 * (3.5 + 4. / 3),
                  'b2': (7 + 4. / 3) / 15,
                  'c1': 0.,
                  'c2': 1. / 16 * (1. / 3 + 1. / 2),
                  'c3': 1. / 16 * (1. / 3 + 1. / 2)}
    G = nx.Graph()
    G.add_edges_from(edges)
    groups = [{'a1', 'a2', 'a3'},
              {'b1', 'b2'},
              {'c1', 'c2', 'c3'}]

    assert_dict_almost_equal(global_brokerage(G, groups), known_vals)


def test_line_graph():
    edges = [('a1', 'b1'), ('b1', 'b2'), ('b2', 'c1'), ('c1', 'c2'),
             ('c2', 'b3'), ('b3', 'a2')]
    known_vals = {'a1': 0, 'a2': 0., 'b1': 4. / 12, 'b2': 6. / 12, 'b3': 4. / 12,
                  'c1': 6. / 11, 'c2': 5. / 11}
    G = nx.Graph()
    G.add_edges_from(edges)
    groups = [{'a1', 'a2'}, {'b1', 'b2', 'b3'}, {'c1', 'c2'}]

    assert_dict_almost_equal(global_brokerage(G, groups), known_vals)


def test_2_groups_unnormalized():
    edges = [('b1', 'a1'), ('a1', 'a2'), ('a1', 'b2'),
             ('a2', 'a3'), ('a3', 'b2')]
    groups = [{'a1', 'a2', 'a3'}, {'b1', 'b2'}]
    known_vals = {'a1': 2.5, 'a2': 0.5, 'a3': 0.5, 'b1': 0.0, 'b2': 0.5}
    G = nx.Graph()
    G.add_edges_from(edges)

    assert_dict_almost_equal(global_brokerage(G, groups, normalized=False),
                             known_vals)


def test_2_groups_line_graph():
    edges = [('a1', 'a2'), ('a2', 'b1'), ('b1', 'b2'), ('b2', 'b3')]
    groups = [{'a1', 'a2'}, {'b1', 'b2', 'b3'}]
    known_vals = {'a1': 0, 'a2': 1, 'b1': 1, 'b2': 0.5, 'b3': 0}
    G = nx.Graph()
    G.add_edges_from(edges)

    assert_dict_almost_equal(global_brokerage(G, groups), known_vals)


class TestDiGraph:
    def setup(self):
        edges = [('a1', 'a2'), ('a1', 'b2'), ('a2', 'a1'), ('a2', 'b1'),
                 ('b1', 'a1'), ('b1', 'b2')]
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)
        self.groups = [{'a1', 'a2'}, {'b1', 'b2'}]

    def test_unnormalized(self):
        known_vals = {'a1': 1.5, 'a2': 1., 'b1': 0.5, 'b2': 0.}

        B = global_brokerage(self.G, self.groups, normalized=False)
        assert_dict_almost_equal(B, known_vals)

    def test_normalized(self):
        known_vals = {'a1': 0.375, 'a2': 0.25, 'b1': 0.125, 'b2': 0.}

        assert_dict_almost_equal(global_brokerage(self.G, self.groups),
                                 known_vals)


class TestWeightedGraph:
    def setup(self):
        edges = [('a1', 'a2', 1), ('a2', 'b2', 3), ('a1', 'b1', 1),
                 ('b1', 'b3', 2), ('b2', 'b3', 1)]
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(edges)
        self.groups = [{'a1', 'a2'}, {'b1', 'b2', 'b3'}]

    def test_with_weights(self):
        known_vals = {'a1': 0.5, 'a2': 1. / 6, 'b1': 0.5, 'b2':
                      0.125, 'b3': 0.125}

        B = global_brokerage(self.G, self.groups, weight='weight')
        assert_dict_almost_equal(B, known_vals)

    def test_without_weights(self):
        known_vals = {'a1': 1. / 3, 'a2': 1. / 3, 'b1': 0.25, 'b2':
                      0.25, 'b3': 0.}

        B = global_brokerage(self.G, self.groups)
        assert_dict_almost_equal(B, known_vals)
