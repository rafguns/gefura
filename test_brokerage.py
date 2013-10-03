from __future__ import division

import networkx as nx

from brokerage import global_brokerage, local_brokerage
from nose.tools import assert_almost_equal, raises


def assert_dict_almost_equal(d1, d2):
    for k in d1:
        assert_almost_equal(d1[k], d2[k])


def test_3_groups():
    edges = [('a1', 'a2'), ('a1', 'a3'), ('a2', 'a3'), ('a3', 'b1'),
             ('a2', 'b2'), ('b1', 'b2'), ('b2', 'c1'), ('b1', 'c3'),
             ('b2', 'c2'), ('c2', 'c3')]
    known_vals = {'a1': 0, 'a2': 13 / 48, 'a3': 17 / 96, 'b1': 29 / 90,
                  'b2': 5 / 9, 'c1': 0, 'c2': 5 / 96, 'c3': 5 / 96}
    G = nx.Graph()
    G.add_edges_from(edges)
    groups = [{'a1', 'a2', 'a3'}, {'b1', 'b2'}, {'c1', 'c2', 'c3'}]

    assert_dict_almost_equal(global_brokerage(G, groups), known_vals)


def test_line_graph():
    edges = [('a1', 'b1'), ('b1', 'b2'), ('b2', 'c1'), ('c1', 'c2'),
             ('c2', 'b3'), ('b3', 'a2')]
    known_vals = {'a1': 0, 'a2': 0, 'b1': 4 / 12, 'b2': 6 / 12, 'b3': 4 / 12,
                  'c1': 6 / 11, 'c2': 5 / 11}
    G = nx.Graph()
    G.add_edges_from(edges)
    groups = [{'a1', 'a2'}, {'b1', 'b2', 'b3'}, {'c1', 'c2'}]

    assert_dict_almost_equal(global_brokerage(G, groups), known_vals)


def test_2_groups_unnormalized():
    edges = [('b1', 'a1'), ('a1', 'a2'), ('a1', 'b2'),
             ('a2', 'a3'), ('a3', 'b2')]
    groups = [{'a1', 'a2', 'a3'}, {'b1', 'b2'}]
    known_vals = {'a1': 2.5, 'a2': 0.5, 'a3': 0.5, 'b1': 0, 'b2': 0.5}
    G = nx.Graph()
    G.add_edges_from(edges)

    assert_dict_almost_equal(global_brokerage(G, groups, normalized=False),
                             known_vals)
    assert_dict_almost_equal(local_brokerage(G, groups, normalized=False),
                             known_vals)


def test_2_groups_line_graph():
    edges = [('a1', 'a2'), ('a2', 'b1'), ('b1', 'b2'), ('b2', 'b3')]
    groups = [{'a1', 'a2'}, {'b1', 'b2', 'b3'}]
    known_vals = {'a1': 0, 'a2': 1, 'b1': 1, 'b2': 0.5, 'b3': 0}
    G = nx.Graph()
    G.add_edges_from(edges)

    assert_dict_almost_equal(global_brokerage(G, groups), known_vals)
    assert_dict_almost_equal(local_brokerage(G, groups), known_vals)


class TestDiGraph:
    def setup(self):
        edges = [('a1', 'a2'), ('a1', 'b2'), ('a2', 'a1'), ('a2', 'b1'),
                 ('b1', 'a1'), ('b1', 'b2')]
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)
        self.groups = [{'a1', 'a2'}, {'b1', 'b2'}]

    def test_global(self):
        known_vals_unnormalized = {'a1': 1.5, 'a2': 1, 'b1': 0.5, 'b2': 0}
        known_vals_normalized = {'a1': 0.375, 'a2': 0.25, 'b1': 0.125,
                                 'b2': 0}

        BG = global_brokerage(self.G, self.groups, normalized=False)
        assert_dict_almost_equal(BG, known_vals_unnormalized)
        assert_dict_almost_equal(global_brokerage(self.G, self.groups),
                                 known_vals_normalized)

    def test_local(self):
        known_vals_unnormalized_out = {'a1': 0.5, 'a2': 1, 'b1': 0, 'b2': 0}
        known_vals_unnormalized_in = {'a1': 1, 'a2': 0, 'b1': 0.5, 'b2': 0}
        known_vals_unnormalized_all = {'a1': 1.5, 'a2': 1, 'b1': 0.5, 'b2': 0}
        known_vals_normalized_out = {'a1': 0.25, 'a2': 0.5, 'b1': 0, 'b2': 0}
        known_vals_normalized_in = {'a1': 0.5, 'a2': 0, 'b1': 0.25, 'b2': 0}
        known_vals_normalized_all = {'a1': 0.375, 'a2': 0.25, 'b1': 0.125,
                                     'b2': 0}

        BL_out = local_brokerage(self.G, self.groups, normalized=False)
        assert_dict_almost_equal(BL_out, known_vals_unnormalized_out)
        BL_in = local_brokerage(self.G, self.groups, normalized=False,
                                direction='in')
        assert_dict_almost_equal(BL_in, known_vals_unnormalized_in)
        BL_all = local_brokerage(self.G, self.groups, normalized=False,
                                 direction='all')
        assert_dict_almost_equal(BL_all, known_vals_unnormalized_all)

        BL_out = local_brokerage(self.G, self.groups)
        assert_dict_almost_equal(BL_out, known_vals_normalized_out)
        BL_in = local_brokerage(self.G, self.groups, direction='in')
        assert_dict_almost_equal(BL_in, known_vals_normalized_in)
        BL_all = local_brokerage(self.G, self.groups, direction='all')
        assert_dict_almost_equal(BL_all, known_vals_normalized_all)


class TestWeightedGraph:
    def setup(self):
        edges = [('a1', 'a2', 1), ('a2', 'b2', 3), ('a1', 'b1', 1),
                 ('b1', 'b3', 2), ('b2', 'b3', 1)]
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(edges)
        self.groups = [{'a1', 'a2'}, {'b1', 'b2', 'b3'}]

    def test_with_weights(self):
        known_vals = {'a1': 0.5, 'a2': 1 / 6,
                      'b1': 0.5, 'b2': 0.125, 'b3': 0.125}

        B = global_brokerage(self.G, self.groups, weight='weight')
        assert_dict_almost_equal(B, known_vals)

    def test_without_weights(self):
        known_vals = {'a1': 1 / 3, 'a2': 1 / 3,
                      'b1': 0.25, 'b2': 0.25, 'b3': 0}

        B = global_brokerage(self.G, self.groups)
        assert_dict_almost_equal(B, known_vals)


class TestLocal:
    def setup(self):
        edges = [('a1', 'b1'), ('a2', 'b1'), ('b1', 'b2'), ('b2', 'c1'),
                 ('b2', 'c2'), ('a1', 'c1')]
        self.G = nx.Graph()
        self.G.add_edges_from(edges)
        self.groups = [{'a1', 'a2'}, {'b1', 'b2'}, {'c1', 'c2'}]

    def test_normalized(self):
        known_BL = {'a1': 0.125, 'a2': 0, 'b1': 0.375, 'b2': 0.375,
                    'c1': 0.125, 'c2': 0}
        BL = local_brokerage(self.G, self.groups)
        assert_dict_almost_equal(BL, known_BL)

    def test_unnormalized(self):
        known_BL = {'a1': 0.5, 'a2': 0, 'b1': 1.5, 'b2': 1.5,
                    'c1': 0.5, 'c2': 0}
        BL = local_brokerage(self.G, self.groups, normalized=False)
        assert_dict_almost_equal(BL, known_BL)


def test_local_line_graph():
    edges = [('a3', 'a2'), ('a2', 'c1'), ('c1', 'b1'), ('b1', 'a1'),
             ('a1', 'b2'), ('b2', 'b3')]
    groups = [{'a1', 'a2', 'a3'}, {'b1', 'b2', 'b3'}, {'c1'}]
    known_vals = {'a1': 4, 'a2': 4, 'a3': 0, 'b1': 6, 'b2': 4, 'b3': 0,
                  'c1': 0}
    G = nx.Graph()
    G.add_edges_from(edges)

    assert_dict_almost_equal(local_brokerage(G, groups, normalized=False),
                             known_vals)


@raises(ValueError)
def test_local_directed_wrong_direction_value():
    local_brokerage(nx.DiGraph(), [], direction="foobar")


def test_local_directed():
    edges = [('a1', 'a2'), ('a1', 'b1'), ('a2', 'a1'), ('a2', 'a3'),
             ('b1', 'b2'), ('b1', 'c1'), ('c1', 'a1'), ('c2', 'c1')]
    groups = [{'a1', 'a2', 'a3'}, {'b1', 'b2'}, {'c1', 'c2'}]
    known_out = {'a1': 0.375, 'a2': 0, 'a3': 0, 'b1': 0, 'b2': 0, 'c1': 1,
                 'c2': 0}
    known_in = {'a1': 0.75, 'a2': 0.375, 'a3': 0, 'b1': 0.8, 'b2': 0,
                'c1': 0, 'c2': 0}
    known_all = {'a1': 9 / 16, 'a2': 3 / 16, 'a3': 0, 'b1': 0.4, 'b2': 0,
                 'c1': 0.5, 'c2': 0}
    G = nx.DiGraph()
    G.add_edges_from(edges)

    for d, vals in (('out', known_out), ('in', known_in), ('all', known_all)):
        assert_dict_almost_equal(local_brokerage(G, groups, direction=d), vals)
