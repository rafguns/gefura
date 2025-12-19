import networkx as nx

import gefura

G = nx.path_graph(5)
groups = [{0, 2}, {1}, {3, 4}]
assert gefura.global_gefura(G, groups) == {0: 0.0, 1: 0.5, 2: 0.8, 3: 0.6, 4: 0.0}
