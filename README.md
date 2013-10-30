brokerage
=========

brokerage is a small Python module that implements brokerage measures, i.e. indicators that characterize to what extent a node forms a 'bridge' between groups in a network. 

These measures are adaptations of [betweenness centrality](http://en.wikipedia.org/wiki/Betweenness_centrality) where only shortest paths between nodes from different groups are taken into account. The brokerage measures are also known as 'Q-measures' (ot to be mistaken with e.g. modularity Q).

This module implements global and local brokerage. Both directed and undirected, as well as weighted and unweighted networks are supported.


Definition
----------

Global brokerage is defined as

![B_G(a) = \frac{1}{M} \sum_{\substack{g, h \in V\\group(g) \neq group(h)}} \frac{p_{g,h}(a)}{p_{g,h}}](http://latex.codecogs.com/gif.latex?B_G%28a%29%20%3D%20%5Cfrac%7B1%7D%7BM%7D%20%5Csum_%7B%5Csubstack%7Bg%2Ch%20%5Cin%20V%5C%5Cgroup%28g%29%20%5Cneq%20group%28h%29%7D%7D%20%5Cfrac%7Bp_%7Bg%2Ch%7D%28a%29%7D%7Bp_%7Bg%2Ch%7D%7D)

where _M_ is

![M = \sum_{k,l} |G_k \setminus \{a\}| \cdot |G_l \setminus \{a\}|](http://latex.codecogs.com/gif.latex?M%20%3D%20%5Csum_%7Bk%2Cl%7D%20%7CG_k%20%5Csetminus%20%5C%7Ba%5C%7D%7C%20%5Ccdot%20%7CG_l%20%5Csetminus%20%5C%7Ba%5C%7D%7C)


Example
-------

```python
>>> import networkx as nx
>>> from brokerage import global_brokerage
>>> G = nx.path_graph(5)
>>> groups = [{0, 2}, {1}, {3, 4}]
>>> brokerage.global_brokerage(G, groups)
{0: 0.0, 1: 0.5, 2: 0.8, 3: 0.6, 4: 0.0}
```
