gefura
======

[![Test with tox](https://github.com/rafguns/gefura/actions/workflows/tox.yml/badge.svg)](https://github.com/rafguns/gefura/actions/workflows/tox.yml)
[![codecov](https://codecov.io/gh/rafguns/gefura/branch/master/graph/badge.svg?token=7OVWA23949)](https://codecov.io/gh/rafguns/gefura)

**gefura** is a small Python module that implements gefura measures, i.e. indicators that characterize to what extent a node forms a 'bridge' between groups in a network. Its name derives from Old Greek *γεφυρα*, meaning *bridge*.

These measures are adaptations of [betweenness centrality](http://en.wikipedia.org/wiki/Betweenness_centrality) where only shortest paths between nodes from different groups are taken into account. Previously they were known as Q-measures. Because *Q* is already used for many other concepts (e.g. modularity in networks), we have chosen the new name gefura.

This module implements global and local gefura. Both directed and undirected, as well as weighted and unweighted networks are supported.

**gefura** only depends on [NetworkX](http://networkx.github.io/).


Definition
----------
Global gefura is an indicator of brokerage between all groups in the network. The global gefura measure (unnormalized) of node $a$ is defined as

$$\Gamma_G(a) = \sum_{g, h \in V, group(g) \neq group(h)} \frac{p_{g,h}(a)}{p_{g,h}}$$

where $p_{g,h}$ is the number of shortest paths between nodes $g$ and $h$, and $p_{g,h}(a)$ is the number of shortest paths between $g$ and $h$ that pass through $a$ ($g \neq h\neq a$).

Local gefura is an indicator of brokerage between a node's own group and all other groups. If node $a$ belongs to group $A$, its local gefura measure (unnormalized) is

$$\Gamma_L(a) = \sum_{g \in A, h \notin A} \frac{p_{g,h}(a)}{p_{g,h}}$$


Example
-------

```python
>>> import networkx as nx
>>> from gefura import global_gefura
>>> G = nx.path_graph(5)
>>> groups = [{0, 2}, {1}, {3, 4}]
>>> global_gefura(G, groups)
{0: 0.0, 1: 0.5, 2: 0.8, 3: 0.6, 4: 0.0}
```
