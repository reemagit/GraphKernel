# GraphKernel
Python package to evaluate several similarity measures between nodes in a networkX graph and rank network nodes according to their similarity to a subset of source nodes. This package is optimized to make use of vectorized numpy and scipy expression in order to obtain near-native performances. It can be utilized for quickly comparing different network-propagation-based similarity measures in network science problems such as **gene prioritization** and **disease module detection**.

The methods in this package have been inspired by the network propagation framework explained in the wonderful paper 'Network propagation: a universal amplifier of genetic associations', Cowen et al. (2017), Nat Rev Genetics.

## Example of basic usage

```
from __future__ import division
import networkx as nx

import GraphKernel

g = nx.barabasi_albert_graph(100,1)

gk = GraphKernel.GraphKernel(g, verbose_level=1)

hk4 = gk.eval_heat_kernel(4) # Heat Kernel at time instant t=4
rwr06 = gk.eval_rwr_kernel(0.6) # Random Walk with Restart kernel with restart probability alpha=0.6
rw3 = gk.eval_rw_kernel(3) # Simple Random Walk with 3 steps
dsd2 = gk.eval_dsd_kernel(2) # Diffusion State Distance with 2 time steps

source_nodes = [23, 45, 11, 98] # node ids chosen at random

proj = gk.get_projection(source_nodes, rwr06) # Projection of source nodes on whole network with RWR kernel (alpha=0.6)

print proj

ranking = gk.get_ranking(proj)

print ranking
```
