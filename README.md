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

source_nodes = [23, 45, 11, 98] # node ids chosen at random

# Heat kernel
hk4 = gk.eval_heat_kernel(4) # Heat Kernel at time instant t=4
proj_hk = gk.get_projection(source_nodes, hk4) # Projection of source nodes on whole network
ranking_hk = gk.get_ranking(proj_hk)

print proj_hk
print ranking_hk

# Random Walk with Restart kernel
rwr06 = gk.eval_rwr_kernel(0.6) # Random Walk with Restart kernel with restart probability alpha=0.6
proj_rwr = gk.get_projection(source_nodes, rwr06) # Projection of source nodes on whole network
ranking_rwr = gk.get_ranking(proj_rwr)

print proj_rwr
print ranking_rwr

# Random Walk kernel
rw3 = gk.eval_rw_kernel(3) # Simple Random Walk with 3 steps
proj_rw = gk.get_projection(source_nodes, rw3) # Projection of source nodes on whole network
ranking_rw = gk.get_ranking(proj_rw)

print proj_rw
print ranking_rw

# Diffusion State Distance kernel
dsd2 = gk.eval_dsd_kernel(2) # Diffusion State Distance with 2 time steps
proj_dsd = gk.get_projection(source_nodes, dsd2) # Projection of source nodes on whole network
ranking_dsd = gk.get_ranking(proj_dsd)

print proj_dsd
print ranking_dsd
```
