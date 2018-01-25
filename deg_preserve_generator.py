# Functions for generating degree-preserved sets of nodes in a networkX graph

def get_degree_binning(g, bin_size):
    # adaptive degree binning. bin_size is the minimum nodes present in each bin
    degree_to_nodes = {}
    for node, degree in g.degree().iteritems():
        degree_to_nodes.setdefault(degree, []).append(node)
    values = degree_to_nodes.keys()
    values.sort()
    bins = []
    i = 0
    while i < len(values):
        low = values[i]
        val = degree_to_nodes[values[i]]
        while len(val) < bin_size:
            i += 1
            if i == len(values):
                break
            val.extend(degree_to_nodes[values[i]])
        if i == len(values):
            i -= 1
        high = values[i]
        i += 1
        # print low, high, len(val)
        if len(val) < bin_size:
            low_, high_, val_ = bins[-1]
            bins[-1] = (low_, high, val_ + val)
        else:
            bins.append((low, high, val))
    return bins


def get_degree_equivalents(seeds, bins, g):
    seed_to_nodes = {}
    for seed in seeds:
        d = g.degree(seed)
        for l, h, nodes in bins:
            if l <= d and h >= d:
                mod_nodes = list(nodes)
                mod_nodes.remove(seed)
                seed_to_nodes[seed] = mod_nodes
                break
    return seed_to_nodes


def pick_random_nodes_matching_selected(network, bins, nodes_selected, n_random, degree_aware=True, connected=False):
    """
    Use get_degree_binning to get bins
    """
    import random
    values = []
    nodes = network.nodes()
    for i in xrange(n_random):
        if degree_aware:
            if connected:
                raise ValueError("Not implemented!")
            nodes_random = []
            node_to_equivalent_nodes = get_degree_equivalents(nodes_selected, bins, network)
            for node, equivalent_nodes in node_to_equivalent_nodes.iteritems():
                nodes_random.append(random.choice(equivalent_nodes))
        else:
            if connected:
                nodes_random = [random.choice(nodes)]
                i = 1
                while True:
                    if i == len(nodes_selected):
                        break
                    node_random = random.choice(nodes_random)
                    node_selected = random.choice(network.neighbors(node_random))
                    if node_selected in nodes_random:
                        continue
                    nodes_random.append(node_selected)
                    i += 1
            else:
                nodes_random = random.sample(nodes, len(nodes_selected))
        values.append(nodes_random)
    return values

def gen_degree_preserved_sets(nodeset, network, n_samples, bin_minimum_occupancy=30):
    """
        Sample degree-preserved sets of nodes, where degree sequence is extracted from nodeset

        Parameters
        ----------
        nodeset : list
            List of nodes of network from which the degree sequence is to be extracted

        network : networkX graph
            Network

        n_samples : int
            Number of samples to generate

        bin_minimum_occupancy : int
            Parameter for adaptive binning. Bin boundaries are adapted such that each bin is populated by at least bin_minimum_occupancy elements

        Returns
        -------
        list of lists
            List where each element is a list of random nodes with degree sequence equivalent to the one of nodeset

    """
    bins = get_degree_binning(network, bin_minimum_occupancy)
    return pick_random_nodes_matching_selected(network=network, bins=bins, nodes_selected=nodeset, n_random=n_samples)