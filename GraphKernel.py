from __future__ import division
import numpy as np
import networkx as nx
import warnings
import kernels
import statsig
from functools import partial

class GIDMapper:
    def __init__(self, nodelist):
        self.nodelist = nodelist
        self._gid2id_dict = {gid: i for i, gid in enumerate(nodelist)}
        self._dtype = type(nodelist[0])

    def id2gid(self, ids):
        if isinstance(ids, int):
            return self.nodelist[ids]
        else: # if it is a list of ids
            return [int(self.nodelist[i]) for i in ids]

    def gid2id(self, gids):
        if isinstance(gids, self._dtype):
            return self._gid2id_dict[gids]
        else: # if it is a list of gids
            return [self._gid2id_dict[gid] for gid in gids]

class GraphKernel:
    def __init__(self, graph=None, savefile=None, nodelist=None, weight='weight', verbose_level=1):
        """
            Instantiates a GraphKernel object. Can be initialized by passing a networkx graph or from a savefile

            Parameters
            ----------
            graph : NetworkX graph or numpy ndarray
                Input graph as NetworkX graph or as numpy adjacency matrix (numpy array)

            savefile : str
                Path of savefile

            nodelist : list
                List of node IDs to be used for rows/columns ordering in the adjacency matrix. If None nodelist = graph.nodes().
                This parameter is best left as default

            weight : str
                Name of the NetworkX edge property to be considered as edge weight for weighted graphs
                If graph is provided as numpy adjacency matrix this parameter is ignored

            verbose_level : int
                Level of verbosity. Current implemented levels: 0 for no output, 1 for basic output

            Returns
            -------
            (numpy array, numpy array)
                Mean and std.dev. vectors of seedsets projections distribution

        """
        self.verbose_level = verbose_level
        if savefile is None:
            self.speak("Initializing GraphKernel...", newline=False, verbose_level=1)
            if isinstance(graph, np.ndarray) or isinstance(graph, np.matrixlib.defmatrix.matrix):
                self.adj = np.asarray(graph)
                if nodelist is None:
                    nodelist = range(self.adj.shape[0])
            else: # if NetworkX graph
                self.adj = np.array(nx.adjacency_matrix(graph ,nodelist=nodelist, weight='weight').todense())
                if nodelist is None:
                    nodelist = graph.nodes()
            self.nodelist = nodelist
            self.gm = GIDMapper(nodelist=nodelist)
            self.kernels = {}
            self.speak("Complete.", newline=True, verbose_level=1)
        else:
            self.load_kernels(savefile)

    def eval_rw_kernel(self, nRw):
        """
            Simple Random Walk kernel.

            Parameters
            ----------
            nRw : int
                Number of steps of random walk

            Returns
            -------
            str
                Kernel ID (KID) to identify the corresponding kernel

        """
        kid = 'rw_'+str(nRw)
        if kid not in self.kernels:
            self.speak('Initializing RW kernel (this may take a while)...', newline=False, verbose_level=1)
            self.kernels[kid] = kernels.rw_kernel(self.adj, nRw)
            self.speak('Complete.', newline=True)
        return kid

    def eval_rwr_kernel(self, alpha):
        """
            Random Walk with Restart kernel.

            Parameters
            ----------
            alpha : float
                Restart probability of random walk

            Returns
            -------
            str
                Kernel ID (KID) to identify the corresponding kernel

        """
        kid = 'rwr_' + str(alpha)
        if kid not in self.kernels:
            self.speak('Initializing RWR kernel (this may take a while)...', newline=False, verbose_level=1)
            self.kernels[kid] = kernels.rwr_kernel(self.adj, alpha)
            self.speak('Complete.', newline=True)
        return kid

    def eval_dsd_kernel(self, nRw):
        """
            Diffusion State Distance kernel, as defined in Cao et al., 2013.

            Parameters
            ----------
            nRw : int
                Number of steps of random walk

            Returns
            -------
            str
                Kernel ID (KID) to identify the corresponding kernel

        """
        kid = 'dsd_'+str(nRw)
        if kid not in self.kernels:
            self.speak('Initializing DSD kernel (this may take a while)...', newline=False, verbose_level=1)
            self.kernels[kid] = kernels.dsd_kernel(self.adj, nRw)
            self.speak('Complete.', newline=True)
        return kid

    def eval_heat_kernel(self, t):
        """
            Heat kernel, as defined in Cowen et al., 2017

            Parameters
            ----------
            t : float
                Diffusion time value

            Returns
            -------
            str
                Kernel ID (KID) to identify the corresponding kernel

        """
        kid = 'hk_' + str(t)
        if kid not in self.kernels:
            self.speak('Initializing heat kernel (this may take a while)...', newline=False, verbose_level=1)
            self.kernels[kid] = kernels.heat_kernel(self.adj, t)
            self.speak('Complete.', newline=True)
        return kid

    def eval_istvan_kernel(self):
        kid = 'ist'
        if kid not in self.kernels:
            self.speak('Initializing Istvan kernel (this may take a while)...', newline=False, verbose_level=1)
            self.kernels[kid] = kernels.istvan_kernel(self.adj)
            self.speak('Complete.', newline=True)
        return kid

    def eval_kernel_statistics(self, kernel, n_samples=None, rdmmode='CONFIGURATION_MODEL'):
        if rdmmode == 'CONFIGURATION_MODEL':
            if n_samples is None:
                raise ValueError("n_samples argument must be provided if samples is set to 'CONFIGURATION_MODEL'.")
            graph = self.rebuild_nx_graph()
            degseq = [nx.degree(graph,node) for node in self.nodelist]

            # Welford's method (ca. 1960) to calculate running variance
            N = n_samples
            M = 0
            S = 0
            try:
                from tqdm import tnrange # if tqdm is present use tqdm progress bar
                rangefunc = tnrange
            except ImportError: # if tqdm is not present it will fallback on standard loop
                rangefunc = range
            self.speak('Calculating kernel statistics (this may take a long while)...', newline=False, verbose_level=1)
            for k in rangefunc(N):
                rdmgraph = nx.relabel_nodes(nx.configuration_model(degseq), mapping={i:self.nodelist[i] for i in range(len(self.nodelist))})
                rdmadj = np.asarray(nx.adjacency_matrix(rdmgraph, nodelist=self.nodelist).todense())
                kmatrix = self.kid2func(kernel)(A=rdmadj)
                x = kmatrix
                oldM = M
                M = M + (x - M) / (k + 1)
                S = S + (x - M) * (x - oldM)
            self.kernels[kernel + '_cmmean'] = M
            self.kernels[kernel + '_cmstd'] = np.sqrt(S / (N - 1))
            self.speak('Complete', newline=True, verbose_level=1)
        else:
            raise ValueError('Incorrect rdmmode parameter selected ({}): possible modes are CONFIGURATION_MODEL.'.format(rdmmode))

    def onehot_encode(self, nodeset, norm=False):
        vec = np.zeros(self.adj.shape[0])
        vec[self.gm.gid2id(nodeset)] = 1
        if norm:
            vec /= vec.sum()
        return vec

    def vec2dict(self, nodevec):
        return {self.gm.id2gid(i): nodevec[i] for i in xrange(len(nodevec))}

    def dict2vec(self, nodedict):
        return np.array([nodedict[node] for node in self.nodelist])

    def get_projection(self, seedset, kernel, destset=None, correction=False, rdm_seedsets=None, significance_formula='ZSCORE', norm=False, return_dict=True):
        """
            Computes the projection (similarity) of the nodes in the seedset with the destset nodes. If destset is None the projection to the full network is computed.

            Parameters
            ----------
            seedset : list
                List of source nodes

            kernel : str or numpy matrix
                Kernel ID (KID) of the chosen kernel or numpy kernel matrix A_ij where represents the projection from node j (source) to node i (destination)

            destset : list
                List of destination nodes. If return_dict is False this parameter is ignored.

            correction : str or False
                Projection score correction to account for statistical biases such as high degree.
                Options are:
                    - False: no correction
                    - 'SEEDSET_SIZE': the final score vector is divided by the number of source nodes
                    - 'DEGREE_CENTRALITY': the score of each destination node is divided by its degree
                    - 'RDM_SEED': statistical significance of the score is evaluated by considering random samples of the source nodes in seedset.
                        Random samples of seedset have to be provided through the rdm_seedsets parameter.
                        In this mode the output projection nodes will be the significance values of the uncorrected scores,
                        calculated according to the formula specified by the significance_formula parameter
                    - 'CONFIGURATION MODEL': statistical significance of the score is evaluated by considering random configuration model samples of the network.

            rdm_seedsets : list of lists
                List of lists containing random samples of the seedset list. To be used for statistical significance evaluation.
                If correction is not set to 'RDM_SEED' this parameter is ignored

            significance_formula : str
                Formula to calculate statistical significance.
                Options are:
                    - 'ZSCORE': (value - mean) / std.dev.
                    - 'ISTVAN': value - mean - 2 * std.dev.

            norm : bool
                Whether to normalize the output projection vector. Useful if comparing projections of several source nodesets with different sizes

            return_dict: bool
                Whether the output projection has to be returned as a {node_id : value} dict or as a dense N-dim vector

            Returns
            -------
            dict or list
                Output projection from seedset nodes to destset nodes

        """
        if isinstance(kernel, basestring):
            kid = kernel
            kernel = self.kernels[kernel]
        else:
            kid = None
        seedvec = self.onehot_encode(seedset)
        if not correction:
            nodevec = np.dot(kernel, seedvec)
        elif correction == 'SEEDSET_SIZE': # number of genes in the seed set
            nodevec = np.dot(kernel, seedvec) / seedvec.sum()
        elif correction == 'DEGREE_CENTRALITY': # degree centrality
            k0 = self.adj.sum(axis=1)
            nodevec = np.dot(kernel, seedvec) / k0
        elif correction == 'RDM_SEED':
            if rdm_seedsets is None:
                raise ValueError('rdm_seedsets param must be set when in RDM_SEED mode!')
            samples_proj = self.get_projections_batch(rdm_seedsets, kernel)
            if significance_formula == 'ZSCORE':
                nodevec = statsig.zscore(np.dot(kernel, seedvec), samples_proj)
            elif significance_formula == 'ONETAIL':
                nodevec = statsig.onetail(np.dot(kernel, seedvec), samples_proj)
            elif significance_formula == 'ISTVAN':
                nodevec = statsig.istvan(np.dot(kernel, seedvec), samples_proj)
            else:
                raise ValueError('Incorrect significance formula selected ({}): possible modes are ZSCORE, ONETAIL, ISTVAN.'.format(significance_formula))
        elif correction == 'CONFIGURATION_MODEL':
            if kid is None or kid+'_cmmean' not in self.kernels.keys() or kid+'_cmstd' not in self.kernels.keys():
                raise ValueError('CONFIGURATION_MODEL correction can be invoked only for pre_calculated kernels and kernel statistics. Call eval_kernel_statistics() function on the selected kernel to make this mode accessible.')
            mean, std = self[kid+'_cmmean'], self[kid+'_cmstd']
            if significance_formula == 'ZSCORE':
                nodevec = statsig.zscore(np.dot(kernel, seedvec), mean=np.dot(mean, seedvec), std=np.dot(std, seedvec))
            elif significance_formula == 'ISTVAN':
                nodevec = statsig.istvan(np.dot(kernel, seedvec), mean=np.dot(mean, seedvec), std=np.dot(std, seedvec))
            else:
                raise ValueError('Incorrect significance formula selected ({}): possible modes are ZSCORE, ISTVAN.'.format(significance_formula))
        else:
            raise ValueError('Incorrect mode selected ({}): possible modes are SEEDSET_SIZE, DEGREE_CENTRALITY, RDM_SEED, CONFIGURATION_MODEL.'.format(correction))
        if norm:
            nodevec /= nodevec.sum()
        if return_dict:
            valuedict = self.vec2dict(nodevec)
            if destset is not None:
                return {key:value for key,value in valuedict if key in destset}
            else:
                return valuedict
        else:
            return nodevec

    def get_projections_batch(self, seedsets, kernel):
        """
            Evaluates list of projections from list of sets of source nodes

            Parameters
            ----------
            seedsets : list of lists
                Each list is a set of source nodes to evaluate a distribution of projections

            kernel : str or numpy matrix
                Kernel ID (KID) of the chosen kernel or numpy kernel matrix A_ij where represents the projection from node j (source) to node i (destination)

            Returns
            -------
            list of numpy arrays
                N x N_samples numpy matrix, where N_samples is len(seedsets), and each column is a numpy vector of projection scores

        """
        if isinstance(kernel, basestring):
            kernel = self.kernels[kernel]
        seedvecs = np.array(map(self.onehot_encode, seedsets)).T
        samples = np.dot(kernel, seedvecs)
        return samples

    def get_projection_statistics(self, seedsets, kernel):
        """
            Evaluates mean and standard deviation of projections from seedsets source nodes to network nodes

            Parameters
            ----------
            seedsets : list of lists
                Each list is a set of source nodes to evaluate a distribution of projections

            kernel : str or numpy matrix
                Kernel ID (KID) of the chosen kernel or numpy kernel matrix A_ij where represents the projection from node j (source) to node i (destination)

            Returns
            -------
            (numpy array, numpy array)
                Mean and std.dev. vectors of seedsets projections distribution

        """

        samples = self.get_projections_batch(seedsets=seedsets, kernel=kernel)
        return samples.mean(axis=1), samples.std(axis=1)

    def get_ranking(self, projection, candidateset=None):
        """
            Evaluates ranking of nodes from a projection vector or dict

            Parameters
            ----------
            projection : dict or numpy array
                Projection dict/vector obtained with get_projection function

            candidate_set : set of destination nodes to consider for the ranking. If None all network nodes are considered

            Returns
            -------
            list
                List of network nodes ordered by increasing rank

        """
        if isinstance(projection,dict):
            projection = self.dict2vec(projection)
        if candidateset is None:
            return self.gm.id2gid(np.argsort(projection)[::-1])
        else:
            return [self.gm.id2gid(i) for i in np.argsort(projection)[::-1] if self.gm.id2gid(i) in candidateset]

    def available_kernels(self):
        """
            Returns list of KIDs of kernels cached in GraphKernel object. To directly obtain a kernel matrix use the getitem operator
            e.g.
                kernel = gk[kid]   where gk is a GraphKernel instance and kid is the Kernel ID

            Returns
            -------
            list
                List of Kernel IDs cached in GraphKernel instance

        """
        return self.kernels.keys()

    def save(self, filename, kidlist=None, description=None):
        """
            Save kernels to file

            Parameters
            ----------
            filename : str
                Path of savefile (.h5 format)

            kidlist : list
                List of KIDs to save on file. If None all kernels are saved

            description : str
                Optional description text embedded in the kernel savefile

        """
        import h5py
        if kidlist is None:
            kidlist = self.kernels.keys()
        elif isinstance(kidlist, basestring):
            kidlist = [kidlist]
        self.speak("Saving kernels...", newline=False, verbose_level=1)
        with h5py.File(filename, 'w') as hf:
            for kid in kidlist:
                hf.create_dataset(kid, data=self.kernels[kid], compression="gzip")
            hf.create_dataset('nodelist', data=self.nodelist)
            hf.create_dataset('adjacency', data=self.adj)
            if description is not None:
                hf.create_dataset('description', data=description, compression="gzip")
        self.speak("Complete.", newline=True, verbose_level=1)

    def load(self, filename):
        """
            Load kernels from file

            Parameters
            ----------
            filename : str
                Path of savefile

        """
        import h5py
        if hasattr(self, 'kernels') and len(self.kernels) > 0:
            warnings.warn('Loaded GraphKernel is overwriting an existing kernel set.')
        self.speak("Loading kernels...", newline=False, verbose_level=1)
        with h5py.File(filename, 'r') as hf:
            data = {}
            for key in hf.keys():
                if key == 'description':
                    data['description'] = hf['description'].value
                elif key == 'nodelist':
                    self.nodelist = hf['nodelist'][:]
                elif key == 'adjacency':
                    self.adj = hf['adjacency'][:]
                else:
                    data[key] = hf[key][:]
        self.kernels = data
        self.gm = GIDMapper(nodelist=self.nodelist)
        self.speak("Complete.", newline=True, verbose_level=1)

    def rebuild_nx_graph(self):
        graph = nx.from_numpy_matrix(self.adj)
        return nx.relabel_nodes(graph, {i:self.nodelist[i] for i in range(len(self.nodelist))})

    def kid2func(self, kid):
        kid = kid.split('_')
        if kid[0] == 'rw':
            return partial(kernels.rw_kernel, nRw=int(kid[1]))
        elif kid[0] == 'rwr':
            return partial(kernels.rwr_kernel, alpha=float(kid[1]))
        elif kid[0] == 'hk':
            return partial(kernels.heat_kernel, t=float(kid[1]))
        elif kid[0] == 'dsd':
            return partial(kernels.dsd_kernel, nRw=int(kid[1]))
        elif kid[0] == 'ist':
            return kernels.istvan_kernel

    def __getitem__(self, kid):
        return self.kernels[kid]

    def speak(self, message, newline=False, verbose_level=1):
        if self.verbose_level >= verbose_level:
            print message,
            if newline:
                print


if __name__ == '__main__':
    # Example usage
    g = nx.barabasi_albert_graph(300, 3)
    gk = GraphKernel(g)
    k = gk.eval_rwr_kernel(0.3)
    proj = gk.get_projection(range(100, 110), k, correction='DEGREE_CENTRALITY')
    rank = gk.get_ranking(proj, range(50))