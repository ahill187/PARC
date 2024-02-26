import numpy as np
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg
import time
from umap.umap_ import find_ab_params, simplicial_set_embedding
from parc.k_nearest_neighbours import create_hnsw_index
from parc.metrics import accuracy
from parc.graph import Community


class PARC:
    """Phenotyping by accelerated refined community-partitioning.

    Attributes:
        x_data: (np.array) a Numpy array of the input x data, with dimensions
            (n_samples, n_features).
        y_data_true: (np.array) a Numpy array of the output y labels.
        dist_std_local: (int) similar to the jac_std_global parameter. Avoid setting local and
            global pruning to both be below 0.5 as this is very aggresive pruning.
            Higher ``dist_std_local`` means more edges are kept.
        jac_std_global: (float) 0.15 is a recommended value performing empirically similar
            to ``median``. Generally values between 0-1.5 are reasonable.
            Higher ``jac_std_global`` means more edges are kept.
        keep_all_local_dist: (bool) whether or not to do local pruning.
            Default is None, which omits local pruning for samples > 300 000 cells.
        large_community_factor: (float) if a cluster exceeds this share of the entire cell population,
            then the PARC will be run on the large cluster. At 0.4 it does not come into play.
        small_community_size: (int) the smallest population size to be considered a community.
        small_community_timeout: (int) the maximum number of seconds trying to check an outlying
            small community.
        jac_weighted_edges: (bool) whether to partition using the weighted graph.
        knn: (int) the number of clusters k for the k-nearest neighbours algorithm.
        n_iter_leiden: (int) the number of iterations for the Leiden algorithm.
        random_seed: (int) the random seed to enable reproducible Leiden clustering.
        n_threads: (int) the number of threads used in the KNN algorithm.
        distance: (string) the distance metric to be used in the KNN algorithm:
            "l2": Euclidean distance L^2 norm, d = sum((x_i - y_i)^2)
            "cosine": cosine similarity, d = 1.0 - sum(x_i*y_i) / sqrt(sum(x_i*x_i) * sum(y_i*y_i))
            "ip": inner product distance, d = 1.0 - sum(x_i*y_i)
        partition_type: (string) the partition type to be used in the Leiden algorithm:
            "ModularityVP": ModularityVertexPartition, ``resolution_parameter=1``
            "RBVP": RBConfigurationVP, Reichardt and Bornholdt’s Potts model. Note that this is the
                same as ModularityVertexPartition when setting 𝛾 = 1 and normalising by 2m.
        resolution_parameter: (float) the resolution parameter to be used in the Leiden algorithm.
            In order to change ``resolution_parameter``, we switch to ``RBVP``.
        knn_struct: (TODO) the hnsw index of the KNN graph on which we perform queries.
        neighbor_graph: (TODO) CSR affinity matrix for pre-computed nearest neighbors.
        hnsw_param_ef_construction: (int) a higher value increases accuracy of index construction.
            Even for O(100 000) cells, 150-200 is adequate.
    """

    def __init__(self, x_data, y_data_true=None, dist_std_local=3, jac_std_global="median",
                 keep_all_local_dist=None, large_community_factor=0.4, small_community_size=10,
                 jac_weighted_edges=True, knn=30, n_iter_leiden=5, random_seed=42,
                 n_threads=-1, distance="l2", small_community_timeout=15, partition_type="ModularityVP",
                 resolution_parameter=1.0, knn_struct=None, neighbor_graph=None,
                 hnsw_param_ef_construction=150):

        self.y_data_true = y_data_true
        self._x_data = x_data
        self.dist_std_local = dist_std_local
        self.jac_std_global = jac_std_global
        self._keep_all_local_dist = keep_all_local_dist
        self.large_community_factor = large_community_factor
        self.small_community_size = small_community_size
        self.jac_weighted_edges = jac_weighted_edges
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden
        self.random_seed = random_seed
        self.n_threads = n_threads
        self.distance = distance
        self.small_community_timeout = small_community_timeout
        self.resolution_parameter = resolution_parameter
        self._partition_type = partition_type
        self.knn_struct = knn_struct
        self.neighbor_graph = neighbor_graph
        self.hnsw_param_ef_construction = hnsw_param_ef_construction

    @property
    def x_data(self):
        return self._x_data

    @x_data.setter
    def x_data(self, x_data):
        self._x_data = x_data
        if self.y_data_true is None:
            self.y_data_true = [1] * x_data.shape[0]

    @property
    def keep_all_local_dist(self):
        return self._keep_all_local_dist

    @keep_all_local_dist.setter
    def keep_all_local_dist(self, keep_all_local_dist):
        if keep_all_local_dist is None:
            if self.x_data.shape[0] > 300000:
                keep_all_local_dist = True  # skips local pruning to increase speed
            else:
                keep_all_local_dist = False

        self.keep_all_local_dist = keep_all_local_dist

    @property
    def partition_type(self):
        return self._partition_type

    @partition_type.setter
    def partition_type(self, partition_type):
        if self.resolution_parameter != 1:
            self.partition_type = "RBVP"
        else:
            self.partition_type = partition_type

    def get_knn_struct(self):
        if self.knn_struct is None:
            self.knn_struct = self.make_knn_struct()
        return self.knn_struct

    def make_knn_struct(self, too_big=False, big_cluster=None):
        """Create a Hierarchical Navigable Small Worlds (HNSW) graph.

        See `hnswlib.Index
        <https://github.com/nmslib/hnswlib/blob/master/python_bindings/LazyIndex.py>`__.

        Args:
            too_big: (bool) TODO.
            big_cluster: TODO.

        Returns:
            (hnswlib.Index): TODO.
        """
        if self.knn > 190:
            print(f"knn = {self.knn}; consider using a lower K_in for KNN graph construction")

        if not too_big:
            data = self.x_data
            distance = self.distance
        else:
            data = big_cluster
            distance = "l2"

        hnsw_index = create_hnsw_index(
            data, distance, self.knn, self.n_threads,
            self.hnsw_param_ef_construction, too_big
        )

        return hnsw_index

    def knngraph_full(self):#, neighbor_array, distance_array):
        k_umap = 15
        # neighbors in array are not listed in in any order of proximity
        self.knn_struct.set_ef(k_umap+1)
        neighbor_array, distance_array = self.knn_struct.knn_query(self.x_data, k=k_umap)

        row_list = []
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]

        row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))

        row_min = np.min(distance_array, axis=1)
        row_sigma = np.std(distance_array, axis=1)

        distance_array = (distance_array - row_min[:, np.newaxis]) / row_sigma[:, np.newaxis]

        col_list = neighbor_array.flatten().tolist()
        distance_array = distance_array.flatten()
        distance_array = np.sqrt(distance_array)
        distance_array = distance_array * -1

        weight_list = np.exp(distance_array)

        threshold = np.mean(weight_list) + 2 * np.std(weight_list)

        weight_list[weight_list >= threshold] = threshold

        weight_list = weight_list.tolist()

        graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                           shape=(n_cells, n_cells))

        graph_transpose = graph.T
        prod_matrix = graph.multiply(graph_transpose)

        graph = graph_transpose + graph - prod_matrix
        return graph

    def prune_local(self, neighbor_array, distance_array):
        """Prune the nearest neighbors array.

        If ``keep_all_local_dist`` is true, remove any neighbors which are further away than
        the specified cutoff distance. Also, remove any self-loops. Return in the ``csr_matrix``
        format.

        If ``keep_all_local_dist`` is false, then don't perform any pruning and return the original
        arrays in the ``csr_matrix`` format.

        Args:
            neighbor_array: (np.array) An array with dimensions (n_samples, k) listing the
                k nearest neighbors for each data point.
            neighbor_array: (np.array) An array with dimensions (n_samples, k) listing the
                distances to each of the k nearest neighbors for each data point.

        Returns:
            (Compressed Sparse Row Matrix) A sparse matrix with dimensions (n_samples, n_samples),
                containing the pruned distances.
        """
        # neighbor array not listed in in any order of proximity
        row_list = []
        col_list = []
        weight_list = []

        n_neighbors = neighbor_array.shape[1]
        n_samples = neighbor_array.shape[0]
        discard_count = 0
        if not self.keep_all_local_dist:  # locally prune based on (squared) l2 distance

            print(f"""Starting local pruning based on Euclidean distance metric at
                   {self.dist_std_local} standard deviations above mean""")
            distance_array = distance_array + 0.1
            for neighbors, sample_index in zip(neighbor_array, range(n_samples)):
                distances = distance_array[sample_index, :]
                max_distance = np.mean(distances) + self.dist_std_local * np.std(distances)
                to_keep = np.where(distances < max_distance)[0]
                updated_neighbors = neighbors[np.ix_(to_keep)]
                updated_distances = distances[np.ix_(to_keep)]
                discard_count = discard_count + (n_neighbors - len(to_keep))

                for index in range(len(updated_neighbors)):
                    if sample_index != neighbors[index]:  # remove self-loops
                        row_list.append(index)
                        col_list.append(updated_neighbors[index])
                        dist = np.sqrt(updated_distances[index])
                        weight_list.append(1 / (dist + 0.1))

        if self.keep_all_local_dist:  # dont prune based on distance
            row_list.extend(list(np.transpose(
                np.ones((n_neighbors, n_samples)) * range(0, n_samples)).flatten()
            ))
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1. / (distance_array.flatten() + 0.1)).tolist()

        csr_graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                               shape=(n_samples, n_samples))
        return csr_graph

    def prune_global(self, csr_array, jac_std_threshold, jac_weighted_edges=True):
        """Prune the graph globally based on the Jaccard similarity measure.

        The ``csr_array`` contains the locally-pruned pairwise distances. From this, we can
        use the Jaccard similarity metric to compute the similarity score for each edge. We then
        remove any edges from the graph that do not meet a minimum similarity threshold.

        Args:
            csr_array: (Compressed Sparse Row Matrix) A sparse matrix with dimensions
                (n_samples, n_samples), containing the locally-pruned pair-wise distances.

        Returns:
            (igraph.Graph) a Graph object which has now been locally and globally pruned.
        """

        input_nodes, output_nodes = csr_array.nonzero()
        edges = list(zip(input_nodes, output_nodes))
        edges_copy = np.asarray(edges.copy())

        graph = ig.Graph(edges, edge_attrs={'weight': csr_array.data.tolist()})

        similarities = np.asarray(graph.similarity_jaccard(pairs=list(edges_copy)))

        print("Starting global pruning")

        if jac_std_threshold == "median":
            threshold = np.median(similarities)
        else:
            threshold = np.mean(similarities) - jac_std_threshold * np.std(similarities)

        indices_similar = np.where(similarities > threshold)[0]

        if jac_weighted_edges:
            graph_pruned = ig.Graph(
                n=csr_array.shape[0],
                edges=list(edges_copy[indices_similar]),
                edge_attrs={"weight": list(similarities[indices_similar])}
            )
        else:
            graph_pruned = ig.Graph(
                n=csr_array.shape[0],
                edges=list(edges_copy[indices_similar])
            )
        graph_pruned.simplify(combine_edges="sum")  # "first"
        return graph_pruned

    def get_leiden_partition(self, graph, jac_weighted_edges=True):
        if jac_weighted_edges:
            weights = "weight"
        else:
            weights = None

        if self.partition_type == "ModularityVP":
            print("Leiden algorithm find partition: partition type = ModularityVertexPartition")
            partition = leidenalg.find_partition(
                graph=graph,
                partition_type=leidenalg.ModularityVertexPartition, weights=weights,
                n_iterations=self.n_iter_leiden, seed=self.random_seed
            )
        else:
            print("Leiden algorithm find partition: partition type = RBConfigurationVertexPartition")
            partition = leidenalg.find_partition(
                graph=graph,
                partition_type=leidenalg.RBConfigurationVertexPartition, weights=weights,
                n_iterations=self.n_iter_leiden, seed=self.random_seed,
                resolution_parameter=self.resolution_parameter
            )
        return partition

    def check_if_cluster_oversized(self, node_communities, community_id, big_cluster_sizes=[]):
        """Check if the community is too big.

        Args:
            node_communities: (np.array) an array containing the community assignments for each
                node.
            community_id: (int) the integer id of the community.

        Returns:
            too_big: (bool) whether or not the community is too big.
            big_cluster_indices: (list) a list of node indices for the community, if it is too big.
            big_cluster_sizes: (list) the sizes of the communities that are too big.
        """

        too_big = False
        n_samples = node_communities.shape[0]
        cluster_indices = np.where(node_communities == community_id)[0]
        cluster_size = len(cluster_indices)
        big_cluster_indices = []
        not_yet_expanded = cluster_size not in big_cluster_sizes
        if cluster_size > self.large_community_factor * n_samples and not_yet_expanded:
            too_big = True
            big_cluster_indices = cluster_indices
            big_cluster_sizes.append(cluster_size)
            print(f"""Community {community_id} is too big, cluster size = {cluster_size}.
                It will be expanded.""")
        return too_big, big_cluster_indices, big_cluster_sizes

    def get_next_big_community(self, node_communities, big_cluster_sizes):
        """Find the next community which is too big, if it exists.

        Args:
            node_communities: (np.array) an array containing the community assignments for each
                node.
            big_cluster_sizes: (list) a list of community sizes which have already been identified
                and processed as too big.

        Returns:
            too_big: (bool) whether or not the community is too big.
            big_cluster_indices: (list) a list of node indices for the community, if it is too big.
            big_cluster_sizes: (list) the sizes of the communities that are too big.
        """
        too_big = False
        communities = set(node_communities)

        for community_id in communities:
            too_big, big_cluster_indices, big_cluster_sizes = self.check_if_cluster_oversized(
                node_communities, community_id, big_cluster_sizes
            )
            if too_big:
                break

        return too_big, big_cluster_indices, big_cluster_sizes

    def get_small_communities(self, node_communities, small_community_size):
        """Find communities which are below the small_community_size cutoff.

        Args:
            node_communities: (np.array) an array containing the community assignments for each
                node.
            small_community_size: (int) the maximum number of nodes in a community. Communities
                with less nodes than the small_community_size are considered to be
                small communities.

        Returns:
            communities: (np.array) an array where each element is a community object.
            community_ids_small: (np.array) an array of the community ids corresponding to the
                small communities.
        """
        communities = []
        community_ids = set(list(node_communities.flatten()))

        for community_id in community_ids:
            nodes = np.where(node_communities == community_id)[0]
            community_size = len(nodes)

            if community_size < small_community_size:
                community = Community(
                    id=community_id,
                    nodes=list(nodes)
                )
                communities.append(community)

        return communities

    def reassign_small_communities(
        self, node_communities, small_community_size, neighbor_array, exclude_neighbors_small=True
    ):
        """Move small communities into neighboring communities.

        Args:
            node_communities: (np.array) an array containing the community assignments for each
                node.
            small_community_size: (int) the maximum number of nodes in a community. Communities
                with less nodes than the small_community_size are considered to be
                small communities.
            neighbor_array: TODO.
            exclude_neighbors_small: (bool) If true, don't reassign nodes in small communities to
                other small communities. If false, nodes may be reassigned to other small
                communities which are larger than their current community.

        Returns:
            node_communities: (np.array) an array containing the community assignments for each
                node.
            small_community_exists: (bool) are there any small communities?
        """
        communities_small = self.get_small_communities(
            node_communities, small_community_size
        )
        if communities_small == []:
            small_community_exists = False
        else:
            small_community_exists = True

        for community in communities_small:
            for node in community.nodes:
                neighbors = neighbor_array[node]
                node_communities_neighbors = node_communities[neighbors]
                node_communities_neighbors = list(node_communities_neighbors.flatten())

                if exclude_neighbors_small:
                    community_ids_small = [community.id for community in communities_small]
                    community_ids_neighbors_available = set(node_communities_neighbors) - \
                        set(community_ids_small)
                else:
                    community_ids_neighbors_available = set(node_communities_neighbors)

                if len(community_ids_neighbors_available) > 0:
                    node_communities_neighbors_available = [
                        community_id for community_id in node_communities_neighbors if
                        community_id in list(community_ids_neighbors_available)
                    ]
                    best_group = max(
                        node_communities_neighbors_available,
                        key=node_communities_neighbors_available.count
                    )
                    node_communities[node] = best_group
        return node_communities, small_community_exists

    def run_parc_big(
        self, x_data, jac_std_threshold=0.3, jac_weighted_edges=True, small_community_size=10
    ):
        n_elements = x_data.shape[0]
        hnsw = self.make_knn_struct(too_big=True, big_cluster=x_data)
        if n_elements <= 10:
            print(f"Number of samples = {n_elements} is low; consider increasing the large_community_factor")
        if n_elements > self.knn:
            knnbig = self.knn
        else:
            knnbig = int(max(5, 0.2 * n_elements))

        neighbor_array, distance_array = hnsw.knn_query(x_data, k=knnbig)

        csr_array = self.prune_local(neighbor_array, distance_array)
        graph = self.prune_global(csr_array, jac_std_threshold, jac_weighted_edges)

        partition = self.get_leiden_partition(graph, jac_weighted_edges)

        node_communities = np.asarray(partition.membership)
        node_communities = np.reshape(node_communities, (n_elements, 1))
        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]

        node_communities, small_community_exists = self.reassign_small_communities(
            node_communities, small_community_size, neighbor_array, exclude_neighbors_small=True
        )

        time_start = time.time()
        print("Handling fragments")
        while (small_community_exists) & (time.time() - time_start < self.small_community_timeout):
            node_communities, small_community_exists = self.reassign_small_communities(
                node_communities, small_community_size, neighbor_array,
                exclude_neighbors_small=False
            )

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]

        return node_communities

    def run_parc(self):
        print(
            f"Input data has shape {self.x_data.shape[0]} (samples) x"
            f"{self.x_data.shape[1]} (features)"
        )
        time_start = time.time()

        x_data = self.x_data
        small_community_size = self.small_community_size

        n_elements = x_data.shape[0]

        if self.neighbor_graph is not None:
            csr_array = self.neighbor_graph
            neighbor_array = np.split(csr_array.indices, csr_array.indptr)[1:-1]
        else:
            knn_struct = self.get_knn_struct()
            neighbor_array, distance_array = knn_struct.knn_query(x_data, k=self.knn)
            csr_array = self.prune_local(neighbor_array, distance_array)

        graph = self.prune_global(csr_array, self.jac_std_global)

        print("Starting community detection...")
        partition = self.get_leiden_partition(graph, self.jac_weighted_edges)

        node_communities = np.asarray(partition.membership)
        node_communities = np.reshape(node_communities, (n_elements, 1))

        # Check if the 0th cluster is too big. This is always the largest cluster.
        too_big, big_cluster_indices, big_cluster_sizes = self.check_if_cluster_oversized(
            node_communities=node_communities, community_id=0
        )

        while too_big:

            x_data_big = x_data[big_cluster_indices, :]
            node_communities_big_cluster = self.run_parc_big(x_data_big)
            node_communities_big_cluster = node_communities_big_cluster + 100000

            for cluster_index, index in zip(big_cluster_indices, range(0, len(big_cluster_indices))):
                node_communities[cluster_index] = node_communities_big_cluster[index]

            node_communities = np.asarray(np.unique(
                list(node_communities.flatten()),
                return_inverse=True
            )[1])

            print(f"New set of labels: {set(node_communities)}")

            too_big, big_cluster_indices, big_cluster_sizes = self.get_next_big_community(
                node_communities, big_cluster_sizes
            )

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]

        node_communities, small_community_exists = self.reassign_small_communities(
            node_communities, small_community_size, neighbor_array, exclude_neighbors_small=True
        )

        time_start = time.time()
        while (small_community_exists) & ((time.time() - time_start) < self.small_community_timeout):
            node_communities, small_community_exists = self.reassign_small_communities(
                node_communities, small_community_size, neighbor_array,
                exclude_neighbors_small=False
            )

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
        node_communities = list(node_communities.flatten())

        self.y_data_pred = node_communities
        run_time = time.time() - time_start
        print(f"time elapsed: {run_time} seconds")
        self.compute_performance_metrics(run_time)
        return

    def compute_performance_metrics(self, run_time):
        if self.y_data_true is None:
            print("Unable to compute performance metrics, true target values not provided.")
            return

        targets = list(set(self.y_data_true))
        N = len(list(self.y_data_true))
        self.f1_accumulated = 0
        self.f1_mean = 0
        self.stats_df = pd.DataFrame({
            'jac_std_global': [self.jac_std_global],
            'dist_std_local': [self.dist_std_local],
            'runtime(s)': [run_time]
        })
        self.majority_truth_labels = []
        list_roc = []
        if len(targets) > 1:
            f1_accumulated = 0
            f1_acc_noweighting = 0
            for target in targets:
                print(f"target is {target}")
                vals_roc, majority_truth_labels, numclusters_targetval = accuracy(
                    self.y_data_true, self.y_data_pred, target=target
                )
                f1_current = vals_roc[1]
                print('target', target, 'has f1-score of %.2f' % (f1_current * 100))
                f1_accumulated = f1_accumulated + f1_current * (list(self.y_data_true).count(target)) / N
                f1_acc_noweighting = f1_acc_noweighting + f1_current

                list_roc.append(
                    [self.jac_std_global, self.dist_std_local, target] + vals_roc
                    + [numclusters_targetval] + [run_time]
                )

            f1_mean = f1_acc_noweighting / len(targets)
            print("f1-score (unweighted) mean %.2f" % (f1_mean * 100), '%')
            print('f1-score weighted (by population) %.2f' % (f1_accumulated * 100), '%')

            df_accuracy = pd.DataFrame(
                list_roc,
                columns=[
                    'jac_std_global', 'dist_std_local', 'onevsall-target', 'error rate',
                    'f1-score', 'tnr', 'fnr', 'tpr', 'fpr', 'precision', 'recall', 'num_groups',
                    'population of target', 'num clusters', 'clustering runtime'
                ]
            )

            self.f1_accumulated = f1_accumulated
            self.f1_mean = f1_mean
            self.stats_df = df_accuracy
            self.majority_truth_labels = majority_truth_labels

    def run_umap_hnsw(self, X_input, graph, n_components=2, alpha: float = 1.0,
                      negative_sample_rate: int = 5, gamma: float = 1.0, spread=1.0, min_dist=0.1,
                      n_epochs=0, init_pos="spectral", random_state_seed=1, densmap=False,
                      densmap_kwds={}, output_dens=False):
        """Perform a fuzzy simplicial set embedding, using a specified initialisation method and
        then minimizing the fuzzy set cross entropy between the 1-skeletons of the high and
        low-dimensional fuzzy simplicial sets.

        See `umap.umap_ simplicial_set_embedding
        <https://github.com/lmcinnes/umap/blob/master/umap/umap_.py>`__.

        Args:
            X_input: (array) an array containing the input data, with shape n_samples x n_features.
            graph: (array) the 1-skeleton of the high dimensional fuzzy simplicial set as
                represented by a graph for which we require a sparse matrix for the
                (weighted) adjacency matrix.
            n_components: (int) the dimensionality of the Euclidean space into which the data
                will be embedded.
            alpha: (float) the initial learning rate for stochastic gradient descent (SGD).
            negative_sample_rate: (int) the number of negative samples to select per positive
                sample in the optimization process. Increasing this value will result in
                greater repulsive force being applied, greater optimization cost, but
                slightly more accuracy.
            gamma: (float) weight to apply to negative samples.
            spread: (float) the upper range of the x-value to be used to fit a curve for the
                low-dimensional fuzzy simplicial complex construction. This curve should be
                close to an offset exponential decay. Must be greater than 0. See
                ``umap.umap_.find_ab_params``.
            min_dist: (float) See ``umap.umap_.find_ab_params``.
            n_epochs: (int) the number of training epochs to be used in optimizing the
                low-dimensional embedding. Larger values result in more accurate embeddings.
                If 0 is specified, a value will be selected based on the size of the input dataset
                (200 for large datasets, 500 for small).
            init_pos: (string) how to initialize the low-dimensional embedding. One of:
                `spectral`: use a spectral embedding of the fuzzy 1-skeleton
                `random`: assign initial embedding positions at random
                `pca`: use the first n components from Principal Component Analysis (PCA)
                    applied to the input data.
            random_state_seed: (int) an integer to pass as a seed for the Numpy RandomState.
            densmap: (bool) whether to use the density-augmented objective function to optimize
                the embedding according to the densMAP algorithm.
            densmap_kwds: (dict) keyword arguments to be used by the densMAP optimization.
            output_dens: (bool) whether to output local radii in the original data
                and the embedding.

        Returns:
            embedding: array of shape (n_samples, n_components)
                The optimized of ``graph`` into an ``n_components`` dimensional
                euclidean space.
        """

        a, b = find_ab_params(spread, min_dist)
        print(f"a: {a}, b: {b}, spread: {spread}, dist: {min_dist}")

        X_umap = simplicial_set_embedding(
            data=X_input,
            graph=graph,
            n_components=n_components,
            initial_alpha=alpha,
            a=a,
            b=b,
            n_epochs=0,
            metric_kwds={},
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            init=init_pos,
            random_state=np.random.RandomState(random_state_seed),
            metric="euclidean",
            verbose=1,
            densmap=densmap,
            densmap_kwds=densmap_kwds,
            output_dens=output_dens
        )
        return X_umap[0]
