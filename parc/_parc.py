import numpy as np
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg
import time
from parc.metrics import accuracy
from parc.logger import get_logger

logger = get_logger(__name__)


class PARC:
    def __init__(self, x_data, y_data_true=None, knn=30, n_iter_leiden=5, random_seed=42,
                 distance_metric="l2", n_threads=-1, hnsw_param_ef_construction=150,
                 neighbor_graph=None, knn_struct=None,
                 l2_std_factor=3, keep_all_local_dist=None,
                 jac_threshold_type="median", jac_std_factor=0.0, jac_weighted_edges=True,
                 resolution_parameter=1.0, partition_type="ModularityVP",
                 large_community_factor=0.4, small_community_size=10, small_community_timeout=15
                 ):
        """Phenotyping by Accelerated Refined Community-partitioning.

        Attributes:
            x_data (np.array): a Numpy array of the input x data, with dimensions
                (n_samples, n_features).
            y_data_true (np.array): a Numpy array of the true output y labels.
            y_data_pred (np.array): a Numpy array of the predicted output y labels.
            knn (int): the number of nearest neighbors k for the k-nearest neighbours algorithm.
                Larger k means more neighbors in a cluster and therefore less clusters.
            n_iter_leiden (int): the number of iterations for the Leiden algorithm.
            random_seed (int): the random seed to enable reproducible Leiden clustering.
            distance_metric (string): the distance metric to be used in the KNN algorithm:

                - ``l2``: Euclidean distance L^2 norm:

                  .. code-block:: python

                    d = sum((x_i - y_i)^2)
                - ``cosine``: cosine similarity

                  .. code-block:: python

                    d = 1.0 - sum(x_i*y_i) / sqrt(sum(x_i*x_i) * sum(y_i*y_i))
                - ``ip``: inner product distance

                  .. code-block:: python

                    d = 1.0 - sum(x_i*y_i)
            n_threads (int): the number of threads used in the KNN algorithm.
            hnsw_param_ef_construction (int): a higher value increases accuracy of index construction.
                Even for O(100 000) cells, 150-200 is adequate.
            neighbor_graph (Compressed Sparse Row Matrix): A sparse matrix with dimensions
                (n_samples, n_samples), containing the distances between nodes.
            knn_struct (hnswlib.Index): the HNSW index of the KNN graph on which we perform queries.
            l2_std_factor (float): The multiplier used in calculating the Euclidean distance threshold
                for the distance between two nodes during local pruning:

                .. code-block:: python

                    max_distance = np.mean(distances) + l2_std_factor * np.std(distances)

                Avoid setting both the ``jac_std_factor`` (global) and the ``l2_std_factor`` (local)
                to < 0.5 as this is very aggressive pruning.
                Higher ``l2_std_factor`` means more edges are kept.
            keep_all_local_dist (bool): whether or not to do local pruning.
                If None (default), set to ``True`` if the number of samples is > 300 000,
                and set to ``False`` otherwise.
            jac_threshold_type (str): One of "median" or "mean". Determines how the Jaccard similarity
                threshold is calculated during global pruning.
            jac_std_factor (float): The multiplier used in calculating the Jaccard similarity threshold
                for the similarity between two nodes during global pruning for
                ``jac_threshold_type = "mean"``:

                .. code-block:: python

                    threshold = np.mean(similarities) - jac_std_factor * np.std(similarities)

                Setting ``jac_std_factor = 0.15`` and ``jac_threshold_type="mean"``
                performs empirically similar to ``jac_threshold_type="median"``, which does not use
                the ``jac_std_factor``.
                Generally values between 0-1.5 are reasonable.
                Higher ``jac_std_factor`` means more edges are kept.
            jac_weighted_edges (bool): whether to partition using the weighted graph.
            resolution_parameter (float): the resolution parameter to be used in the Leiden algorithm.
                In order to change ``resolution_parameter``, we switch to ``RBVP``.
            partition_type (str): the partition type to be used in the Leiden algorithm:
                - ``ModularityVP``: ModularityVertexPartition, ``resolution_parameter=1``
                - ``RBVP``: RBConfigurationVP, Reichardt and Bornholdt’s Potts model. Note that this
                  is the same as ``ModularityVP`` when setting 𝛾 = 1 and normalising by 2m.
            small_community_size (int): the smallest population size to be considered a community.
            small_community_timeout (int): the maximum number of seconds trying to check an outlying
                small community.
        """
        self.y_data_true = y_data_true
        self.x_data = x_data
        self.y_data_pred = None
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden
        self.random_seed = random_seed
        self.distance_metric = distance_metric
        self.n_threads = n_threads
        self.hnsw_param_ef_construction = hnsw_param_ef_construction
        self.neighbor_graph = neighbor_graph
        self.knn_struct = knn_struct
        self.l2_std_factor = l2_std_factor
        self.jac_std_factor = jac_std_factor
        self.jac_threshold_type = jac_threshold_type
        self.jac_weighted_edges = jac_weighted_edges
        self.keep_all_local_dist = keep_all_local_dist
        self.large_community_factor = large_community_factor
        self.small_community_size = small_community_size
        self.small_community_timeout = small_community_timeout
        self.resolution_parameter = resolution_parameter
        self.partition_type = partition_type

    @property
    def x_data(self):
        return self._x_data

    @x_data.setter
    def x_data(self, x_data):
        self._x_data = x_data
        if self.y_data_true is None:
            self.y_data_true = [1] * x_data.shape[0]

    @property
    def knn_struct(self):
        return self._knn_struct

    @knn_struct.setter
    def knn_struct(self, knn_struct):
        if knn_struct is None and self.neighbor_graph is None:
            logger.message(f"knn_struct and neighbor_graph not provided, creating knn_struct.")
            knn_struct = self.make_knn_struct()
        self._knn_struct = knn_struct

    @property
    def keep_all_local_dist(self):
        return self._keep_all_local_dist

    @keep_all_local_dist.setter
    def keep_all_local_dist(self, keep_all_local_dist):
        if keep_all_local_dist is None:
            if self.x_data.shape[0] > 300000:
                logger.message(
                    f"Sample size is {self.x_data.shape[0]}, setting keep_all_local_dist "
                    f"to True so that local pruning will be skipped and algorithm will be faster."
                )
                keep_all_local_dist = True
            else:
                keep_all_local_dist = False

        self._keep_all_local_dist = keep_all_local_dist

    @property
    def partition_type(self):
        return self._partition_type

    @partition_type.setter
    def partition_type(self, partition_type):
        if self.resolution_parameter != 1:
            self._partition_type = "RBVP"
        else:
            self._partition_type = partition_type

    def make_knn_struct(self, is_large_community=False, big_cluster=None):
        if self.knn > 190:
            logger.message('consider using a lower K_in for KNN graph construction')
        ef_query = max(100, self.knn + 1)  # ef always should be >K. higher ef, more accurate query
        if is_large_community == False:
            num_dims = self.x_data.shape[1]
            n_elements = self.x_data.shape[0]
            p = hnswlib.Index(space=self.distance_metric, dim=num_dims)
            p.set_num_threads(self.n_threads)
            if n_elements < 10000:
                ef_query = min(n_elements - 10, 500)
                ef_construction = ef_query
            else:
                ef_construction = self.hnsw_param_ef_construction
            if (num_dims > 30) & (n_elements<=50000) :
                p.init_index(max_elements=n_elements, ef_construction=ef_construction,
                             M=48)  ## good for scRNA seq where dimensionality is high
            else:
                p.init_index(max_elements=n_elements, ef_construction=ef_construction, M=24 ) #30
            p.add_items(self.x_data)
        if is_large_community == True:
            num_dims = big_cluster.shape[1]
            n_elements = big_cluster.shape[0]
            p = hnswlib.Index(space='l2', dim=num_dims)
            p.init_index(max_elements=n_elements, ef_construction=200, M=30)
            p.add_items(big_cluster)
        p.set_ef(ef_query)  # ef should always be > k

        return p

    def create_knn_graph(self):
        """Create a full k-nearest neighbors graph using the HNSW algorithm.

        Returns:
            scipy.sparse.csr_matrix: A compressed sparse row matrix with dimensions
            (n_samples, n_samples), containing the pruned distances.
        """

        k_umap = 15

        # neighbors in array are not listed in in any order of proximity
        self.knn_struct.set_ef(k_umap + 1)
        neighbor_array, distance_array = self.knn_struct.knn_query(self.x_data, k=k_umap)

        row_list = []
        col_list = neighbor_array.flatten().tolist()
        n_neighbors = neighbor_array.shape[1]
        n_samples = neighbor_array.shape[0]

        row_list.extend(list(np.transpose(
            np.ones((n_neighbors, n_samples)) * range(0, n_samples)).flatten()
        ))

        row_min = np.min(distance_array, axis=1)
        row_sigma = np.std(distance_array, axis=1)

        distance_array = (distance_array - row_min[:, np.newaxis]) / row_sigma[:, np.newaxis]
        distance_array = np.sqrt(distance_array.flatten()) * -1

        weight_list = np.exp(distance_array)

        threshold = np.mean(weight_list) + 2 * np.std(weight_list)

        weight_list[weight_list >= threshold] = threshold

        weight_list = weight_list.tolist()

        graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                           shape=(n_samples, n_samples))

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
            neighbor_array (np.array): An array with dimensions (n_samples, k) listing the
                k nearest neighbors for each data point.
            neighbor_array (np.array): An array with dimensions (n_samples, k) listing the
                distances to each of the k nearest neighbors for each data point.

        Returns:
            scipy.sparse.csr_matrix: A compressed sparse row matrix with dimensions
            (n_samples, n_samples), containing the pruned distances.
        """
        # neighbor array not listed in in any order of proximity
        row_list = []
        col_list = []
        weight_list = []

        n_neighbors = neighbor_array.shape[1]
        n_samples = neighbor_array.shape[0]
        discard_count = 0
        if self.keep_all_local_dist:
            logger.message("Skipping local pruning")
            row_list.extend(list(np.transpose(
                np.ones((n_neighbors, n_samples)) * range(0, n_samples)).flatten()
            ))
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1. / (distance_array.flatten() + 0.1)).tolist()

        else:
            logger.message(
                f"Starting local pruning based on Euclidean (L2) distance metric at "
                f"{self.l2_std_factor} standard deviations above mean"
            )
            distance_array = distance_array + 0.1
            for neighbors, sample_index in zip(neighbor_array, range(n_samples)):
                distances = distance_array[sample_index, :]
                max_distance = np.mean(distances) + self.l2_std_factor * np.std(distances)
                to_keep = np.where(distances < max_distance)[0]
                updated_neighbors = neighbors[np.ix_(to_keep)]
                updated_distances = distances[np.ix_(to_keep)]
                discard_count = discard_count + (n_neighbors - len(to_keep))

                for index in range(len(updated_neighbors)):
                    if sample_index != neighbors[index]:  # remove self-loops
                        row_list.append(sample_index)
                        col_list.append(updated_neighbors[index])
                        dist = np.sqrt(updated_distances[index])
                        weight_list.append(1 / (dist + 0.1))

        csr_graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                               shape=(n_samples, n_samples))
        return csr_graph

    def prune_global(self, csr_array, jac_std_factor, jac_threshold_type, jac_weighted_edges=True):
        """Prune the graph globally based on the Jaccard similarity measure.

        The ``csr_array`` contains the locally-pruned pairwise distances. From this, we can
        use the Jaccard similarity metric to compute the similarity score for each edge. We then
        remove any edges from the graph that do not meet a minimum similarity threshold.

        Args:
            csr_array (Compressed Sparse Row Matrix): A sparse matrix with dimensions
                (n_samples, n_samples), containing the locally-pruned pair-wise distances.

        Returns:
            igraph.Graph: a Graph object which has now been locally and globally pruned.
        """

        input_nodes, output_nodes = csr_array.nonzero()
        edges = list(zip(input_nodes, output_nodes))
        edges_copy = np.asarray(edges.copy())

        graph = ig.Graph(edges, edge_attrs={'weight': csr_array.data.tolist()})

        similarities = np.asarray(graph.similarity_jaccard(pairs=list(edges_copy)))

        logger.message("Starting global pruning")

        if jac_threshold_type == "median":
            threshold = np.median(similarities)
        else:
            threshold = np.mean(similarities) - jac_std_factor * np.std(similarities)

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
            logger.message(
                "Leiden algorithm find partition: partition type = ModularityVertexPartition"
            )
            partition = leidenalg.find_partition(
                graph=graph,
                partition_type=leidenalg.ModularityVertexPartition, weights=weights,
                n_iterations=self.n_iter_leiden, seed=self.random_seed
            )
        else:
            logger.message(
                "Leiden algorithm find partition: partition type = RBConfigurationVertexPartition"
            )
            partition = leidenalg.find_partition(
                graph=graph,
                partition_type=leidenalg.RBConfigurationVertexPartition, weights=weights,
                n_iterations=self.n_iter_leiden, seed=self.random_seed,
                resolution_parameter=self.resolution_parameter
            )
        return partition

    def check_if_large_community(self, node_communities, community_id, large_community_sizes=[]):
        """Check if the community size is greater than the max community size.

        Args:
            node_communities (np.array): an array containing the community assignments for each
                node.
            community_id (int): the integer id of the community.

        Returns:
            (bool, int, list): a 3-tuple consisting of:

                - ``is_large_community``: whether or not the community is too large.
                - ``community_size``: the size of the community.
                - ``large_community_indices``: a list of node indices for the community,
                  if it is too large, otherwise an empty array.

        """
        is_large_community = False
        n_samples = node_communities.shape[0]
        community_indices = np.where(node_communities == community_id)[0]
        community_size = len(community_indices)
        large_community_indices = []
        not_yet_expanded = community_size not in large_community_sizes
        logger.info(f"Community has already been expanded? {not not_yet_expanded}")
        logger.info(
            f"Community size: {community_size}, large_community_factor: {self.large_community_factor}, "
            f"n_samples: {n_samples}, large_community_sizes: {large_community_sizes}"
        )
        if community_size > self.large_community_factor * n_samples and not_yet_expanded:
            is_large_community = True
            large_community_indices = community_indices
            logger.message(
                f"Community {community_id} is too big, community size = {community_size}, "
                f"large_community_factor = {self.large_community_factor}. It will be expanded."
            )
        return is_large_community, community_size, large_community_indices

    def reassign_large_communities(self, x_data, node_communities):
        """Find communities which are above the large_community_factor cutoff and split them.

        1. Check if the 0th community is too large. Since this is always the largest community,
           if it does not meet the threshold for a large community then all the other
           communities will also not be too large.
        2. If the 0th community is too large, then iterate through the rest of the communities. For
           each large community, run the PARC algorithm (but don't check for large communities).
           Reassign communities to split communities.

        Args:
            x_data (np.array): an array containing the input data, with shape
                (n_samples x n_features).
            node_communities (np.array): an array containing the community assignments for each
                node.

        Returns:
            np.array: an array containing the new community assignments for each node.
        """
        # the 0th community is the largest community, so we check this first
        logger.message(
            f"Checking size of Community 0, large_community_factor={self.large_community_factor}"
        )
        node_communities = np.copy(node_communities)
        large_community_exists, community_size, large_community_indices = self.check_if_large_community(
            node_communities=node_communities, community_id=0
        )
        if large_community_exists:
            large_community_sizes = [community_size]
        logger.message(f"Community 0 is too large? {large_community_exists}")

        while large_community_exists:

            node_communities_large_community = self.run_toobig_subPARC(
                x_data[large_community_indices, :]
            )
            node_communities_large_community = node_communities_large_community + 100000

            logger.message(f"shape node_communities: {node_communities.shape}")

            for community_index, index in zip(large_community_indices, range(len(large_community_indices))):
                node_communities[community_index] = node_communities_large_community[index]

            node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
            logger.message(f"new set of labels: {set(node_communities)}")
            large_community_exists = False
            node_communities = np.asarray(node_communities)

            for community_id in set(node_communities):
                large_community_exists, community_size, large_community_indices = self.check_if_large_community(
                    node_communities, community_id, large_community_sizes
                )
                if large_community_exists:
                    large_community_sizes.append(community_size)
                    logger.message(
                        f"Community {community_id} is too big with population {community_size}. "
                        f"It will be expanded."
                    )
                    break

        return node_communities

    def run_toobig_subPARC(self, x_data, jac_std_factor=0.3, jac_threshold_type="mean",
                           jac_weighted_edges=True):
        n_elements = x_data.shape[0]
        hnsw = self.make_knn_struct(is_large_community=True, big_cluster=x_data)
        if n_elements <= 10: logger.message('consider increasing the large_community_factor')
        if n_elements > self.knn:
            knn = self.knn
        else:
            knn = int(max(5, 0.2 * n_elements))

        neighbor_array, distance_array = hnsw.knn_query(x_data, k=knn)
        csr_array = self.prune_local(neighbor_array, distance_array)
        graph = self.prune_global(csr_array, jac_std_factor, jac_threshold_type)
        partition = self.get_leiden_partition(graph, jac_weighted_edges)

        node_communities = np.asarray(partition.membership)
        node_communities = np.reshape(node_communities, (n_elements, 1))
        communities_small = []
        community_ids_small = []
        small_community_exists = False
        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
        for community_id in set(node_communities):
            nodes = list(np.where(node_communities == community_id)[0])
            community_size = len(nodes)
            if community_size < 10:
                small_community_exists = True
                communities_small.append(nodes)
                community_ids_small.append(community_id)

        for small_cluster in communities_small:
            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell, :]
                group_of_old_neighbors = node_communities[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(community_ids_small)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    node_communities[single_cell] = best_group

        time_smallpop_start = time.time()
        logger.message('handling fragments')
        while small_community_exists & (time.time() - time_smallpop_start < self.small_community_timeout):
            communities_small = []
            small_community_exists = False
            for community_id in set(list(node_communities.flatten())):
                nodes = list(np.where(node_communities == community_id)[0])
                community_size = len(nodes)
                if community_size < 10:
                    small_community_exists = True
                    communities_small.append(nodes)
            for small_cluster in communities_small:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = node_communities[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    node_communities[single_cell] = best_group

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]

        return node_communities

    def run_parc(self):
        logger.message(
            f"Input data has shape {self.x_data.shape[0]} (samples) x "
            f"{self.x_data.shape[1]} (features)"
        )

        start_time = time.time()

        x_data = self.x_data
        large_community_factor = self.large_community_factor
        small_community_size = self.small_community_size
        jac_std_factor = self.jac_std_factor
        jac_weighted_edges = self.jac_weighted_edges
        n_elements = x_data.shape[0]

        if self.neighbor_graph is None:
            neighbor_array, distance_array = self.knn_struct.knn_query(x_data, k=self.knn)
            csr_array = self.prune_local(neighbor_array, distance_array)
        else:
            csr_array = self.neighbor_graph
            neighbor_array = np.split(csr_array.indices, csr_array.indptr)[1:-1]

        graph = self.prune_global(csr_array, self.jac_std_factor, self.jac_threshold_type)

        logger.message("Starting community detection")
        partition = self.get_leiden_partition(graph, self.jac_weighted_edges)
        time_end_PARC = time.time()
        node_communities = np.asarray(partition.membership)
        node_communities = np.reshape(node_communities, (n_elements, 1))

        node_communities = self.reassign_large_communities(x_data, node_communities)

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
        communities_small = []
        community_ids_small = []
        small_community_exists = False

        for community_id in set(node_communities):
            nodes = list(np.where(node_communities == community_id)[0])
            community_size = len(nodes)

            if community_size < small_community_size:
                small_community_exists = True
                communities_small.append(nodes)
                community_ids_small.append(community_id)

        for small_cluster in communities_small:

            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell]
                group_of_old_neighbors = node_communities[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(community_ids_small)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    node_communities[single_cell] = best_group
        time_smallpop_start = time.time()
        while small_community_exists & ((time.time() - time_smallpop_start) < self.small_community_timeout):
            communities_small = []
            small_community_exists = False
            for community_id in set(list(node_communities.flatten())):
                nodes = list(np.where(node_communities == community_id)[0])
                community_size = len(nodes)
                if community_size < small_community_size:
                    small_community_exists = True
                    communities_small.append(nodes)
            for small_cluster in communities_small:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell]
                    group_of_old_neighbors = node_communities[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    node_communities[single_cell] = best_group

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
        node_communities = list(node_communities.flatten())

        self.y_data_pred = node_communities
        end_time = time.time()
        run_time = end_time - start_time
        logger.message(f"Time elapsed: {run_time} seconds")
        self.compute_performance_metrics(run_time)

    def compute_performance_metrics(self, run_time):

        targets = list(set(self.y_data_true))
        n_samples = len(list(self.y_data_true))
        self.f1_accumulated = 0
        self.f1_mean = 0
        self.stats_df = pd.DataFrame({
            'jac_std_factor': [self.jac_std_factor],
            'l2_std_factor': [self.l2_std_factor],
            'runtime(s)': [run_time]
        })
        self.majority_truth_labels = []
        list_roc = []
        if len(targets) > 1:
            f1_accumulated = 0
            f1_acc_noweighting = 0
            for target in targets:
                vals_roc, majority_truth_labels, numclusters_targetval = accuracy(
                    self.y_data_true, self.y_data_pred, target
                )
                f1_current = vals_roc[1]
                logger.message(f"target {target} has f1-score of {np.round(f1_current * 100, 2)}")
                f1_accumulated += f1_current * (list(self.y_data_true).count(target)) / n_samples
                f1_acc_noweighting = f1_acc_noweighting + f1_current

                list_roc.append(
                    [self.jac_std_factor, self.l2_std_factor, target] +
                    vals_roc +
                    [numclusters_targetval] +
                    [run_time]
                )

            f1_mean = f1_acc_noweighting / len(targets)
            logger.message(f"f1-score (unweighted) mean {np.round(f1_mean * 100, 2)}")
            logger.message(f"f1-score weighted (by population) {np.round(f1_accumulated * 100, 2)}")

            df_accuracy = pd.DataFrame(
                list_roc,
                columns=['jac_std_factor', 'l2_std_factor', 'onevsall-target', 'error rate',
                        'f1-score', 'tnr', 'fnr',
                        'tpr', 'fpr', 'precision', 'recall', 'num_groups',
                        'population of target', 'num clusters', 'clustering runtime']
            )

            self.f1_accumulated = f1_accumulated
            self.f1_mean = f1_mean
            self.stats_df = df_accuracy
            self.majority_truth_labels = majority_truth_labels
        return
