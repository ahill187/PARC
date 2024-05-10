import numpy as np
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
from progress.bar import Bar
import igraph as ig
import leidenalg
import time
from umap.umap_ import find_ab_params, simplicial_set_embedding
from parc.logger import get_logger

logger = get_logger(__name__)


class PARC:
    def __init__(self, x_data, y_data_true=None, knn=30, n_iter_leiden=5, random_seed=42,
                 distance_metric="l2", n_threads=-1, hnsw_param_ef_construction=150,
                 neighbor_graph=None, knn_struct=None,
                 l2_std_factor=3, keep_all_local_dist=None,
                 jac_threshold_type="median", jac_std_factor=0.15, jac_weighted_edges=True,
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
            jac_threshold_type (str): One of ``"median"`` or ``"mean"``. Determines how the
                Jaccard similarity threshold is calculated during global pruning.
            jac_std_factor (float): The multiplier used in calculating the Jaccard similarity
                threshold for the similarity between two nodes during global pruning for
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

            large_community_factor (float): A factor used to determine if a community is too large.
                If the community size is greater than ``large_community_factor * n_samples``,
                then the community is too large and the PARC algorithm will be run on the single
                community to split it up. The default value of ``0.4`` ensures that all communities
                will be less than the cutoff size.
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

    def make_knn_struct(self, too_big=False, big_cluster=None):
        if self.knn > 190:
            logger.message(f"knn = {self.knn}; consider using a lower k for KNN graph construction")
        ef_query = max(100, self.knn + 1)  # ef always should be >K. higher ef, more accurate query
        if not too_big:
            num_dims = self.x_data.shape[1]
            n_elements = self.x_data.shape[0]
            p = hnswlib.Index(space=self.distance_metric, dim=num_dims)
            p.set_num_threads(self.n_threads)  # allow user to set threads used in KNN construction
            if n_elements < 10000:
                ef_query = min(n_elements - 10, 500)
                ef_construction = ef_query
            else:
                ef_construction = self.hnsw_param_ef_construction
            if (num_dims > 30) & (n_elements <= 50000) :
                p.init_index(
                    max_elements=n_elements, ef_construction=ef_construction, M=48
                ) # good for scRNA seq where dimensionality is high
            else:
                p.init_index(max_elements=n_elements, ef_construction=ef_construction, M=24) #30
            p.add_items(self.x_data)
        if too_big:
            num_dims = big_cluster.shape[1]
            n_elements = big_cluster.shape[0]
            p = hnswlib.Index(space='l2', dim=num_dims)
            p.init_index(max_elements=n_elements, ef_construction=200, M=30)
            p.add_items(big_cluster)
        p.set_ef(ef_query)  # ef should always be > k

        return p

    def knngraph_full(self):
        k_umap = 15
        # neighbors in array are not listed in in any order of proximity
        self.knn_struct.set_ef(k_umap+1)
        neighbor_array, distance_array = self.knn_struct.knn_query(self.x_data, k=k_umap)

        row_list = []
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]

        row_list.extend(
            list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten())
        )


        row_min = np.min(distance_array, axis=1)
        row_sigma = np.std(distance_array, axis=1)

        distance_array = (distance_array - row_min[:,np.newaxis])/row_sigma[:,np.newaxis]

        col_list = neighbor_array.flatten().tolist()
        distance_array = distance_array.flatten()
        distance_array = np.sqrt(distance_array)
        distance_array = distance_array * -1

        weight_list = np.exp(distance_array)


        threshold = np.mean(weight_list) + 2* np.std(weight_list)

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
            neighbor_array (np.array): An array with dimensions (n_samples, k) listing the
                k nearest neighbors for each data point.
            distance_array (np.array): An array with dimensions (n_samples, k) listing the
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
        n_cells = neighbor_array.shape[0]
        rowi = 0
        discard_count = 0
        if not self.keep_all_local_dist:  # locally prune based on (squared) l2 distance

            logger.message(
                f"Starting local pruning based on Euclidean (L2) distance metric at "
                f"{self.l2_std_factor} standard deviations above mean"
            )
            distance_array = distance_array + 0.1
            bar = Bar("Local pruning...", max=n_cells)
            for row in neighbor_array:
                distlist = distance_array[rowi, :]
                to_keep = np.where(
                    distlist < np.mean(distlist) + self.l2_std_factor * np.std(distlist)
                )[0]  # 0*std
                updated_nn_ind = row[np.ix_(to_keep)]
                updated_nn_weights = distlist[np.ix_(to_keep)]
                discard_count = discard_count + (n_neighbors - len(to_keep))

                for ik in range(len(updated_nn_ind)):
                    if rowi != row[ik]:  # remove self-loops
                        row_list.append(rowi)
                        col_list.append(updated_nn_ind[ik])
                        dist = np.sqrt(updated_nn_weights[ik])
                        weight_list.append(1/(dist+0.1))

                rowi = rowi + 1
                bar.next()
            bar.finish()

        if self.keep_all_local_dist:  # dont prune based on distance
            row_list.extend(
                list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten())
            )
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1. / (distance_array.flatten() + 0.1)).tolist()

        csr_graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                               shape=(n_cells, n_cells))
        return csr_graph

    def func_mode(self, ll):  # return MODE of list
        # If multiple items are maximal, the function returns the first one encountered.
        return max(set(ll), key=ll.count)

    def run_toobig_subPARC(self, x_data, jac_std_factor=0.3, jac_threshold_type="mean",
                           jac_weighted_edges=True):
        n_elements = x_data.shape[0]
        hnsw = self.make_knn_struct(too_big=True, big_cluster=x_data)
        if n_elements <= 10:
            logger.message(
                f"Number of samples = {n_samples}, consider increasing the large_community_factor"
            )
        if n_elements > self.knn:
            knnbig = self.knn
        else:
            knnbig = int(max(5, 0.2 * n_elements))

        neighbor_array, distance_array = hnsw.knn_query(x_data, k=knnbig)
        csr_array = self.prune_local(neighbor_array, distance_array)
        input_nodes, output_nodes = csr_array.nonzero()

        edges = list(zip(input_nodes.tolist(), output_nodes.tolist()))
        edges_copy = edges.copy()
        G = ig.Graph(edges, edge_attrs={'weight': csr_array.data.tolist()})
        similarities = G.similarity_jaccard(pairs=edges_copy)  # list of jaccard weights
        new_edgelist = []
        sim_list_array = np.asarray(similarities)
        if jac_threshold_type == "median":
            threshold = np.median(similarities)
        else:
            threshold = np.mean(similarities) - jac_std_factor * np.std(similarities)

        indices_similar = np.where(sim_list_array > threshold)[0]
        for ii in indices_similar: new_edgelist.append(edges_copy[ii])
        sim_list_new = list(sim_list_array[indices_similar])

        if jac_weighted_edges:
            graph_pruned = ig.Graph(
                n=n_elements, edges=list(new_edgelist), edge_attrs={'weight': sim_list_new}
            )
        else:
            graph_pruned = ig.Graph(n=n_elements, edges=list(new_edgelist))
        graph_pruned.simplify(combine_edges='sum')
        if jac_weighted_edges:
            if self.partition_type =='ModularityVP':
                logger.message(
                    "Leiden algorithm find partition: partition type = ModularityVertexPartition"
                )
                partition = leidenalg.find_partition(
                    graph_pruned, leidenalg.ModularityVertexPartition, weights='weight',
                    n_iterations=self.n_iter_leiden, seed=self.random_seed
                )
            else:
                partition = leidenalg.find_partition(
                    graph_pruned, leidenalg.RBConfigurationVertexPartition, weights='weight',
                    n_iterations=self.n_iter_leiden, seed=self.random_seed,
                    resolution_parameter=self.resolution_parameter
                )
                logger.message(
                    "Leiden algorithm find partition: partition type = RBConfigurationVertexPartition"
                )
        else:
            if self.partition_type == 'ModularityVP':
                logger.message(
                    "Leiden algorithm find partition: partition type = ModularityVertexPartition"
                )
                partition = leidenalg.find_partition(
                    graph_pruned, leidenalg.ModularityVertexPartition,
                    n_iterations=self.n_iter_leiden, seed=self.random_seed
                )
            else:
                logger.message(
                    "Leiden algorithm find partition: partition type = RBConfigurationVertexPartition"
                )
                partition = leidenalg.find_partition(
                    graph_pruned, leidenalg.RBConfigurationVertexPartition,
                    n_iterations=self.n_iter_leiden, seed=self.random_seed,
                    resolution_parameter=self.resolution_parameter
                )

        node_communities = np.asarray(partition.membership)
        node_communities = np.reshape(node_communities, (n_elements, 1))
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False
        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
        for cluster in set(node_communities):
            population = len(np.where(node_communities == cluster)[0])
            if population < 10:
                small_pop_exist = True
                small_pop_list.append(list(np.where(node_communities == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:
            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell, :]
                group_of_old_neighbors = node_communities[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    node_communities[single_cell] = best_group

        time_smallpop_start = time.time()

        while (small_pop_exist) & (time.time() - time_smallpop_start < self.small_community_timeout):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(node_communities.flatten())):
                population = len(np.where(node_communities == cluster)[0])
                if population < 10:
                    small_pop_exist = True

                    small_pop_list.append(np.where(node_communities == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = node_communities[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    node_communities[single_cell] = best_group

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]

        return node_communities

    def run_subPARC(self):


        x_data = self.x_data
        large_community_factor = self.large_community_factor
        small_community_size = self.small_community_size
        jac_std_factor = self.jac_std_factor
        jac_threshold_type = self.jac_threshold_type
        jac_weighted_edges = self.jac_weighted_edges
        knn = self.knn
        n_elements = x_data.shape[0]


        if self.neighbor_graph is not None:
            csr_array = self.neighbor_graph
            neighbor_array = np.split(csr_array.indices, csr_array.indptr)[1:-1]
        else:
            if self.knn_struct is None:
                logger.info('knn struct was not available, so making one')
                self.knn_struct = self.make_knn_struct()
            else:
                logger.info("knn struct already exists")
            neighbor_array, distance_array = self.knn_struct.knn_query(x_data, k=knn)
            csr_array = self.prune_local(neighbor_array, distance_array)

        input_nodes, output_nodes = csr_array.nonzero()

        edges = list(zip(input_nodes, output_nodes))

        edges_copy = edges.copy()

        graph = ig.Graph(edges, edge_attrs={'weight': csr_array.data.tolist()})
        similarities = graph.similarity_jaccard(pairs=edges_copy)

        logger.message("Starting global pruning...")

        sim_list_array = np.asarray(similarities)
        edge_list_copy_array = np.asarray(edges_copy)

        if jac_threshold_type == "median":
            threshold = np.median(similarities)
        else:
            threshold = np.mean(similarities) - jac_std_factor * np.std(similarities)
        indices_similar = np.where(sim_list_array > threshold)[0]

        new_edgelist = list(edge_list_copy_array[indices_similar])
        sim_list_new = list(sim_list_array[indices_similar])

        graph_pruned = ig.Graph(
            n=n_elements, edges=list(new_edgelist), edge_attrs={'weight': sim_list_new}
        )

        graph_pruned.simplify(combine_edges='sum')  # "first"

        logger.message("Starting community detection")
        if jac_weighted_edges:
            if self.partition_type =='ModularityVP':
                logger.message(
                    "Leiden algorithm find partition: partition type = ModularityVertexPartition"
                )
                partition = leidenalg.find_partition(
                    graph_pruned, leidenalg.ModularityVertexPartition, weights='weight',
                    n_iterations=self.n_iter_leiden, seed=self.random_seed
                )
            else:
                logger.message(
                    "Leiden algorithm find partition: partition type = RBConfigurationVertexPartition"
                )
                partition = leidenalg.find_partition(
                    graph_pruned, leidenalg.RBConfigurationVertexPartition, weights='weight',
                    n_iterations=self.n_iter_leiden, seed=self.random_seed,
                    resolution_parameter = self.resolution_parameter
                )

        else:
            if self.partition_type == 'ModularityVP':
                logger.message(
                    "Leiden algorithm find partition: partition type = ModularityVertexPartition"
                )
                partition = leidenalg.find_partition(
                    graph_pruned, leidenalg.ModularityVertexPartition,
                    n_iterations=self.n_iter_leiden, seed=self.random_seed
                )
            else:
                logger.message(
                    "Leiden algorithm find partition: partition type = RBConfigurationVertexPartition"
                )
                partition = leidenalg.find_partition(
                    graph_pruned, leidenalg.RBConfigurationVertexPartition,
                    n_iterations=self.n_iter_leiden, seed=self.random_seed,
                    resolution_parameter = self.resolution_parameter
                )

        node_communities = np.asarray(partition.membership)
        node_communities = np.reshape(node_communities, (n_elements, 1))

        too_big = False

        # the 0th cluster is the largest one. so if cluster 0 is not too big,
        # then the others wont be too big either
        cluster_i_loc = np.where(node_communities == 0)[0]
        pop_i = len(cluster_i_loc)
        if pop_i > large_community_factor * n_elements:
            too_big = True
            cluster_big_loc = cluster_i_loc
            list_pop_too_bigs = [pop_i]
            cluster_too_big = 0

        while too_big:

            x_data_big = x_data[cluster_big_loc, :]
            node_communities_big = self.run_toobig_subPARC(x_data_big)
            node_communities_big = node_communities_big + 100000
            pop_list = []

            for item in set(list(node_communities_big.flatten())):
                pop_list.append([item, list(node_communities_big.flatten()).count(item)])

            jj = 0
            logger.info(f"shape node_communities: {node_communities.shape}")
            for j in cluster_big_loc:
                node_communities[j] = node_communities_big[jj]
                jj = jj + 1
            node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
            logger.info(f"new set of labels: {set(node_communities)}")
            too_big = False
            set_node_communities = set(node_communities)

            node_communities = np.asarray(node_communities)
            for cluster_ii in set_node_communities:
                cluster_ii_loc = np.where(node_communities == cluster_ii)[0]
                pop_ii = len(cluster_ii_loc)
                not_yet_expanded = pop_ii not in list_pop_too_bigs
                if pop_ii > large_community_factor * n_elements and not_yet_expanded:
                    too_big = True
                    logger.info(f"Cluster {cluster_ii} is too big and has population {pop_ii}.")
                    cluster_big_loc = cluster_ii_loc
                    cluster_big = cluster_ii
                    big_pop = pop_ii
            if too_big:
                list_pop_too_bigs.append(big_pop)
                logger.info(
                        f"Community {cluster_big} is too big with population {big_pop}. "
                        f"It will be expanded."
                )
        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False

        for cluster in set(node_communities):
            population = len(np.where(node_communities == cluster)[0])

            if population < small_community_size:
                small_pop_exist = True

                small_pop_list.append(list(np.where(node_communities == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:

            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell]
                group_of_old_neighbors = node_communities[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    node_communities[single_cell] = best_group
        time_smallpop_start = time.time()
        while (small_pop_exist) & ((time.time() - time_smallpop_start) < self.small_community_timeout):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(node_communities.flatten())):
                population = len(np.where(node_communities == cluster)[0])
                if population < small_community_size:
                    small_pop_exist = True
                    small_pop_list.append(np.where(node_communities == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell]
                    group_of_old_neighbors = node_communities[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    node_communities[single_cell] = best_group

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
        node_communities = list(node_communities.flatten())

        pop_list = []
        for item in set(node_communities):
            pop_list.append((item, node_communities.count(item)))

        self.y_data_pred = node_communities
        return

    def accuracy(self, onevsall=1):

        y_data_true = self.y_data_true
        Index_dict = {}
        y_data_pred = self.y_data_pred
        N = len(y_data_pred)
        n_cancer = list(y_data_true).count(onevsall)
        n_pbmc = N - n_cancer

        for k in range(N):
            Index_dict.setdefault(y_data_pred[k], []).append(y_data_true[k])
        num_groups = len(Index_dict)
        sorted_keys = list(sorted(Index_dict.keys()))
        error_count = []
        pbmc_labels = []
        thp1_labels = []
        fp, fn, tp, tn, precision, recall, f1_score = 0, 0, 0, 0, 0, 0, 0

        for kk in sorted_keys:
            vals = [t for t in Index_dict[kk]]
            majority_val = self.func_mode(vals)
            if majority_val == onevsall:
                logger.info(f"Cluster {kk} has majority {onevsall} with population {len(vals)}")
            if kk == -1:
                len_unknown = len(vals)
                logger.info(f"Number of unknown: {len_unknown}")
            if (majority_val == onevsall) and (kk != -1):
                thp1_labels.append(kk)
                fp = fp + len([e for e in vals if e != onevsall])
                tp = tp + len([e for e in vals if e == onevsall])
                list_error = [e for e in vals if e != majority_val]
                e_count = len(list_error)
                error_count.append(e_count)
            elif (majority_val != onevsall) and (kk != -1):
                pbmc_labels.append(kk)
                tn = tn + len([e for e in vals if e != onevsall])
                fn = fn + len([e for e in vals if e == onevsall])
                error_count.append(len([e for e in vals if e != majority_val]))

        predict_class_array = np.array(y_data_pred)
        y_data_pred_array = np.array(y_data_pred)
        number_clusters_for_target = len(thp1_labels)
        for cancer_class in thp1_labels:
            predict_class_array[y_data_pred_array == cancer_class] = 1
        for benign_class in pbmc_labels:
            predict_class_array[y_data_pred_array == benign_class] = 0
        predict_class_array.reshape((predict_class_array.shape[0], -1))
        error_rate = sum(error_count) / N
        n_target = tp + fn
        tnr = tn / n_pbmc
        fnr = fn / n_cancer
        tpr = tp / n_cancer
        fpr = fp / n_pbmc

        if tp != 0 or fn != 0:
            recall = tp / (tp + fn)  # ability to find all positives
        if tp != 0 or fp != 0:
            precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
        if precision != 0 or recall != 0:
            f1_score = precision * recall * 2 / (precision + recall)
        majority_truth_labels = np.empty((len(y_data_true), 1), dtype=object)

        for cluster_i in set(y_data_pred):
            cluster_i_loc = np.where(np.asarray(y_data_pred) == cluster_i)[0]
            y_data_true = np.asarray(y_data_true)
            majority_truth = self.func_mode(list(y_data_true[cluster_i_loc]))
            majority_truth_labels[cluster_i_loc] = majority_truth

        majority_truth_labels = list(majority_truth_labels.flatten())
        accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                        recall, num_groups, n_target]

        return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target

    def run_parc(self):
        logger.message(
            f"Input data has shape {self.x_data.shape[0]} (samples) x "
            f"{self.x_data.shape[1]} (features)"
        )

        if self.y_data_true is None:
            self.y_data_true = [1] * self.x_data.shape[0]
        list_roc = []

        time_start_total = time.time()

        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        self.run_subPARC()
        run_time = time.time() - time_start_total
        logger.message(f"Time elapsed: {run_time} seconds")

        targets = list(set(self.y_data_true))
        N = len(list(self.y_data_true))
        self.f1_accumulated = 0
        self.f1_mean = 0
        self.stats_df = pd.DataFrame({
            'jac_std_factor': [self.jac_std_factor],
            'l2_std_factor': [self.l2_std_factor],
            'runtime(s)': [run_time]
        })
        self.majority_truth_labels = []
        if len(targets) > 1:
            f1_accumulated = 0
            f1_acc_noweighting = 0
            for onevsall_val in targets:
                vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = self.accuracy(onevsall=onevsall_val)
                f1_current = vals_roc[1]
                logger.message(
                    f"target {onevsall_val} has f1-score of {np.round(f1_current * 100, 2)}"
                )
                f1_accumulated += f1_current * (list(self.y_data_true).count(onevsall_val)) / N
                f1_acc_noweighting = f1_acc_noweighting + f1_current

                list_roc.append(
                    [self.jac_std_factor, self.l2_std_factor, onevsall_val] +
                    vals_roc +
                    [numclusters_targetval] +
                    [run_time]
                )

            f1_mean = f1_acc_noweighting / len(targets)
            logger.message(f"f1-score (unweighted) mean {np.round(f1_mean * 100, 2)}")
            logger.message(f"f1-score weighted (by population) {np.round(f1_accumulated * 100, 2)}")

            df_accuracy = pd.DataFrame(
                list_roc,
                columns=[
                    'jac_std_factor', 'l2_std_factor', 'onevsall-target', 'error rate',
                    'f1-score', 'tnr', 'fnr', 'tpr', 'fpr', 'precision', 'recall', 'num_groups',
                    'population of target', 'num clusters', 'clustering runtime'
                ]
            )

            self.f1_accumulated = f1_accumulated
            self.f1_mean = f1_mean
            self.stats_df = df_accuracy
            self.majority_truth_labels = majority_truth_labels
        return

    def run_umap_hnsw(self, x_data, graph, n_components=2, alpha: float = 1.0,
                      negative_sample_rate: int = 5, gamma: float = 1.0, spread=1.0, min_dist=0.1,
                      n_epochs=0, init_pos="spectral", random_state_seed=1, densmap=False,
                      densmap_kwds={}, output_dens=False):
        """Perform a fuzzy simplicial set embedding, using a specified initialisation method and
        then minimizing the fuzzy set cross entropy between the 1-skeletons of the high and
        low-dimensional fuzzy simplicial sets.

        See `umap.umap_ simplicial_set_embedding
        <https://github.com/lmcinnes/umap/blob/master/umap/umap_.py>`__.

        Args:
            x_data: (array) an array containing the input data, with shape n_samples x n_features.
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
        logger.message(f"a: {a}, b: {b}, spread: {spread}, dist: {min_dist}")

        X_umap = simplicial_set_embedding(
            data=x_data,
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
