import numpy as np
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
from progress.bar import Bar
import igraph as ig
import leidenalg
import os
import psutil
import time
import multiprocessing as mp
from umap.umap_ import find_ab_params, simplicial_set_embedding
from parc.k_nearest_neighbors import get_distance_array, get_neighbor_array, get_edges
from parc.utils import get_mode, get_current_memory_usage, get_memory_prune_global
from parc.logger import get_logger

logger = get_logger(__name__)

process = psutil.Process(os.getpid())


class PARC:
    def __init__(self, x_data, y_data_true=None, knn=30, n_iter_leiden=5, random_seed=42,
                 distance_metric="l2", n_threads=-1, hnsw_param_ef_construction=150,
                 neighbor_graph=None, knn_struct=None,
                 l2_std_factor=3, max_samples_local_pruning=300000, keep_all_local_dist=None,
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
            max_samples_local_pruning (int): The maximum number of samples permitted for local
                pruning. If the number of samples is greater than this, ``keep_all_local_dist``
                will be set to ``True`` and local pruning will be skipped.
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
        self.x_data = x_data
        self.y_data_true = y_data_true
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
        self.max_samples_local_pruning = max_samples_local_pruning
        self.keep_all_local_dist = keep_all_local_dist
        self.large_community_factor = large_community_factor
        self.small_community_size = small_community_size
        self.small_community_timeout = small_community_timeout
        self.resolution_parameter = resolution_parameter
        self.partition_type = partition_type

    @property
    def y_data_true(self):
        return self._y_data_true

    @y_data_true.setter
    def y_data_true(self, y_data_true):
        if y_data_true is None:
            y_data_true = [1] * self.x_data.shape[0]
        self._y_data_true = y_data_true

    @property
    def n_threads(self):
        return self._n_threads

    @n_threads.setter
    def n_threads(self, n_threads):
        if n_threads <= 0:
            n_threads = mp.cpu_count() - 1
        self._n_threads = n_threads

    @property
    def keep_all_local_dist(self):
        return self._keep_all_local_dist

    @keep_all_local_dist.setter
    def keep_all_local_dist(self, keep_all_local_dist):
        if keep_all_local_dist is None:
            if self.x_data.shape[0] > self.max_samples_local_pruning:
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

    def make_knn_struct(
        self, x_data, knn=None, distance_metric=None, hnsw_param_m=None,
        hnsw_param_ef_construction=None
    ):
        """Create a Hierarchical Navigable Small Worlds (HNSW) graph.

        See `hnswlib.Index
        <https://github.com/nmslib/hnswlib/blob/master/python_bindings/LazyIndex.py>`__.

        Args:
            x_data (np.array): a Numpy array of the input x data, with dimensions
                (n_samples, n_features).
            knn (int): the number of nearest neighbors k for the k-nearest neighbours algorithm.
                Larger k means more neighbors in a cluster and therefore less clusters.
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

            hnsw_param_ef_construction (int): (optional) The ``ef_construction`` parameter to be
                used in creating the ``hnswlib.Index`` object. A higher value increases accuracy of
                index construction. Even for ``O(100 000)`` cells, 150-200 is adequate.
            hnsw_param_m (int): (optional) The ``m`` parameter to be used in creating the
                ``hnswlib.Index`` object.

        Returns:
            hnswlib.Index: An HNSW object containing the k-nearest neighbours graph.
        """

        if knn is None:
            knn = self.knn

        if knn > 190:
            logger.message(f"knn = {knn}; consider using a lower k for KNN graph construction")

        if distance_metric is None:
            distance_metric = self.distance_metric

        n_features = x_data.shape[1]
        n_samples = x_data.shape[0]

        ef_query = max(100, knn + 1)  # ef always should be >K. higher ef, more accurate query
        if hnsw_param_ef_construction is None:
            if n_samples < 10000:
                ef_query = min(n_samples - 10, 500)
                hnsw_param_ef_construction = ef_query
            else:
                hnsw_param_ef_construction = self.hnsw_param_ef_construction

        if hnsw_param_m is None:
            if (n_features > 30) & (n_samples <= 50000):
                hnsw_param_m = 48 # good for scRNA seq where dimensionality is high
            else:
                hnsw_param_m = 24

        knn_struct = hnswlib.Index(space=distance_metric, dim=n_features)
        knn_struct.set_num_threads(self.n_threads)

        logger.info("Initializing HNSW index...")
        knn_struct.init_index(
            max_elements=n_samples, ef_construction=hnsw_param_ef_construction, M=hnsw_param_m
        )
        knn_struct.add_items(x_data)
        knn_struct.set_ef(ef_query)  # ef should always be > k

        return knn_struct

    def create_knn_graph(self):
        """Create a full k-nearest neighbors graph using the HNSW algorithm.

        Returns:
            scipy.sparse.csr_matrix: A compressed sparse row matrix with dimensions
                (n_samples, n_samples), containing the pruned distances.
        """
        k_umap = 15
        # neighbors in array are not listed in in any order of proximity
        self.knn_struct.set_ef(k_umap+1)
        neighbor_array, distance_array = self.knn_struct.knn_query(self.x_data, k=k_umap)

        row_list = []
        n_neighbors = neighbor_array.shape[1]
        n_samples = neighbor_array.shape[0]

        row_list.extend(
            list(np.transpose(np.ones((n_neighbors, n_samples)) * range(0, n_samples)).flatten())
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
                           shape=(n_samples, n_samples))

        graph_transpose = graph.T
        prod_matrix = graph.multiply(graph_transpose)

        graph = graph_transpose + graph - prod_matrix
        return graph

    def get_neighbor_distance_arrays(
        self, x_data, knn, distance_metric, create_new=False, hnsw_param_m=None,
        hnsw_param_ef_construction=None
    ):
        """Get the neighbor and distance arrays.

        Args:
            x_data (np.array): a Numpy array of the input x data, with dimensions
                (n_samples, n_features).
            knn (int): the number of nearest neighbors k for the k-nearest neighbours algorithm.
                Larger k means more neighbors in a cluster and therefore less clusters.
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

            create_new (bool): If ``False``, check to see if the ``FlowData`` object already
                contains a ``csr_array`` or a ``knn_struct``, and use these to create the
                neighbor and distance arrays. If ``True``, create a new ``csr_array`` and
                ``knn_struct``, even if they exist.
            hnsw_param_ef_construction (int): (optional) The ``ef_construction`` parameter to be
                used in creating the ``hnswlib.Index`` object. A higher value increases accuracy of
                index construction. Even for ``O(100 000)`` cells, 150-200 is adequate.
            hnsw_param_m (int): (optional) The ``m`` parameter to be used in creating the
                ``hnswlib.Index`` object.

        Returns:
            (np.array, np.array): A tuple of the ``neighbor_array`` and the ``distance_array``:

                - ``neighbor_array``: An array with dimensions (n_samples, k) listing the
                  k nearest neighbors for each data point.
                - ``distance_array``: An array with dimensions (n_samples, k) listing the
                  distances to each of the k nearest neighbors for each data point.
        """
        if self.neighbor_graph is not None and not create_new:
            csr_array = self.neighbor_graph
            neighbor_array = get_neighbor_array(csr_array)
            distance_array = get_distance_array(csr_array)
        else:
            if self.knn_struct is None or create_new:
                logger.message("Creating HNSW knn_struct")
                self.knn_struct = self.make_knn_struct(
                    x_data=x_data, knn=knn, distance_metric=distance_metric,
                    hnsw_param_m=hnsw_param_m, hnsw_param_ef_construction=hnsw_param_ef_construction
                )
            neighbor_array, distance_array = self.knn_struct.knn_query(x_data, k=knn)
        return neighbor_array, distance_array

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
        n_samples = neighbor_array.shape[0]
        discard_count = 0

        if self.keep_all_local_dist:
            logger.message(
                f"keep_all_local_dist set to {self.keep_all_local_dist}, skipping local pruning."
            )
            row_list.extend(
                list(np.transpose(np.ones((n_neighbors, n_samples)) * range(0, n_samples)).flatten())
            )
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1. / (distance_array.flatten() + 0.1)).tolist()

        else:
            logger.message(
                f"Starting local pruning based on Euclidean (L2) distance metric at "
                f"{self.l2_std_factor} standard deviations above mean"
            )
            distance_array = distance_array + 0.1
            bar = Bar("Local pruning...", max=n_samples)
            for sample_index, neighbors in zip(range(n_samples), neighbor_array):
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

                bar.next()
            bar.finish()

        csr_graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                               shape=(n_samples, n_samples))
        return csr_graph

    def prune_global(
        self, csr_array, jac_threshold_type, jac_std_factor, jac_weighted_edges, n_samples
    ):
        """Prune the graph globally based on the Jaccard similarity measure.

        The ``csr_array`` contains the locally-pruned pairwise distances. From this, we can
        use the Jaccard similarity metric to compute the similarity score for each edge. We then
        remove any edges from the graph that do not meet a minimum similarity threshold.

        Args:
            csr_array (Compressed Sparse Row Matrix): A sparse matrix with dimensions
                (n_samples, n_samples), containing the locally-pruned pair-wise distances.
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
            jac_weighted_edges (bool): whether to weight the pruned graph. This is always True for
                the top-level PARC run, but can be changed when pruning the large communities.
            n_samples (int): The number of samples in the data.

        Returns:
            igraph.Graph: a ``Graph`` object which has now been locally and globally pruned.
        """
        memory_prune_global = get_memory_prune_global(len(get_edges(csr_array)))
        current_usage = memory_prune_global['total'] - memory_prune_global['available']

        if not memory_prune_global["is_sufficient"]:
            raise MemoryError(
                f"""
                Not enough memory to perform global pruning.
                available memory: {memory_prune_global['available']} GiB
                required memory: {memory_prune_global['required']} GiB
                You can either:
                a) free up memory on your computer by closing other processes; current usage:
                {current_usage} GiB out of {memory_prune_global['total']} GiB on your computer,
                b) reduce the number of k-nearest neighbours, which is currently set to {self.knn},
                c) increase the max_samples_local_pruning parameter, and set keep_all_local_dist
                to False, so that local pruning will reduce the number of edges, or
                d) reduce the number of samples in your data, which is currenty set to {n_samples}.
                """
            )

        edges = get_edges(csr_array)
        logger.info(f"Creating graph with {len(edges)} edges and {n_samples} nodes...")

        graph = ig.Graph(edges, edge_attrs={'weight': csr_array.data.tolist()})
        del csr_array

        similarities = np.asarray(graph.similarity_jaccard(pairs=list(edges)))
        del graph

        logger.message("Starting global pruning...")

        if jac_threshold_type == "median":
            threshold = np.median(similarities)
        else:
            threshold = np.mean(similarities) - jac_std_factor * np.std(similarities)

        indices_similar = np.where(similarities > threshold)[0]

        logger.message(f"Creating graph with {len(edges)} edges and {n_samples} nodes...")
        if jac_weighted_edges:
            graph_pruned = ig.Graph(
                n=n_samples,
                edges=list(np.asarray(edges)[indices_similar]),
                edge_attrs={"weight": list(similarities[indices_similar])}
            )
        else:
            graph_pruned = ig.Graph(
                n=n_samples,
                edges=list(np.asarray(edges)[indices_similar])
            )

        graph_pruned.simplify(combine_edges="sum")  # "first"
        return graph_pruned

    def get_leiden_partition(self, graph, jac_weighted_edges=True):
        """Partition the graph using the Leiden algorithm.

        A partition is a set of communities.

        Args:
            graph (igraph.Graph): a ``Graph`` object which has been locally and globally pruned.
            jac_weighted_edges (bool): whether to partition using the weighted graph.

        Returns:
            leidenalg.VertexPartition:
                See `leidenalg.VertexPartition on GitHub
                <https://github.com/vtraag/leidenalg/blob/main/src/leidenalg/VertexPartition.py>`_.
        """
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

    def run_toobig_subPARC(self, x_data, jac_std_factor=0.3, jac_threshold_type="mean",
                           jac_weighted_edges=True):

        n_samples = x_data.shape[0]
        if n_samples <= 10:
            logger.message(
                f"Number of samples = {n_samples}, consider increasing the large_community_factor"
            )
        if n_samples > self.knn:
            knn = self.knn
        else:
            knn = int(max(5, 0.2 * n_samples))

        neighbor_array, distance_array = self.get_neighbor_distance_arrays(
            x_data=x_data, knn=knn, create_new=True, distance_metric="l2",
            hnsw_param_m=30, hnsw_param_ef_construction=200
        )

        csr_array = self.prune_local(neighbor_array, distance_array)
        graph_pruned = self.prune_global(
            csr_array=csr_array,
            jac_std_factor=jac_std_factor,
            jac_threshold_type=jac_threshold_type,
            jac_weighted_edges=jac_weighted_edges,
            n_samples=n_samples
        )
        del csr_array

        partition = self.get_leiden_partition(graph_pruned, jac_weighted_edges)

        node_communities = np.asarray(partition.membership)
        node_communities = np.reshape(node_communities, (n_samples, 1))
        del partition

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

    def run_parc(self):

        time_start = time.time()
        logger.message(
            f"Input data has shape {self.x_data.shape[0]} (samples) x "
            f"{self.x_data.shape[1]} (features)"
        )
        large_community_factor = self.large_community_factor
        small_community_size = self.small_community_size
        jac_std_factor = self.jac_std_factor
        jac_threshold_type = self.jac_threshold_type
        jac_weighted_edges = self.jac_weighted_edges
        n_samples = self.x_data.shape[0]

        neighbor_array, distance_array = self.get_neighbor_distance_arrays(
            x_data=self.x_data, knn=self.knn, create_new=False, distance_metric=self.distance_metric,
        )
        csr_array = self.prune_local(neighbor_array, distance_array)

        graph_pruned = self.prune_global(
            csr_array=csr_array,
            jac_std_factor=jac_std_factor,
            jac_threshold_type=jac_threshold_type,
            n_samples=n_samples,
            jac_weighted_edges=True
        )
        del csr_array

        logger.message("Starting Leiden community detection...")
        partition = self.get_leiden_partition(graph_pruned, jac_weighted_edges)
        del graph_pruned
        logger.info(get_current_memory_usage(process))

        node_communities = np.asarray(partition.membership)
        node_communities = np.reshape(node_communities, (n_samples, 1))
        del partition
        logger.info(get_current_memory_usage(process))

        too_big = False

        # the 0th cluster is the largest one. so if cluster 0 is not too big,
        # then the others wont be too big either
        community_indices = np.where(node_communities == 0)[0]
        community_size = len(community_indices)
        if community_size > large_community_factor * n_samples:
            too_big = True
            cluster_big_loc = community_indices
            list_pop_too_bigs = [community_size]
            cluster_too_big = 0

        while too_big:

            x_data_big = self.x_data[cluster_big_loc, :]
            node_communities_big = self.run_toobig_subPARC(x_data_big)
            node_communities_big = node_communities_big + 100000

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
            for community_id in set_node_communities:
                cluster_ii_loc = np.where(node_communities == community_id)[0]
                community_size = len(cluster_ii_loc)
                not_yet_expanded = community_size not in list_pop_too_bigs
                if community_size > large_community_factor * n_samples and not_yet_expanded:
                    too_big = True
                    logger.info(f"Cluster {community_id} is too big and has population {community_size}.")
                    cluster_big_loc = cluster_ii_loc
                    cluster_big = community_id
                    big_pop = community_size
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

        self.y_data_pred = node_communities
        run_time = time.time() - time_start
        logger.message(f"Time elapsed: {run_time} seconds")
        self.compute_performance_metrics(run_time)

    def accuracy(self, target=1):

        y_data_true = self.y_data_true
        Index_dict = {}
        y_data_pred = self.y_data_pred
        n_samples = len(y_data_pred)
        n_target = list(y_data_true).count(target)
        n_pbmc = n_samples - n_target

        for k in range(n_samples):
            Index_dict.setdefault(y_data_pred[k], []).append(y_data_true[k])
        num_groups = len(Index_dict)
        sorted_keys = list(sorted(Index_dict.keys()))
        error_count = []
        negative_labels = []
        positive_labels = []
        fp, fn, tp, tn, precision, recall, f1_score = 0, 0, 0, 0, 0, 0, 0

        for kk in sorted_keys:
            vals = [t for t in Index_dict[kk]]
            majority_val = get_mode(vals)
            if majority_val == target:
                logger.info(f"Cluster {kk} has majority {target} with population {len(vals)}")
            if kk == -1:
                len_unknown = len(vals)
                logger.info(f"Number of unknown: {len_unknown}")
            if (majority_val == target) and (kk != -1):
                positive_labels.append(kk)
                fp = fp + len([e for e in vals if e != target])
                tp = tp + len([e for e in vals if e == target])
                list_error = [e for e in vals if e != majority_val]
                e_count = len(list_error)
                error_count.append(e_count)
            elif (majority_val != target) and (kk != -1):
                negative_labels.append(kk)
                tn = tn + len([e for e in vals if e != target])
                fn = fn + len([e for e in vals if e == target])
                error_count.append(len([e for e in vals if e != majority_val]))

        predict_class_array = np.array(y_data_pred)
        y_data_pred_array = np.array(y_data_pred)
        number_clusters_for_target = len(positive_labels)
        for cancer_class in positive_labels:
            predict_class_array[y_data_pred_array == cancer_class] = 1
        for benign_class in negative_labels:
            predict_class_array[y_data_pred_array == benign_class] = 0
        predict_class_array.reshape((predict_class_array.shape[0], -1))
        error_rate = sum(error_count) / n_samples
        n_target = tp + fn
        tnr = tn / n_pbmc
        fnr = fn / n_target
        tpr = tp / n_target
        fpr = fp / n_pbmc

        if tp != 0 or fn != 0:
            recall = tp / (tp + fn)  # ability to find all positives
        if tp != 0 or fp != 0:
            precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
        if precision != 0 or recall != 0:
            f1_score = precision * recall * 2 / (precision + recall)
        majority_truth_labels = np.empty((len(y_data_true), 1), dtype=object)

        for cluster_i in set(y_data_pred):
            community_indices = np.where(np.asarray(y_data_pred) == cluster_i)[0]
            y_data_true = np.asarray(y_data_true)
            majority_truth = get_mode(list(y_data_true[community_indices]))
            majority_truth_labels[community_indices] = majority_truth

        majority_truth_labels = list(majority_truth_labels.flatten())
        accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                        recall, num_groups, n_target]

        return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target


    def compute_performance_metrics(self, run_time):

        list_roc = []
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
        if len(targets) > 1:
            f1_accumulated = 0
            f1_acc_noweighting = 0
            for target in targets:
                vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = self.accuracy(target=target)
                f1_current = vals_roc[1]
                logger.message(
                    f"target {target} has f1-score of {np.round(f1_current * 100, 2)}"
                )
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
