import logging
import numpy as np
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg
import time
from parc.k_nearest_neighbours import create_hnsw_index
from parc.metrics import compute_performance_metrics
from parc.graph import Community

logger = logging.getLogger(__name__)


class PARC:
    """Phenotyping by accelerated refined community-partitioning.

    Attributes:
        x_data: (np.array) a Numpy array of the input x data, with dimensions
            (n_samples, n_features).
        y_data_true: (np.array) a Numpy array of the output y labels.
        jac_threshold_type: (str) One of "median" or "mean". Determines how the Jaccard similarity
            threshold is calculated during global pruning.
        jac_std_factor: (float) The multiplier used in calculating the Jaccard similarity threshold
            for the similarity between two nodes during global pruning for
            ``jac_threshold_type = "mean"``:

            .. code-block:: python

                threshold = np.mean(similarities) - jac_std_factor * np.std(similarities)

            Setting ``jac_std_factor = 0.15`` and ``jac_threshold_type="mean"``
            performs empirically similar to ``jac_threshold_type="median"``, which does not use
            the ``jac_std_factor``.
            Generally values between 0-1.5 are reasonable.
            Higher ``jac_std_factor`` means more edges are kept.
        l2_std_factor: (int) The multiplier used in calculating the Euclidean distance threshold
            for the distance between two nodes during local pruning:

            .. code-block:: python

                max_distance = np.mean(distances) + l2_std_factor * np.std(distances)

            Avoid setting both the ``jac_std_factor`` (global) and the ``l2_std_factor`` (local)
            to < 0.5 as this is very aggressive pruning.
            Higher ``l2_std_factor`` means more edges are kept.
        keep_all_local_dist: (bool) whether or not to do local pruning.
            If None (default), set to ``true`` if the number of samples is > 300 000,
            and set to ``false`` otherwise.
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
        distance_metric: (string) the distance metric to be used in the KNN algorithm:
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
        neighbor_graph: (Compressed Sparse Row Matrix) A sparse matrix with dimensions
            (n_samples, n_samples), containing the distances between nodes.
        hnsw_param_ef_construction: (int) a higher value increases accuracy of index construction.
            Even for O(100 000) cells, 150-200 is adequate.
        hnsw_param_m: (int) TODO.
    """

    def __init__(self, x_data, y_data_true=None, l2_std_factor=3, jac_std_factor=0,
                 jac_threshold_type="median", keep_all_local_dist=None, large_community_factor=0.4,
                 small_community_size=10, jac_weighted_edges=True, knn=30, n_iter_leiden=5,
                 random_seed=42, n_threads=-1, distance_metric="l2", small_community_timeout=15,
                 partition_type="ModularityVP", resolution_parameter=1.0, knn_struct=None,
                 neighbor_graph=None, hnsw_param_ef_construction=150, hnsw_param_m=24,
                 hnsw_param_allow_override=True):

        self.y_data_true = y_data_true
        self.y_data_pred = None
        self.x_data = x_data
        self.l2_std_factor = l2_std_factor
        self.jac_std_factor = jac_std_factor
        self.jac_threshold_type = jac_threshold_type
        self.keep_all_local_dist = keep_all_local_dist
        self.large_community_factor = large_community_factor
        self.small_community_size = small_community_size
        self.jac_weighted_edges = jac_weighted_edges
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden
        self.random_seed = random_seed
        self.n_threads = n_threads
        self.distance_metric = distance_metric
        self.small_community_timeout = small_community_timeout
        self.resolution_parameter = resolution_parameter
        self.partition_type = partition_type
        self.knn_struct = knn_struct
        self.neighbor_array = None
        self.distance_array = None
        self.neighbor_graph = neighbor_graph
        self.hnsw_param_ef_construction = hnsw_param_ef_construction
        self.hnsw_param_m = hnsw_param_m
        self.hnsw_param_allow_override = hnsw_param_allow_override

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

    @property
    def neighbor_graph(self):
        return self._neighbor_graph

    @neighbor_graph.setter
    def neighbor_graph(self, neighbor_graph):
        if neighbor_graph is not None:
            self.neighbor_array = np.split(neighbor_graph.indices, neighbor_graph.indptr)[1:-1]
            self.distance_array = neighbor_graph.toarray()
        self._neighbor_graph = neighbor_graph

    def get_neighbor_distance_arrays(self, x_data):
        if self.neighbor_array is not None and self.distance_array is not None:
            return self.neighbor_array, self.distance_array
        else:
            knn_struct = self.get_knn_struct()
            neighbor_array, distance_array = knn_struct.knn_query(x_data, k=self.knn)
            return neighbor_array, distance_array

    def get_knn_struct(self):
        if self.knn_struct is None:
            self.knn_struct = self.make_knn_struct()
        return self.knn_struct

    def make_knn_struct(self):
        """Create a Hierarchical Navigable Small Worlds (HNSW) graph.

        See `hnswlib.Index
        <https://github.com/nmslib/hnswlib/blob/master/python_bindings/LazyIndex.py>`__.

        Returns:
            (hnswlib.Index): TODO.
        """
        if self.knn > 190:
            print(f"knn = {self.knn}; consider using a lower k for KNN graph construction")

        hnsw_index = create_hnsw_index(
            data=self.x_data,
            distance_metric=self.distance_metric,
            k=self.knn,
            M=self.hnsw_param_m,
            default_ef_construction=self.hnsw_param_ef_construction,
            n_threads=self.n_threads,
            allow_override=self.hnsw_param_allow_override
        )

        return hnsw_index

    def create_knn_graph(self):
        """Create a full k-nearest neighbors graph using the HNSW algorithm.

        Returns:
            (Compressed Sparse Row Matrix) A sparse matrix with dimensions (n_samples, n_samples),
                containing the pruned distances.
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
        if self.keep_all_local_dist:  # don't prune based on distance
            row_list.extend(list(np.transpose(
                np.ones((n_neighbors, n_samples)) * range(0, n_samples)).flatten()
            ))
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1. / (distance_array.flatten() + 0.1)).tolist()
        else:  # locally prune based on (squared) l2 distance

            print(f"""Starting local pruning based on Euclidean distance metric at
                   {self.l2_std_factor} standard deviations above mean""")
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
                        row_list.append(index)
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

    def check_if_large_community(self, node_communities, community_id, large_community_sizes=[]):
        """Check if the community size is greater than the max community size.

        Args:
            node_communities: (np.array) an array containing the community assignments for each
                node.
            community_id: (int) the integer id of the community.

        Returns:
            is_large_community: (bool) whether or not the community is too big.
            large_community_indices: (list) a list of node indices for the community,
                if it is too big.
            large_community_sizes: (list) the sizes of the communities that are too big.
        """

        is_large_community = False
        n_samples = node_communities.shape[0]
        community_indices = np.where(node_communities == community_id)[0]
        community_size = len(community_indices)
        large_community_indices = []
        not_yet_expanded = community_size not in large_community_sizes
        if community_size > self.large_community_factor * n_samples and not_yet_expanded:
            is_large_community = True
            large_community_indices = community_indices
            large_community_sizes.append(community_size)
            print(f"""Community {community_id} is too big, community size = {community_size}.
                It will be expanded.""")
        return is_large_community, large_community_indices, large_community_sizes

    def get_next_big_community(self, node_communities, large_community_sizes):
        """Find the next community which is too big, if it exists.

        Args:
            node_communities: (np.array) an array containing the community assignments for each
                node.
            large_community_sizes: (list) a list of community sizes corresponding to communities
                which have already been identified and processed as being too large.

        Returns:
            large_community_exists: (bool) whether or not the community is too big.
            large_community_indices: (list) a list of node indices for the community,
                if it is too big.
            large_community_sizes: (list) the sizes of the communities that are too big.
        """
        large_community_exists = False
        communities = set(node_communities)

        for community_id in communities:
            large_community_exists, large_community_indices, large_community_sizes = \
                self.check_if_large_community(
                    node_communities, community_id, large_community_sizes
                )
            if large_community_exists:
                break

        return large_community_exists, large_community_indices, large_community_sizes

    def reassign_large_communities(self, x_data, node_communities):
        """Find communities which are above the large_community_factor cutoff and split them.

        1. Check if the 0th community is too large. Since this is always the largest community,
            if it does not meet the threshold for a large community then all the other
            communities will also not be too large.
        2. If the 0th community is too large, then iterate through the rest of the communities. For
            each large community, run the PARC algorithm (but don't check for large communities).
            Reassign communities to split communities.

        Args:
            x_data: (np.array) an array containing the input data, with shape
                (n_samples x n_features).
            node_communities: (np.array) an array containing the community assignments for each
                node.

        Returns:
            node_communities: (np.array) an array containing the new community assignments for each
                node.
        """

        large_community_exists, large_community_indices, large_community_sizes = \
            self.check_if_large_community(
                node_communities=node_communities, community_id=0
            )
        while large_community_exists:
            if len(large_community_indices) <= self.knn:
                k = int(max(5, 0.2 * x_data.shape[0]))
            else:
                k = self.knn

            parc_model_large_community = PARC(
                x_data=x_data[large_community_indices, :],
                small_community_size=10,
                jac_std_factor=0.3,
                jac_threshold_type="mean",
                distance_metric="l2",
                n_threads=None,
                knn=k,
                hnsw_param_ef_construction=200,
                hnsw_param_m=30,
                hnsw_param_allow_override=False
            )
            node_communities_large_community = parc_model_large_community.run_parc(
                should_check_large_communities=False, should_compute_metrics=False
            )
            node_communities_large_community = node_communities_large_community + 100000

            for community_index, index in zip(large_community_indices, range(len(large_community_indices))):
                node_communities[community_index] = node_communities_large_community[index]

            node_communities = np.asarray(np.unique(
                list(node_communities.flatten()),
                return_inverse=True
            )[1])

            print(f"New set of labels: {set(node_communities)}")

            large_community_exists, large_community_indices, large_community_sizes = \
                self.get_next_big_community(
                    node_communities, large_community_sizes
                )

        return node_communities

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

    def run_parc(
        self, should_check_large_communities=True, should_compute_metrics=True
    ):

        time_start = time.time()
        x_data = self.x_data
        small_community_size = self.small_community_size

        print(
            f"Input data has shape {self.x_data.shape[0]} (samples) x"
            f"{self.x_data.shape[1]} (features)"
        )

        n_elements = x_data.shape[0]

        neighbor_array, distance_array = self.get_neighbor_distance_arrays(x_data)
        if self.neighbor_graph is None:
            csr_array = self.prune_local(neighbor_array, distance_array)
        else:
            csr_array = self.neighbor_graph

        graph = self.prune_global(csr_array, self.jac_std_factor, self.jac_threshold_type)

        print("Starting community detection...")
        partition = self.get_leiden_partition(graph, self.jac_weighted_edges)
        node_communities = np.reshape(np.asarray(partition.membership), (n_elements, 1))

        if should_check_large_communities:
            node_communities = self.reassign_large_communities(x_data, node_communities)

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]

        node_communities, small_community_exists = self.reassign_small_communities(
            node_communities, small_community_size, neighbor_array, exclude_neighbors_small=True
        )

        time_start_sc = time.time()
        while (small_community_exists) & ((time.time() - time_start_sc) < self.small_community_timeout):
            node_communities, small_community_exists = self.reassign_small_communities(
                node_communities, small_community_size, neighbor_array,
                exclude_neighbors_small=False
            )

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
        node_communities = list(node_communities.flatten())

        run_time = time.time() - time_start
        print(f"time elapsed: {run_time} seconds")
        self.y_data_pred = node_communities

        if should_compute_metrics:
            self.compute_performance_metrics(run_time)

        return node_communities

    def compute_performance_metrics(self, run_time):
        if self.y_data_true is None:
            logger.error("Unable to compute performance metrics, true target values not provided.")
            return

        f1_accumulated, f1_mean, stats_df, majority_truth_labels = compute_performance_metrics(
            self.y_data_true, self.y_data_pred, self.jac_std_factor, self.l2_std_factor, run_time
        )

        self.f1_accumulated = f1_accumulated
        self.f1_mean = f1_mean
        self.stats_df = stats_df
        self.majority_truth_labels = majority_truth_labels
