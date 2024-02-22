import numpy as np
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg
import time
from umap.umap_ import find_ab_params, simplicial_set_embedding
from parc.k_nearest_neighbours import create_hnsw_index
from parc.utils import get_mode


class PARC:
    """Phenotyping by accelerated refined community-partitioning.

    Attributes:
        data: (np.array) a Numpy array of the input x data, with dimensions (n_samples, n_features).
        true_label: (np.array) a Numpy array of the output y labels.
        dist_std_local: (int) similar to the jac_std_global parameter. Avoid setting local and
            global pruning to both be below 0.5 as this is very aggresive pruning.
            Higher ``dist_std_local`` means more edges are kept.
        jac_std_global: (float) 0.15 is a recommended value performing empirically similar
            to ``median``. Generally values between 0-1.5 are reasonable.
            Higher ``jac_std_global`` means more edges are kept.
        keep_all_local_dist: (bool) whether or not to do local pruning.
            Default is 'auto' which omits LOCAL pruning for samples > 300 000 cells.
        too_big_factor: (float) if a cluster exceeds this share of the entire cell population,
            then the PARC will be run on the large cluster. At 0.4 it does not come into play.
        small_pop: (int) the smallest cluster population size to be considered a community.
        jac_weighted_edges: (bool) whether to partition using the weighted graph.
        knn: (int) the number of clusters k for the k-nearest neighbours algorithm.
        n_iter_leiden: (int) the number of iterations for the Leiden algorithm.
        random_seed: (int) the random seed to enable reproducible Leiden clustering.
        num_threads: (int) the number of threads used in the KNN algorithm.
        distance: (string) the distance metric to be used in the KNN algorithm:
            "l2": Euclidean distance L^2 norm, d = sum((x_i - y_i)^2)
            "cosine": cosine similarity, d = 1.0 - sum(x_i*y_i) / sqrt(sum(x_i*x_i) * sum(y_i*y_i))
            "ip": inner product distance, d = 1.0 - sum(x_i*y_i)
        time_smallpop: (int) number of seconds trying to check an outlier
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

    def __init__(self, data, true_label=None, dist_std_local=3, jac_std_global="median",
                 keep_all_local_dist='auto', too_big_factor=0.4, small_pop=10,
                 jac_weighted_edges=True, knn=30, n_iter_leiden=5, random_seed=42,
                 num_threads=-1, distance='l2', time_smallpop=15, partition_type="ModularityVP",
                 resolution_parameter=1.0, knn_struct=None, neighbor_graph=None,
                 hnsw_param_ef_construction=150):

        if keep_all_local_dist == "auto":
            if data.shape[0] > 300000:
                keep_all_local_dist = True  # skips local pruning to increase speed
            else:
                keep_all_local_dist = False
        if resolution_parameter != 1:
            partition_type = "RBVP"
        self.data = data
        self.true_label = true_label
        self.dist_std_local = dist_std_local
        self.jac_std_global = jac_std_global
        self.keep_all_local_dist = keep_all_local_dist
        self.too_big_factor = too_big_factor
        self.small_pop = small_pop
        self.jac_weighted_edges = jac_weighted_edges
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden
        self.random_seed = random_seed
        self.num_threads = num_threads
        self.distance = distance
        self.time_smallpop = time_smallpop
        self.partition_type = partition_type
        self.resolution_parameter = resolution_parameter
        self.knn_struct = knn_struct
        self.neighbor_graph = neighbor_graph
        self.hnsw_param_ef_construction = hnsw_param_ef_construction

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
            data = self.data
            distance = self.distance
        else:
            data = big_cluster
            distance = "l2"

        hnsw_index = create_hnsw_index(
            data, distance, self.knn, self.num_threads,
            self.hnsw_param_ef_construction, too_big
        )

        return hnsw_index

    def knngraph_full(self):#, neighbor_array, distance_array):
        k_umap = 15
        # neighbors in array are not listed in in any order of proximity
        self.knn_struct.set_ef(k_umap+1)
        neighbor_array, distance_array = self.knn_struct.knn_query(self.data, k=k_umap)

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

    def run_toobig_subPARC(self, X_data, jac_std_threshold=0.3, jac_weighted_edges=True):
        n_elements = X_data.shape[0]
        hnsw = self.make_knn_struct(too_big=True, big_cluster=X_data)
        if n_elements <= 10:
            print('consider increasing the too_big_factor')
        if n_elements > self.knn:
            knnbig = self.knn
        else:
            knnbig = int(max(5, 0.2 * n_elements))

        neighbor_array, distance_array = hnsw.knn_query(X_data, k=knnbig)

        csr_array = self.prune_local(neighbor_array, distance_array)
        graph = self.prune_global(csr_array, jac_std_threshold, jac_weighted_edges)

        if jac_weighted_edges:
            if self.partition_type == 'ModularityVP':
                partition = leidenalg.find_partition(
                    graph=graph,
                    partition_type=leidenalg.ModularityVertexPartition, weights='weight',
                    n_iterations=self.n_iter_leiden, seed=self.random_seed)
                print('partition type MVP')
            else:
                partition = leidenalg.find_partition(
                    graph=graph,
                    partition_type=leidenalg.RBConfigurationVertexPartition, weights='weight',
                    n_iterations=self.n_iter_leiden, seed=self.random_seed,
                    resolution_parameter=self.resolution_parameter
                )
                print('partition type RBC')
        else:
            if self.partition_type == 'ModularityVP':
                print('partition type MVP')
                partition = leidenalg.find_partition(
                    graph=graph,
                    partition_type=leidenalg.ModularityVertexPartition,
                    n_iterations=self.n_iter_leiden,
                    seed=self.random_seed
                )
            else:
                print('partition type RBC')
                partition = leidenalg.find_partition(
                    graph=graph, partition_type=leidenalg.RBConfigurationVertexPartition,
                    n_iterations=self.n_iter_leiden, seed=self.random_seed,
                    resolution_parameter=self.resolution_parameter
                )
        # print('Q= %.2f' % partition.quality())
        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False
        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        for cluster in set(PARC_labels_leiden):
            population = len(np.where(PARC_labels_leiden == cluster)[0])
            if population < 10:
                small_pop_exist = True
                small_pop_list.append(list(np.where(PARC_labels_leiden == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:
            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell, :]
                group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    PARC_labels_leiden[single_cell] = best_group

        time_smallpop_start = time.time()
        print('handling fragments')
        while (small_pop_exist) & (time.time() - time_smallpop_start < self.time_smallpop):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < 10:
                    small_pop_exist = True

                    small_pop_list.append(np.where(PARC_labels_leiden == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    PARC_labels_leiden[single_cell] = best_group

        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)

        return PARC_labels_leiden

    def check_if_clusters_oversized(self, cluster_ids, n_samples):
        """Check if the clusters are too big.

        The 0th cluster is the largest one. So if cluster 0 is not too big, then the others won't
        be too big either.

        Args:
            cluster_ids: TODO.
            n_samples: (int) the number of samples in the data.

        Returns:
            too_big: (bool) whether or not the clusters are too big.
            big_cluster_indices: (list) a list of indices of the 0th cluster, if it is too big.
            cluster_size: (int) the size of the 0th cluster.
        """

        too_big = False
        cluster_0_indices = np.where(cluster_ids == 0)[0]
        cluster_size = len(cluster_0_indices)
        big_cluster_indices = []
        if cluster_size > self.too_big_factor * n_samples:  # 0.4
            too_big = True
            big_cluster_indices = cluster_0_indices
        return too_big, big_cluster_indices, cluster_size

    def run_subPARC(self):

        X_data = self.data
        too_big_factor = self.too_big_factor
        small_pop = self.small_pop
        jac_weighted_edges = self.jac_weighted_edges
        knn = self.knn
        n_elements = X_data.shape[0]

        if self.neighbor_graph is not None:
            csr_array = self.neighbor_graph
            neighbor_array = np.split(csr_array.indices, csr_array.indptr)[1:-1]
        else:
            knn_struct = self.get_knn_struct()
            neighbor_array, distance_array = knn_struct.knn_query(X_data, k=knn)
            csr_array = self.prune_local(neighbor_array, distance_array)

        graph = self.prune_global(csr_array, self.jac_std_global)

        print("Starting community detection")
        if jac_weighted_edges:
            weights = "weight"
        else:
            weights = None

        if self.partition_type == 'ModularityVP':
            print('partition type MVP')
            partition = leidenalg.find_partition(
                graph=graph,
                partition_type=leidenalg.ModularityVertexPartition, weights=weights,
                n_iterations=self.n_iter_leiden, seed=self.random_seed
            )
        else:
            print('partition type RBC')
            partition = leidenalg.find_partition(
                graph=graph,
                partition_type=leidenalg.RBConfigurationVertexPartition, weights=weights,
                n_iterations=self.n_iter_leiden, seed=self.random_seed,
                resolution_parameter=self.resolution_parameter
            )

        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))

        too_big, big_cluster_indices, cluster_size = self.check_if_clusters_oversized(
            PARC_labels_leiden, n_elements
        )
        big_cluster_sizes = [cluster_size]

        while too_big:

            X_data_big = X_data[cluster_big_loc, :]
            PARC_labels_leiden_big = self.run_toobig_subPARC(X_data_big)
            # print('set of new big labels ', set(PARC_labels_leiden_big.flatten()))
            PARC_labels_leiden_big = PARC_labels_leiden_big + 100000
            # print('set of new big labels +100000 ', set(list(PARC_labels_leiden_big.flatten())))
            pop_list = []

            for item in set(list(PARC_labels_leiden_big.flatten())):
                pop_list.append([item, list(PARC_labels_leiden_big.flatten()).count(item)])
            print('pop of big clusters', pop_list)
            jj = 0
            print('shape PARC_labels_leiden', PARC_labels_leiden.shape)
            for j in cluster_big_loc:
                PARC_labels_leiden[j] = PARC_labels_leiden_big[jj]
                jj = jj + 1
            dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
            print('new set of labels ', set(PARC_labels_leiden))
            too_big = False
            set_PARC_labels_leiden = set(PARC_labels_leiden)

            PARC_labels_leiden = np.asarray(PARC_labels_leiden)
            for cluster_ii in set_PARC_labels_leiden:
                cluster_ii_loc = np.where(PARC_labels_leiden == cluster_ii)[0]
                pop_ii = len(cluster_ii_loc)
                not_yet_expanded = pop_ii not in list_pop_too_bigs
                if pop_ii > too_big_factor * n_elements and not_yet_expanded:
                    too_big = True
                    print('cluster', cluster_ii, 'is too big and has population', pop_ii)
                    cluster_big_loc = cluster_ii_loc
                    cluster_big = cluster_ii
                    big_pop = pop_ii
            if too_big:
                list_pop_too_bigs.append(big_pop)
                print('cluster', cluster_big, 'is too big with population', big_pop, '. It will be expanded')
        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False

        for cluster in set(PARC_labels_leiden):
            population = len(np.where(PARC_labels_leiden == cluster)[0])

            if population < small_pop:  # 10
                small_pop_exist = True

                small_pop_list.append(list(np.where(PARC_labels_leiden == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:

            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell]
                group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    PARC_labels_leiden[single_cell] = best_group
        time_smallpop_start = time.time()
        while (small_pop_exist) & ((time.time() - time_smallpop_start) < self.time_smallpop):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < small_pop:
                    small_pop_exist = True
                    print(cluster, ' has small population of', population, )
                    small_pop_list.append(np.where(PARC_labels_leiden == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell]
                    group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    PARC_labels_leiden[single_cell] = best_group

        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        PARC_labels_leiden = list(PARC_labels_leiden.flatten())

        pop_list = [(item, PARC_labels_leiden.count(item)) for item in set(PARC_labels_leiden)]

        print('list of cluster labels and populations', len(pop_list), pop_list)

        self.labels = PARC_labels_leiden
        return

    def accuracy(self, onevsall=1):

        true_labels = self.true_label
        Index_dict = {}
        PARC_labels = self.labels
        N = len(PARC_labels)
        n_cancer = list(true_labels).count(onevsall)
        n_pbmc = N - n_cancer

        for k in range(N):
            Index_dict.setdefault(PARC_labels[k], []).append(true_labels[k])
        num_groups = len(Index_dict)
        sorted_keys = list(sorted(Index_dict.keys()))
        error_count = []
        pbmc_labels = []
        thp1_labels = []
        fp, fn, tp, tn, precision, recall, f1_score = 0, 0, 0, 0, 0, 0, 0

        for kk in sorted_keys:
            vals = [t for t in Index_dict[kk]]
            majority_val = get_mode(vals)
            if majority_val == onevsall:
                print(f"cluster {kk} has majority {onevsall} with population {len(vals)}")
            if kk == -1:
                len_unknown = len(vals)
                print('len unknown', len_unknown)
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

        predict_class_array = np.array(PARC_labels)
        PARC_labels_array = np.array(PARC_labels)
        number_clusters_for_target = len(thp1_labels)
        for cancer_class in thp1_labels:
            predict_class_array[PARC_labels_array == cancer_class] = 1
        for benign_class in pbmc_labels:
            predict_class_array[PARC_labels_array == benign_class] = 0
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
        majority_truth_labels = np.empty((len(true_labels), 1), dtype=object)

        for cluster_i in set(PARC_labels):
            cluster_i_loc = np.where(np.asarray(PARC_labels) == cluster_i)[0]
            true_labels = np.asarray(true_labels)
            majority_truth = get_mode(list(true_labels[cluster_i_loc]))
            majority_truth_labels[cluster_i_loc] = majority_truth

        majority_truth_labels = list(majority_truth_labels.flatten())
        accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                        recall, num_groups, n_target]

        return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target

    def run_PARC(self):
        print('input data has shape', self.data.shape[0], '(samples) x', self.data.shape[1], '(features)')
        if self.true_label is None:
            self.true_label = [1] * self.data.shape[0]
        list_roc = []

        time_start_total = time.time()

        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        self.run_subPARC()
        run_time = time.time() - time_start_total
        print('time elapsed {:.1f} seconds'.format(run_time))

        targets = list(set(self.true_label))
        N = len(list(self.true_label))
        self.f1_accumulated = 0
        self.f1_mean = 0
        self.stats_df = pd.DataFrame({
            'jac_std_global': [self.jac_std_global],
            'dist_std_local': [self.dist_std_local],
            'runtime(s)': [run_time]
        })
        self.majority_truth_labels = []
        if len(targets) > 1:
            f1_accumulated = 0
            f1_acc_noweighting = 0
            for onevsall_val in targets:
                print(f"target is {onevsall_val}")
                vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = self.accuracy(
                    onevsall=onevsall_val)
                f1_current = vals_roc[1]
                print('target', onevsall_val, 'has f1-score of %.2f' % (f1_current * 100))
                f1_accumulated = f1_accumulated + f1_current * (list(self.true_label).count(onevsall_val)) / N
                f1_acc_noweighting = f1_acc_noweighting + f1_current

                list_roc.append(
                    [self.jac_std_global, self.dist_std_local, onevsall_val] + vals_roc
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
        return

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
