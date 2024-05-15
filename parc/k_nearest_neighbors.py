import numpy as np


def get_distance_array(csr_array):
    """Given a ``csr_matrix``, get a sorted distance array.

    Args:
        csr_array (scipy.sparse.csr_matrix): A compressed sparse row matrix with dimensions
            (n_samples, n_samples), containing the pruned distances.

    Returns:
        np.array: An array with dimensions (n_samples, k) distances to each of the
            k nearest neighbors for each data point, in order of proximity. For example, if my
            data has 5 samples, with ``k=3`` nearest neighbors, the resulting array might be:

            .. code-block:: python

                distance_array = np.array([
                    [0.10, 0.03, 0.01],
                    [0.50, 0.80, 1.00],
                    [0.13, 0.15, 0.90],
                    [0.40, 0.51, 0.87],
                    [0.98, 1.56, 1.90]
                ])
    """
    similarity_data = 1 / (csr_array.data + 0.00001)
    return np.sort(np.array(np.split(similarity_data, csr_array.indptr)[1:-1]), axis=1)


def get_neighbor_array(csr_array):
    """Given a ``csr_matrix``, get a sorted nearest neighbors array.

    Args:
        csr_array (scipy.sparse.csr_matrix): A compressed sparse row matrix with dimensions
            (n_samples, n_samples), containing the pruned distances.

    Returns:
        np.array: An array with dimensions (n_samples, k) listing the k nearest neighbors for
            each data point, in order of proximity. For example, if my data has 5 samples, with
            ``k=3`` nearest neighbors, the resulting array might be:

            .. code-block:: python

                neighbor_array = np.array([
                    [0, 3, 1],
                    [1, 2, 4],
                    [2, 0, 1],
                    [3, 0, 2],
                    [4, 3, 1]
                ])
    """
    similarity_data = 1 / (csr_array.data + 0.00001)
    sorted_indices = np.argsort(np.array(np.split(similarity_data, csr_array.indptr)[1:-1]), axis=1)
    return np.take_along_axis(
        np.array(np.split(csr_array.indices, csr_array.indptr)[1:-1]),
        sorted_indices,
        axis=1
    )


def get_edges(csr_array):
    """Given a ``csr_matrix``, get the graph edges of the k nearest neighbors as ordered pairs.

    Args:
        csr_array (scipy.sparse.csr_matrix): A compressed sparse row matrix with dimensions
            (n_samples, n_samples), containing the pruned distances.

    Returns:
        list[tuple]: A list of 2-tuples representing an edge, with length ``n_samples * knn``.
            For example, if my data has 5 samples, with ``k=3`` nearest neighbors, and the resulting
            ``neighbor_array`` is:

            .. code-block:: python

                neighbor_array = np.array([
                    [0, 3, 1],
                    [1, 2, 4],
                    [2, 0, 1],
                    [3, 0, 2],
                    [4, 3, 1]
                ])

            then the ``edges`` would be:

            .. code-block:: python

                edges = [
                    (0, 0), (0, 1), (0, 3),
                    (1, 1), (1, 2), (1, 4),
                    (2, 0), (2, 1), (2, 2),
                    (3, 0), (3, 2), (3, 3),
                    (4, 1), (4, 3), (4, 4)
                ]

    """
    input_nodes, output_nodes = csr_array.nonzero()
    edges = list(zip(input_nodes, output_nodes))
    return edges
