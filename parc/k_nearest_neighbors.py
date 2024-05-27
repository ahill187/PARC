import numpy as np
from progress.bar import Bar
from parc.logger import get_logger

logger = get_logger(__name__)


DISTANCE_FACTOR = 0.000001
DISTANCE_PLACEHOLDER = 1000000
NEIGHBOR_PLACEHOLDER = -1


class NearestNeighbors:
    """The ids and distances to the nearest neighbors of a community.

    Attributes:
        community_id (int): The id of the community.
        neighbors (np.array): a k x 1 Numpy vector with the community ids of the k nearest neighbors
            for this community, in order of proximity. For example, if there are ``k=3``
            nearest neighbors:

            .. code-block:: python

                neighbors = np.array([2, 4, 0])

        distances (np.array): A k x 1 Numpy vector, giving the distances to each of the
            k nearest neighbors for each community, in order of proximity.
            For example, if there are ``k=3`` nearest neighbors:

            .. code-block:: python

                distances_collection = np.array([0.13, 0.15, 0.90])

    """

    def __init__(self, community_id, neighbors, distances):
        self.community_id = community_id
        self.neighbors = neighbors
        self.distances = distances

    @property
    def neighbors(self):
        return self._neighbors

    @neighbors.setter
    def neighbors(self, neighbors):
        if neighbors is None:
            raise TypeError(
                "Neighbors is None, must provide list of neighbors"
            )
        else:
            if isinstance(neighbors, list):
                try:
                    neighbors = np.array(neighbors)
                except Exception as error:
                    logger.error(error)

            if isinstance(neighbors, np.ndarray):
                if len(neighbors.shape) > 1:
                    raise ValueError(
                        f"Neighbors must be an n x 1 Numpy array or an n x 1 list; "
                        f"got an array with shape {neighbors.shape}"
                    )
                elif not isinstance(neighbors[0], (int, np.integer)):
                    if isinstance(neighbors[0], float):
                        self._neighbors = neighbors.astype(int)
                        logger.warning(
                            "Neighbors contained float values; converting them to integers."
                        )
                    else:
                        raise TypeError(
                            f"Neighbors must be integers; "
                            f"got an array with data type {type(neighbors[0])}"
                        )
                else:
                    self._neighbors = neighbors
            else:
                raise TypeError(
                    f"Neighbors must be an n x 1 Numpy array or an n x 1 list; "
                    f"got variable of type {type(neighbors)}"
                )

    @property
    def distances(self):
        return self._distances

    @distances.setter
    def distances(self, distances):
        if distances is None:
            raise TypeError(
                "Distances is None, must provide distances"
            )
        else:
            if isinstance(distances, list):
                try:
                    distances = np.array(distances)
                except Exception as error:
                    logger.error(error)

            if isinstance(distances, np.ndarray):
                if len(distances.shape) > 1:
                    raise ValueError(
                        f"Distances must be an n x 1 Numpy array or an n x 1 list; "
                        f"got an array with shape {distances.shape}"
                    )
                elif (not isinstance(distances[0], (int, np.integer))
                      and not isinstance(distances[0], (float, np.floating))):
                    raise TypeError(
                        f"Distances must be either float or integer; "
                        f"got an array with data type {type(distances[0])}"
                    )
                else:
                    self._distances = distances
            else:
                raise TypeError(
                    f"Distances must be an n x 1 Numpy array or an n x 1 list; "
                    f"got variable of type {type(distances)}"
                )

    def remove_indices(self, indices):
        self.neighbors = np.delete(self.neighbors, np.ix_(indices))
        self.distances = np.delete(self.distances, np.ix_(indices))
        return self.neighbors, self.distances


class NearestNeighborsCollection:
    """A collection of nearest neighbors and their distances for a graph.

    Attributes:
        max_neighbors (int): Since the value of k can be different for each community in the
            list of nearest neighbors, this is the maximum value of k.
            For example, if we have 5 communities:

            .. code-block:: python

                neighbors_collection = [
                    np.array([0, 3]),
                    np.array([1]),
                    np.array([2, 4, 0]),
                    np.array([3, 2]),
                    np.array([4, 2, 3])
                ]

            then the maximum value of k would be 3.
        n_communities (int): The total number of communities in the graph.
        neighbors_collection (list[np.array]): a list of k x 1 Numpy vectors with the
            k nearest neighbors for each community, in order of proximity. Note that the
            value of k can be different for each community.
            For example, if we have 5 communities:

            .. code-block:: python

                neighbors_collection = [
                    np.array([0, 3]),
                    np.array([1]),
                    np.array([2, 4, 0]),
                    np.array([3, 2]),
                    np.array([4, 2, 3])
                ]
        distances_collection (list[np.array]): A list of k x 1 Numpy vectors, giving the
            distances to each of the k nearest neighbors for each community, in order of
            proximity. Note that the value of k may be different for each community.
            For example, if my data has 5 communities:

            .. code-block:: python

                distances_collection = [
                    np.array([0.10, 0.03]),
                    np.array([0.50]),
                    np.array([0.13, 0.15, 0.90]),
                    np.array([0.40, 0.51]),
                    np.array([0.98, 1.56, 1.90])
                ]

    """

    def __init__(self, neighbors_collection=None, distances_collection=None, csr_array=None):

        self.max_neighbors = 0
        self.n_communities = 0
        self.neighbors_collection = neighbors_collection
        self.distances_collection = distances_collection
        if csr_array is not None:
            self.initialize_from_csr_array(csr_array)

    def initialize_from_csr_array(self, csr_array):
        """Given a ``csr_matrix``, initialize the distances and neighbors.

        Args:
            csr_array (scipy.sparse.csr_matrix): A compressed sparse row matrix with dimensions
                (n_communities, n_communities), containing the distances to the
                nearest neighbors.
        """
        self.max_neighbors = self.get_max_neighbors(
            neighbors_collection=np.split(csr_array.indices, csr_array.indptr)[1:-1]
        )
        self.distances_collection = self.get_distances_from_csr_array(csr_array)
        self.neighbors_collection = self.get_neighbors_from_csr_array(csr_array)

    @property
    def distances_collection(self):
        return self._distances_collection

    @distances_collection.setter
    def distances_collection(self, distances_collection):
        if distances_collection is not None:
            if isinstance(distances_collection, np.ndarray):
                if len(distances_collection.shape) > 2:
                    raise ValueError(
                        f"Distances must be an n x m array or a list of n x 1 arrays; "
                        f"got an array with shape {distances_collection.shape}"
                    )
                elif distances_collection.shape[1] > distances_collection.shape[0]:
                    raise ValueError(
                        f"Distances array cannot have more neighbors than communities. "
                        f"got an array with {distances_collection.shape[0]} communities and "
                        f"{distances_collection.shape[1]} neighbors."
                    )
                else:
                    self._distances_collection = list(distances_collection)
            elif isinstance(distances_collection, list):
                self._distances_collection = distances_collection
            else:
                raise TypeError(
                    f"Distances must be an n x m array or a list of n x 1 arrays; "
                    f"got variable of type {type(distances_collection)}"
                )
        else:
            self._distances_collection = []

    @property
    def neighbors_collection(self):
        return self._neighbors_collection

    @neighbors_collection.setter
    def neighbors_collection(self, neighbors_collection):
        if neighbors_collection is not None:
            if isinstance(neighbors_collection, np.ndarray):
                if len(neighbors_collection.shape) > 2:
                    raise ValueError(
                        f"Neighbors must be an n x m array or a list of n x 1 arrays; "
                        f"got an array with shape {neighbors_collection.shape}"
                    )
                elif neighbors_collection.shape[1] > neighbors_collection.shape[0]:
                    raise ValueError(
                        f"Neighbors array cannot have more neighbors than communities. "
                        f"got an array with {neighbors_collection.shape[0]} communities and "
                        f"{neighbors_collection.shape[1]} neighbors."
                    )
                else:
                    self._neighbors_collection = list(neighbors_collection)
                    self.max_neighbors = neighbors_collection.shape[1]
                    self.n_communities = neighbors_collection.shape[0]
            elif isinstance(neighbors_collection, list):
                self._neighbors_collection = neighbors_collection
                self.max_neighbors = self.get_max_neighbors(neighbors_collection)
                self.n_communities = len(neighbors_collection)
            else:
                raise TypeError(
                    f"Neighbors must be an n x m array or a list of n x 1 arrays; "
                    f"got variable of type {type(neighbors_collection)}"
                )
        else:
            self._neighbors_collection = []

    def _check_distances(self, distances):
        if isinstance(distances, list):
            try:
                distances = np.array(distances)
            except Exception as error:
                logger.error(error)

        if isinstance(distances, np.ndarray):
            if len(distances.shape) > 1:
                raise ValueError(
                    f"Distances must be an n x 1 Numpy array or an n x 1 list; "
                    f"got an array with shape {distances.shape}"
                )
            elif (not isinstance(distances[0], (int, np.integer))
                  and not isinstance(distances[0], (float, np.floating))):
                raise TypeError(
                    f"Distances must be either float or integer; "
                    f"got an array with data type {type(distances[0])}"
                )
            return distances

        else:
            raise TypeError(
                f"Distances must be an n x 1 Numpy array or an n x 1 list; "
                f"got variable of type {type(distances)}"
            )

    def _check_neighbors(self, neighbors):
        if isinstance(neighbors, list):
            try:
                neighbors = np.array(neighbors)
            except Exception as error:
                logger.error(error)

        if isinstance(neighbors, np.ndarray):
            if len(neighbors.shape) > 1:
                raise ValueError(
                    f"Neighbors must be an n x 1 Numpy array or an n x 1 list; "
                    f"got an array with shape {neighbors.shape}"
                )
            elif not isinstance(neighbors[0], (int, np.integer)):
                if isinstance(neighbors[0], float):
                    neighbors = neighbors.astype(int)
                    logger.warning(
                        "Neighbors contained float values; converting them to integers."
                    )
                else:
                    raise TypeError(
                        f"Neighbors must be integers; "
                        f"got an array with data type {type(neighbors[0])}"
                    )
            return neighbors
        else:
            raise TypeError(
                f"Neighbors must be an n x 1 Numpy array or an n x 1 list; "
                f"got variable of type {type(neighbors)}"
            )

    def get_max_neighbors(self, neighbors_collection):
        """Given a list of neighbors, get the maximum value for k.

        Args:
            neighbors_collection (list[np.array]): a list of Numpy arrays with the
                k nearest neighbors for each community, in order of proximity. Note that the
                value of k can be different for each community.

        Returns:
            int: Since the value of k can be different for each community in the list of
                nearest neighbors, return the maximum value of k (the maximum length of arrays).
                For example, if we have 5 communities:

                .. code-block:: python

                    neighbors_collection = [
                        np.array([0, 3]),
                        np.array([1]),
                        np.array([2, 4, 0]),
                        np.array([3, 2]),
                        np.array([4, 2, 3])
                    ]

                then the maximum value of k would be 3.

        """
        max_neighbors = 0
        for neighbors in neighbors_collection:
            if len(neighbors) > max_neighbors:
                max_neighbors = len(neighbors)
        return max_neighbors

    def append(self, neighbors, distances):
        neighbors = self._check_neighbors(neighbors)
        distances = self._check_distances(distances)

        if neighbors.shape[0] != distances.shape[0]:
            raise ValueError(
                "Neighbors and distances must be same shape; got neighbors with shape "
                f"{neighbors.shape[0]}, distances with shape {distances.shape[0]}."
            )

        self.neighbors_collection.append(neighbors)
        self.distances_collection.append(distances)
        self.n_communities += 1

    def to_list(self):
        """Convert this object to a list of ``NearestNeighbors`` objects.

        Returns:
            list[NearestNeighbors]: A list with dimensions (n_communities, 1), giving the
                ``NearestNeighbors`` object for each community. Each ``NearestNeighbors`` object
                contains the ids and distances to each of the k nearest neighbors for a
                given community, in order of proximity.
        """
        return [
            NearestNeighbors(community_id, neighbors, distances)
            for community_id, neighbors, distances
            in zip(range(self.n_communities), self.neighbors_collection, self.distances_collection)
        ]

    def get_distances(self, community_id=None, as_type="collection"):
        """Get distances to the nearest neighbors for either a single community or all communities.

        Args:
            community_id (int): (optional) If the ``community_id`` is provided, then return the
                distances to the k nearest neighbors for that just that community. If it is not
                provided, return the distances to the k nearest neighbors for all communities.
            as_type (str): One of ``"collection"`` or ``"array"``.

                - ``"collection"``: Return the distances as a collection (k can be different
                  for each community).
                - ``"array"``: Return the distances as a Numpy array. If k is different for
                  different communities, fill the missing values with
                  ``DISTANCE_PLACEHOLDER = 1000000``.
                - ``"flatten"``: Return the distances as a flattened list.

        Returns:
            list[np.array] or np.array:

                1. If the ``community_id`` is provided, a Numpy array containing the distances to
                   the k nearest neighbors for that community, in order of proximity. For example:

                   .. code-block:: python

                       distances = np.array([0, 0.03, 0.01])

                2. If the ``community_id`` is not provided, return the distances to the
                   k nearest neighbors for each community.

                   - if ``as_type="collection"``, return the distances as a list of Numpy arrays.
                     It is possible for ``k`` to be different for each community. For example:

                   .. code-block:: python

                       distances_collection = [
                           np.array([0.10, 0.03]),
                           np.array([0.50]),
                           np.array([0.13, 0.15, 0.90]),
                           np.array([0.40, 0.51]),
                           np.array([0.98, 1.56, 1.90])
                       ]

                    - if ``as_type="array"``, return the distances as a Numpy array. If ``k`` is
                      not the same for all communities, fill the missing values with
                      ``DISTANCE_PLACEHOLDER = 1000000``. For example:

                      .. code-block:: python

                          distances_array = np.array([
                              [0.10, 0.03, 1000000],
                              [0.50, 1000000, 1000000],
                              [0.13, 0.15, 0.90],
                              [0.40, 0.51, 1000000],
                              [0.98, 1.56, 1.90]
                          ])

                    - If ``as_type="flatten"``, return the distances as a flattened list.
                      For example, if the ``distances_collection`` is the same as in the example
                      above, the flattened distances would be:

                      .. code-block:: python

                          distances_flattened = [
                              0.10, 0.03, 0.50, 0.13, 0.15, 0.90, 0.40, 0.51, 0.98, 1.56, 1.90
                          ]

        """
        if community_id is None:
            if as_type == "collection":
                return self.distances_collection
            elif as_type == "array":
                return self.get_distances_array()
            elif as_type == "flatten":
                distances_flattened = []
                for distances in self.distances_collection:
                    distances_flattened += distances.tolist()
                return distances_flattened
        else:
            return self.distances_collection[community_id]

    def get_neighbors(self, community_id=None, as_type="collection"):
        """Get the neighbors for either a single community or all communities.

        Args:
            community_id (int): (optional) If the ``community_id`` is provided, then return the
                k nearest neighbors for that just that community. If it is not provided, return
                the k nearest neighbors for all communities.
            as_type (str): One of ``"collection"``, ``"array"``, or ``"flatten"``.

                - ``"collection"``: Return the neighbors as a collection (k can be different
                  for each community).
                - ``"array"``: Return the neighbors as a Numpy array. If k is different for
                  different communities, fill the missing values with ``NEIGHBOR_PLACEHOLDER = -1``.
                - ``"flatten"``: Return the neighbors as a flattened list.

        Returns:
            list[np.array] or np.array:

                1. If the ``community_id`` is provided, a Numpy array containing the k nearest
                   neighbors for that community, in order of proximity. For example, if
                   the ``community_id = 2``:

                   .. code-block:: python

                       neighbors = np.array([2, 4, 0])

                2. If the ``community_id`` is not provided, return the neighbors for each
                   community.

                   - If ``as_type="collection"``, return the neighors as a list of Numpy arrays.
                     It is possible for ``k`` to be different for each community. For example:

                     .. code-block:: python

                         neighbors_collection = [
                             np.array([0, 3]),
                             np.array([1]),
                             np.array([2, 4, 0]),
                             np.array([3, 2]),
                             np.array([4, 2, 3])
                         ]

                    - If ``as_type="array"``, return the neighbors as a Numpy array. If ``k`` is
                      not the same for all communities, fill the missing values with
                      ``NEIGHBOR_PLACEHOLDER=-1``. For example:

                      .. code-block:: python

                          neighbors_array = np.array([
                              [0, 3, -1],
                              [1, -1, -1],
                              [2, 4, 0],
                              [3, 2, -1],
                              [4, 2, 3]
                          ])

                    - If ``as_type="flatten"``, return the neighbors as a flattened list.
                      For example, if the ``neighbors_collection`` is the same as in the example
                      above, the flattened neighbors would be:

                      .. code-block:: python

                          neighbors_flattened = [
                              0, 3, 1, 2, 4, 0, 3, 2, 4, 2, 3
                          ]

        """
        if community_id is None:
            if as_type == "collection":
                return self.neighbors_collection
            elif as_type == "array":
                return self.get_neighbors_array()
            elif as_type == "flatten":
                neighbors_flattened = []
                for neighbors in self.neighbors_collection:
                    neighbors_flattened += neighbors.tolist()
                return neighbors_flattened
        else:
            return self.neighbors_collection[community_id]

    def get_weights(self, community_id=None, as_type="collection"):
        r"""Get the weights for either a single community or all communities.

        We are assuming that the distances are the L2 distances. We define the weights as:

        .. math::

            \begin{align}
                w &:= dfrac{1}{\sqrt{d}}
            \end{align}


        Args:
            community_id (int): (optional) If the ``community_id`` is provided, then return the
                weights for the k nearest neighbors for that just that community. If it is not
                provided, return the weights for the k nearest neighbors for all communities.
            as_type (str): One of ``"collection"``, ``"array"``, or ``"flatten"``.

                - ``"collection"``: Return the weights as a collection (k can be different
                  for each community).
                - ``"array"``: Return the weights as a Numpy array. If k is different for
                  different communities, fill the missing values with
                  ``WEIGHT_PLACEHOLDER = 0.001``.
                - ``"flatten"``: Return the weights as a flattened list.

        Returns:
            list[np.array] or np.array:

                1. If the ``community_id`` is provided, a Numpy array containing the distances to
                   the k nearest neighbors for that community, in order of proximity. For example:

                   .. code-block:: python

                       weights = np.array([1000000, 5.7, 10.0])

                2. If the ``community_id`` is not provided, return the distances to the
                   k nearest neighbors for each community.

                   - if ``as_type="collection"``, return the weights as a list of Numpy arrays.
                     It is possible for ``k`` to be different for each community. For example:

                   .. code-block:: python

                       weights_collection = [
                           np.array([1000000, 5.7]),
                           np.array([1000000]),
                           np.array([1000000, 2.58, 1.05]),
                           np.array([1000000, 1.4]),
                           np.array([1000000, 0.8, 0.725])
                       ]

                    - If ``as_type="array"``, return the weights as a Numpy array. If ``k`` is
                      not the same for all communities, fill the missing values with
                      ``WEIGHT_PLACEHOLDER = 0.001``. For example:

                      .. code-block:: python

                          weights_array = np.array([
                              [1000000, 5.7, 0.001],
                              [1000000, 0.001, 0.001],
                              [1000000, 2.58, 1.05],
                              [1000000, 1.4, 0.001],
                              [1000000, 0.8, 0.725]
                          ])

                    - If ``as_type="flatten"``, return the weights as a flattened list.
                      For example, if the ``weights_collection`` is the same as in the example
                      above, the flattened weights would be:

                      .. code-block:: python

                          weights_flattened = [
                              1000000, 5.7, 1000000, 1000000, 2.58, 1.05,
                              1000000, 1.4, 1000000, 0.8, 0.725
                          ]

        """
        if community_id is None:
            if as_type == "collection":
                return [
                    1.0 / (np.sqrt(distances) + DISTANCE_FACTOR)
                    for distances in self.distances_collection
                ]
            elif as_type == "array":
                return 1.0 / (np.sqrt(self.get_distances_array()) + DISTANCE_FACTOR)
            elif as_type == "flatten":
                weights_flattened = []
                for distances in self.distances_collection:
                    weights_flattened += (1.0 / (np.sqrt(distances) + DISTANCE_FACTOR)).tolist()
                return weights_flattened
        else:
            return 1.0 / (np.sqrt(self.distances_collection[community_id]) + DISTANCE_FACTOR)

    def get_distances_array_from_csr_array(self, csr_array, sorted=False):
        """Given a ``csr_matrix``, get a sorted distance array.

        Args:
            csr_array (scipy.sparse.csr_matrix): A compressed sparse row matrix with dimensions
                (n_communities, n_communities), containing the pruned distances.

        Returns:
            np.array: An array with dimensions (n_communities, k) distances to each of the
                k nearest neighbors for each data point, in order of proximity. For example, if my
                data has 5 communities, with ``k=3`` nearest neighbors:

                .. code-block:: python

                    distances_array = np.array([
                        [0.10, 0.03, 0.01],
                        [0.50, 0.80, 1.00],
                        [0.13, 0.15, 0.90],
                        [0.40, 0.51, 0.87],
                        [0.98, 1.56, 1.90]
                    ])

                It is possible that the number of nearest neighbors is different for each sample.
                If that is the case, the DISTANCE_PLACEHOLDER = 1000000 will be added to the
                missing values so that a 2D array can be created. For example:

                .. code-block:: python

                    distances_array = np.array([
                        [0.10, 0.03, 1000000],
                        [0.50, 1000000, 1000000],
                        [0.13, 0.15, 0.90],
                        [0.40, 0.51, 1000000],
                        [0.98, 1.56, 1.90]
                    ])
        """

        similarity_data = 1 / (csr_array.data + DISTANCE_FACTOR)**2
        distances_unsorted = np.split(similarity_data, csr_array.indptr)[1:-1]

        distances_array = self.get_distances_array(distances_collection=distances_unsorted)

        if sorted:
            return np.sort(distances_array, axis=1)
        else:
            return distances_array

    def get_distances_array(self, distances_collection=None):
        """Given a collection of distances, return it as a Numpy array.

        Args:
            distances_collection (list[np.array]): A list of Numpy arrays with the distances to the
                k nearest neighbors for each community, in order of proximity. Note that the
                value of k can be different for each community. For example, if my data has
                5 communities, the resulting ``distances_collection`` might be:

                .. code-block:: python

                    distances_collection = [
                        np.array([0.10, 0.03]),
                        np.array([0.50]),
                        np.array([0.13, 0.15, 0.90]),
                        np.array([0.40, 0.51]),
                        np.array([0.98, 1.56, 1.90])
                    ]

                If ``None``, use ``self.distances_collection``.

        Returns:
            np.array: An array with dimensions (n_communities, k) containing the distances to each
                of the k nearest neighbors for each community, in order of proximity. For example,
                if there are 5 communities, with ``k=3`` nearest neighbors:

                .. code-block:: python

                    distances_array = np.array([
                        [0.10, 0.03, 0.01],
                        [0.50, 0.80, 1.00],
                        [0.13, 0.15, 0.90],
                        [0.40, 0.51, 0.87],
                        [0.98, 1.56, 1.90]
                    ])

                It is possible that the number of nearest neighbors is different for each sample.
                If that is the case, the DISTANCE_PLACEHOLDER = 1000000 will be added to the
                missing values so that a 2D array can be created. For example:

                .. code-block:: python

                    distances_array = np.array([
                        [0.10, 0.03, 1000000],
                        [0.50, 1000000, 1000000],
                        [0.13, 0.15, 0.90],
                        [0.40, 0.51, 1000000],
                        [0.98, 1.56, 1.90]
                    ])
        """
        if distances_collection is None:
            distances_collection = self.distances_collection

        distances_array = []
        for distances in distances_collection:
            distances = np.insert(
                distances,
                len(distances),
                [DISTANCE_PLACEHOLDER]*(self.max_neighbors - len(distances))
            )
            distances_array.append(distances)

        distances_array = np.array(distances_array)
        return distances_array

    def get_distances_from_csr_array(self, csr_array):
        """Given a ``csr_matrix``, get a sorted list of distances.

        Args:
            csr_array (scipy.sparse.csr_matrix): A compressed sparse row matrix with dimensions
                (n_communities, n_communities), containing the distances.

        Returns:
            list[np.array]: A list of k x 1 Numpy vectors, giving the distances to each of the
                k nearest neighbors for each data point, in order of proximity. The value of k
                may be different for each sample. For example, if my data has 5 communities,
                the resulting list might be:

                .. code-block:: python

                    distances_collection = [
                        np.array([0.10, 0.03, 0.01]),
                        np.array([0.50, 0.80]),
                        np.array([0.13, 0.15, 0.90, 1.90]),
                        np.array([0.40, 0.51, 0.87]),
                        np.array([0.98, 1.56, 1.90])
                    ]
        """

        distances_array = self.get_distances_array_from_csr_array(csr_array, sorted=True)

        distances_collection = []
        for distances in list(distances_array):
            distances = np.delete(
                distances,
                np.where(distances == DISTANCE_PLACEHOLDER)
            )
            distances_collection.append(distances)

        return distances_collection

    def get_neighbors_array_from_csr_array(self, csr_array, sorted=False):
        """Given a ``csr_matrix``, get a sorted list of nearest neighbors.

        Args:
            csr_array (scipy.sparse.csr_matrix): A compressed sparse row matrix with dimensions
                (n_communities, n_communities), containing the pruned distances.

        Returns:
            np.array: An array with dimensions (n_communities, k) containing the
                k nearest neighbors for each community, in order of proximity. For example, if the
                data has 5 communities, with ``k=3`` nearest neighbors:

                .. code-block:: python

                    neighbors_array = np.array([
                        [0, 1, 2],
                        [1, 0, 3],
                        [2, 4, 0],
                        [3, 2, 4],
                        [4, 2, 3]
                    ])

                It is possible that the number of nearest neighbors is different for each sample.
                If that is the case, the NEIGHBOR_PLACEHOLDER = -1 will be added to the
                missing values so that a 2D array can be created. For example:

                .. code-block:: python

                    neighbors_array = np.array([
                        [0, 1, -1],
                        [1, -1, -1],
                        [2, 4, 0],
                        [3, 2, -1],
                        [4, 2, 3]
                    ])
        """

        neighbors_unsorted = np.split(csr_array.indices, csr_array.indptr)[1:-1]
        neighbors_array = self.get_neighbors_array(neighbors_collection=neighbors_unsorted)

        if sorted:
            distances_array = self.get_distances_array_from_csr_array(csr_array, sorted=False)
            sorted_indices = np.argsort(distances_array, axis=1)

            return np.take_along_axis(
                neighbors_array,
                sorted_indices,
                axis=1
            )
        else:
            return neighbors_array

    def get_neighbors_array(self, neighbors_collection=None):
        """Given a collection of neighbors, return it as a Numpy array.

        Args:
            neighbors_collection (list[np.array]): A list of Numpy arrays with the
                k nearest neighbors for each community, in order of proximity. Note that the
                value of k can be different for each community. For example, if my data has
                5 communities, the resulting ``neighbors_collection`` might be:

                .. code-block:: python

                    neighbors_collection = [
                        np.array([0, 3]),
                        np.array([1]),
                        np.array([2, 4, 0]),
                        np.array([3, 2]),
                        np.array([4, 2, 3])
                    ]

                If ``None``, use ``self.neighbors_collection``.

        Returns:
            np.array: An array with dimensions (n_communities, k) distances to each of the
                k nearest neighbors for each data point, in order of proximity. For example, if my
                data has 5 communities, with ``k=3`` nearest neighbors:

                .. code-block:: python

                    neighbors_array = np.array([
                        [0, 1, 2],
                        [1, 0, 3],
                        [2, 4, 0],
                        [3, 2, 4],
                        [4, 2, 3]
                    ])

                It is possible that the number of nearest neighbors is different for each sample.
                If that is the case, the NEIGHBOR_PLACEHOLDER = -1 will be added to the
                missing values so that a 2D array can be created. For example:

                .. code-block:: python

                    neighbors_array = np.array([
                        [0, 1, -1],
                        [1, -1, -1],
                        [2, 4, 0],
                        [3, 2, -1],
                        [4, 2, 3]
                    ])
        """
        if neighbors_collection is None:
            neighbors_collection = self.neighbors_collection

        neighbors_array = []
        for neighbors in neighbors_collection:
            neighbors = np.insert(
                neighbors,
                len(neighbors),
                [NEIGHBOR_PLACEHOLDER]*(self.max_neighbors - len(neighbors))
            )
            neighbors_array.append(neighbors)

        neighbors_array = np.array(neighbors_array)

        return neighbors_array

    def get_neighbors_from_csr_array(self, csr_array):
        """Given a ``csr_matrix``, get a sorted list of nearest neighbors.

        Args:
            csr_array (scipy.sparse.csr_matrix): A compressed sparse row matrix with dimensions
                (n_communities, n_communities), containing the pruned distances.

        Returns:
            list[np.array]: A list of k x 1 Numpy vectors, giving the k nearest neighbors
                for each data point, in order of proximity. The value of k may be different for
                each sample. For example, if my data has 5 communities, the resulting list might be:

                .. code-block:: python

                    neighbors_collection = [
                        np.array([0, 3]),
                        np.array([1]),
                        np.array([2, 4, 0]),
                        np.array([3, 2]),
                        np.array([4, 2, 3])
                    ]
        """

        neighbors_array = self.get_neighbors_array_from_csr_array(csr_array, sorted=True)

        neighbors_collection = []
        for neighbors in list(neighbors_array):
            neighbors = np.delete(
                neighbors,
                np.where(neighbors == NEIGHBOR_PLACEHOLDER)
            )
            neighbors_collection.append(neighbors)

        return neighbors_collection

    def get_edges(self):
        """Get the graph edges of the k nearest neighbors as ordered pairs.

        Returns:
            list[tuple]: A list of 2-tuples representing an edge, with length ``n_communities * k``.
                For example, if my data has 5 communities, with ``k=3`` nearest neighbors, and the
                resulting ``neighbors_array`` is:

                .. code-block:: python

                    neighbors_array = np.array([
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
        edges = []
        bar = Bar("Generating list of edges...", max=self.n_communities)
        for community_id, neighbors in zip(range(self.n_communities), self.neighbors_collection):
            edges += [(community_id, int(neighbor)) for neighbor in neighbors]
            bar.next()
        bar.finish()
        return edges
