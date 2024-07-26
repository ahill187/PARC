
import numpy as np
from parc.logger import get_logger

logger = get_logger(__name__)

NEIGHBOR_PLACEHOLDER = -1


class NearestNeighbors:
    """The ids and distances to the nearest neighbors of a community.

    Attributes:
        community_id (int): The id of the community.
        neighbors (np.ndarray): a k x 1 Numpy vector with the community ids of the k nearest
            neighbors for this community, in order of proximity. For example, if there are ``k=3``
            nearest neighbors:

            .. code-block:: python

                neighbors = np.array([2, 4, 0])

        distances (np.ndarray): A k x 1 Numpy vector, giving the distances to each of the
            k nearest neighbors for each community, in order of proximity.
            For example, if there are ``k=3`` nearest neighbors:

            .. code-block:: python

                distances_collection = np.array([0.13, 0.15, 0.90])
    """

    def __init__(
        self,
        community_id: int,
        neighbors: list[int] | list[float] | np.ndarray,
        distances: list[float] | np.ndarray
    ):
        self.community_id = community_id
        self.neighbors = neighbors
        self.distances = distances

    @property
    def neighbors(self) -> np.ndarray:
        return self._neighbors

    @neighbors.setter
    def neighbors(self, neighbors: list[int] | list[float] | np.ndarray):
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
    def distances(self) -> np.ndarray:
        return self._distances

    @distances.setter
    def distances(self, distances: list[float] | np.ndarray):
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

    def __len__(self):
        return len(self.neighbors)


class NearestNeighborsCollection:
    """A collection of nearest neighbors and their distances for a graph.

    Attributes:
        collection (list[NearestNeighbors]): a list of k x 1 Numpy vectors with the
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
        n_edges (int): The total number of edges in the graph.
    """

    def __init__(
        self,
        collection: list[NearestNeighbors] | None = None,
        neighbors_collection: list[list[int]] | list[list[float]] | np.ndarray | None = None,
        distances_collection: list[list[float]] | np.ndarray | None = None
    ):
        """Initialize the NearestNeighborsCollection object.

        Args:
            collection: A list of ``NearestNeighbors`` objects.
            neighbors_collection: A list of k x 1 Numpy vectors with the
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
            distances_collection: A list of k x 1 Numpy vectors, giving the distances to each of
                the k nearest neighbors for each community, in order of proximity. Note that the
                value of k may be different for each community.
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

        self.max_neighbors = 0
        self.n_communities = 0
        self.n_edges = 0
        if collection is None and neighbors_collection is None and distances_collection is None:
            raise ValueError(
                "User must provide either: a) collection or b) neighbors_collection "
                "and distances_collection."
            )
        elif collection is not None:
            self.collection = collection
        else:
            neighbors_collection = self._check_neighbors_collection(neighbors_collection)
            distances_collection = self._check_distances_collection(distances_collection)
            if len(neighbors_collection) != len(distances_collection):
                raise ValueError(
                    f"Neighbors and distances collections must have the same length; "
                    f"got {len(neighbors_collection)} neighbors and {len(distances_collection)} "
                    "distances."
                )
            else:
                collection = []
                for index, neighbors, distances in zip(
                    range(len(neighbors_collection)), neighbors_collection, distances_collection
                ):
                    if len(neighbors) != len(distances):
                        raise ValueError(
                            f"Neighbors and distances must have the same length; "
                            f"got {len(neighbors)} neighbors and {len(distances)} distances."
                        )
                    else:
                        collection.append(
                            NearestNeighbors(
                                community_id=index,
                                neighbors=neighbors,
                                distances=distances
                            )
                        )
                self.collection = collection

    @property
    def collection(self) -> list[NearestNeighbors]:
        return self._collection

    @collection.setter
    def collection(self, collection: list[NearestNeighbors]):
        if isinstance(collection, list) and all(
            isinstance(nearest_neighbors, NearestNeighbors) for nearest_neighbors in collection
        ):
            self._collection = collection
            self.n_communities = len(collection)
            self.n_edges = int(np.sum([len(nearest_neighbors) for nearest_neighbors in collection]))
            self.max_neighbors = self.get_max_neighbors(collection)
        else:
            raise TypeError(
                f"Collection must be a list of NearestNeighbors; "
                f"got variable of type {type(collection)}"
            )

    def _check_neighbors_collection(self, neighbors_collection):
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
                return list(neighbors_collection)
        elif isinstance(neighbors_collection, list):
            return neighbors_collection
        else:
            raise TypeError(
                f"Neighbors must be an n x m array or a list of n x 1 arrays; "
                f"got variable of type {type(neighbors_collection)}"
            )

    def _check_distances_collection(self, distances_collection):
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
                return list(distances_collection)
        elif isinstance(distances_collection, list):
            return distances_collection
        else:
            raise TypeError(
                f"Distances must be an n x m array or a list of n x 1 arrays; "
                f"got variable of type {type(distances_collection)}"
            )

    def get_max_neighbors(self, collection: list[NearestNeighbors]) -> int:
        """Given a collection of nearest neighbors, get the maximum value for k.

        Args:
            collection (list[NearestNeighbors]): a list of ``NearestNeighbors`` objects with the
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
        for nearest_neighbors in collection:
            if len(nearest_neighbors) > max_neighbors:
                max_neighbors = len(nearest_neighbors)
        return max_neighbors

    def get_neighbors_collection(self) -> list[np.ndarray]:
        """Return a list of k nearest neighbors.

        Returns:
            list[np.ndarray]: An array with dimensions (n_communities, k) representing the
                k nearest neighbors for each data point, in order of proximity. For example, if my
                data has 5 communities, with ``k=3`` nearest neighbors:

                .. code-block:: python

                    neighbors_collection = [
                        np.array([0, 1, 2]),
                        np.array([1, 0, 3]),
                        np.array([2, 4, 0]),
                        np.array([3, 2, 4]),
                        np.array([4, 2, 3])
                    ]

                It is possible that the number of nearest neighbors is different for each sample.
                For example:

                .. code-block:: python

                    neighbors_collection = [
                        np.array([0, 1]),
                        np.array([1]),
                        np.array([2, 4, 0]),
                        np.array([3, 2]),
                        np.array([4, 2, 3])
                    ]

        """
        return [nearest_neighbors.neighbors for nearest_neighbors in self.collection]

    def get_neighbors_array(self) -> np.ndarray:
        """Given a collection of neighbors, return it as a Numpy array.

        Returns:
            np.ndarray: An array with dimensions (n_communities, k) representing the
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
        neighbors_array = []
        for neighbors in self.get_neighbors_collection():
            neighbors = np.insert(
                neighbors,
                len(neighbors),
                [NEIGHBOR_PLACEHOLDER] * (self.max_neighbors - len(neighbors))
            )
            neighbors_array.append(neighbors)

        neighbors_array = np.array(neighbors_array)

        return neighbors_array
