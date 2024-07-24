
import numpy as np
from parc.logger import get_logger

logger = get_logger(__name__)


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
