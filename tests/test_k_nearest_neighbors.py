import pytest
import numpy as np
from parc.k_nearest_neighbors import NearestNeighbors, NearestNeighborsCollection
from parc.logger import get_logger

logger = get_logger(__name__)


NEIGHBORS_COLLECTION = [
    np.array([0, 1, 2]),
    np.array([1, 0, 5, 4]),
    np.array([2, 4]),
    np.array([3, 2, 4, 0]),
    np.array([4, 2, 5]),
    np.array([5, 4, 1, 0])
]

DISTANCES_COLLECTION = [
    np.array([0.00, 0.10, 0.20]),
    np.array([0.00, 0.10, 0.21, 0.22]),
    np.array([0.00, 0.19]),
    np.array([0.00, 0.23, 0.30, 0.40]),
    np.array([0.00, 0.19, 0.20]),
    np.array([0.00, 0.20, 0.21, 0.27])
]

COLLECTION = [
    NearestNeighbors(
        community_id=i, neighbors=NEIGHBORS_COLLECTION[i], distances=DISTANCES_COLLECTION[i]
    )
    for i in range(len(NEIGHBORS_COLLECTION))
]


@pytest.mark.parametrize(
    "neighbors_collection, distances_collection, collection, max_neighbors, n_edges, n_communities",
    [
        (NEIGHBORS_COLLECTION, DISTANCES_COLLECTION, COLLECTION, 4, 20, 6)
    ]
)
def test_nearest_neighbors_collection_constructor(
    neighbors_collection, distances_collection, collection, max_neighbors, n_edges, n_communities
):
    nnc = NearestNeighborsCollection(
        neighbors_collection=neighbors_collection,
        distances_collection=distances_collection
    )
    np.testing.assert_array_equal(nnc.collection[0].neighbors, collection[0].neighbors)
    assert nnc.max_neighbors == max_neighbors
    assert nnc.n_edges == n_edges
    assert nnc.n_communities == n_communities
