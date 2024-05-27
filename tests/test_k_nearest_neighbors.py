import pytest
import numpy as np
from parc.k_nearest_neighbors import NearestNeighbors, NearestNeighborsCollection, \
    DISTANCE_PLACEHOLDER, NEIGHBOR_PLACEHOLDER
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

WEIGHTS_COLLECTION = [
    np.array([1000000, 3.162, 2.236]),
    np.array([1000000, 3.162, 2.182, 2.132]),
    np.array([1000000, 2.294]),
    np.array([1000000, 2.085, 1.826, 1.581]),
    np.array([1000000, 2.294, 2.236]),
    np.array([1000000, 2.236, 2.182, 1.924])
]

EDGES = [
    (0, 0), (0, 1), (0, 2),
    (1, 1), (1, 0), (1, 5), (1, 4),
    (2, 2), (2, 4),
    (3, 3), (3, 2), (3, 4), (3, 0),
    (4, 4), (4, 2), (4, 5),
    (5, 5), (5, 4), (5, 1), (5, 0)
]

NEIGHBOR_ARRAY = np.array([
    [0, 1, 2, NEIGHBOR_PLACEHOLDER],
    [1, 0, 5, 4],
    [2, 4, NEIGHBOR_PLACEHOLDER, NEIGHBOR_PLACEHOLDER],
    [3, 2, 4, 0],
    [4, 2, 5, NEIGHBOR_PLACEHOLDER],
    [5, 4, 1, 0]
])

DISTANCE_ARRAY = np.array([
    [0.00, 0.10, 0.20, DISTANCE_PLACEHOLDER],
    [0.00, 0.10, 0.21, 0.22],
    [0.00, 0.19, DISTANCE_PLACEHOLDER, DISTANCE_PLACEHOLDER],
    [0.00, 0.23, 0.30, 0.40],
    [0.00, 0.19, 0.20, DISTANCE_PLACEHOLDER],
    [0.00, 0.20, 0.21, 0.27]
])

WEIGHTS_ARRAY = np.array([
    [1000000, 3.162, 2.236, 0.001],
    [1000000, 3.162, 2.182, 2.132],
    [1000000, 2.294, 0.001, 0.001],
    [1000000, 2.085, 1.826, 1.581],
    [1000000, 2.294, 2.236, 0.001],
    [1000000, 2.236, 2.182, 1.924]
])

NEIGHBORS_FLATTENED = [
    0, 1, 2,
    1, 0, 5, 4,
    2, 4,
    3, 2, 4, 0,
    4, 2, 5,
    5, 4, 1, 0
]

DISTANCES_FLATTENED = [
    0.00, 0.10, 0.20,
    0.00, 0.10, 0.21, 0.22,
    0.00, 0.19,
    0.00, 0.23, 0.30, 0.40,
    0.00, 0.19, 0.20,
    0.00, 0.20, 0.21, 0.27
]

WEIGHTS_FLATTENED = [
    1000000, 3.162, 2.236,
    1000000, 3.162, 2.182, 2.132,
    1000000, 2.294,
    1000000, 2.085, 1.826, 1.581,
    1000000, 2.294, 2.236,
    1000000, 2.236, 2.182, 1.924
]


@pytest.mark.parametrize(
    "neighbors_collection, distances_collection",
    [
        (
            NEIGHBORS_COLLECTION, DISTANCES_COLLECTION
        )
    ]
)
@pytest.mark.parametrize(
    "expected_neighbors_list, as_type",
    [
        (NEIGHBOR_ARRAY, "array"),
        (NEIGHBORS_FLATTENED, "flatten"),
        (NEIGHBORS_COLLECTION, "collection")
    ]
)
def test_nearest_neighbors_collection_get_neighbors(
    neighbors_collection, distances_collection,
    expected_neighbors_list, as_type
):

    nearest_neighbors_collection = NearestNeighborsCollection(
        neighbors_collection=neighbors_collection,
        distances_collection=distances_collection
    )

    neighbors_list = nearest_neighbors_collection.get_neighbors(as_type=as_type)
    for neighbors, expected_neighbors in zip(neighbors_list, expected_neighbors_list):
        if isinstance(neighbors, np.ndarray):
            assert list(neighbors) == list(expected_neighbors)
        else:
            assert neighbors == expected_neighbors


@pytest.mark.parametrize(
    "neighbors_collection, distances_collection",
    [
        (
            NEIGHBORS_COLLECTION, DISTANCES_COLLECTION
        )
    ]
)
@pytest.mark.parametrize(
    "expected_distances_list, as_type",
    [
        (DISTANCE_ARRAY, "array"),
        (DISTANCES_FLATTENED, "flatten"),
        (DISTANCES_COLLECTION, "collection")
    ]
)
def test_nearest_neighbors_collection_get_distances(
    neighbors_collection, distances_collection,
    expected_distances_list, as_type
):

    nearest_neighbors_collection = NearestNeighborsCollection(
        neighbors_collection=neighbors_collection,
        distances_collection=distances_collection
    )

    distances_list = nearest_neighbors_collection.get_distances(as_type=as_type)
    for distances, expected_distances in zip(distances_list, expected_distances_list):
        if isinstance(distances, np.ndarray):
            assert list(distances) == list(expected_distances)
        else:
            assert distances == expected_distances


@pytest.mark.parametrize(
    "neighbors_collection, distances_collection",
    [
        (
            NEIGHBORS_COLLECTION, DISTANCES_COLLECTION
        )
    ]
)
@pytest.mark.parametrize(
    "expected_weights_list, as_type",
    [
        (WEIGHTS_ARRAY, "array"),
        (WEIGHTS_FLATTENED, "flatten"),
        (WEIGHTS_COLLECTION, "collection")
    ]
)
def test_nearest_neighbors_collection_get_weights(
    neighbors_collection, distances_collection,
    expected_weights_list, as_type
):

    nearest_neighbors_collection = NearestNeighborsCollection(
        neighbors_collection=neighbors_collection,
        distances_collection=distances_collection
    )

    weights_list = nearest_neighbors_collection.get_weights(as_type=as_type)
    for weights, expected_weights in zip(weights_list, expected_weights_list):
        if isinstance(weights, np.ndarray):
            assert list(np.round(weights, 3)) == list(expected_weights)
        else:
            assert np.round(weights, 3) == expected_weights


@pytest.mark.parametrize(
    "neighbors_collection, distances_collection, expected_edges",
    [
        (
            NEIGHBORS_COLLECTION, DISTANCES_COLLECTION, EDGES
        )
    ]
)
def test_nearest_neighbors_collection_get_edges(
    neighbors_collection, distances_collection, expected_edges
):

    nearest_neighbors_collection = NearestNeighborsCollection(
        neighbors_collection=neighbors_collection,
        distances_collection=distances_collection
    )

    edges = nearest_neighbors_collection.get_edges()
    for edge, expected_edge in zip(edges, expected_edges):
        assert edge == expected_edge
