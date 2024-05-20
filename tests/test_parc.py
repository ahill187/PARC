import pytest
from sklearn import datasets
import numpy as np
import time
from parc._parc import PARC
from parc.k_nearest_neighbors import NearestNeighborsCollection
from parc.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture
def iris_data():
    iris = datasets.load_iris()
    x_data = iris.data
    y_data = iris.target
    return x_data, y_data


@pytest.fixture
def forest_data():
    forests = datasets.fetch_covtype()
    x_data = forests.data[list(range(0, 30000)), :]
    y_data = forests.target[list(range(0, 30000))]
    return x_data, y_data


NEIGHBOR_ARRAY = np.array([
    [0, 1, 2, 4],
    [1, 0, 5, 4],
    [2, 4, 0, 1],
    [3, 2, 4, 0],
    [4, 2, 5, 1],
    [5, 4, 1, 0]
])

EXPECTED_NEIGHBOR_ARRAY = np.array([
    [1, -1],
    [0, -1],
    [4, -1],
    [2, 4],
    [2, 5],
    [4, 1]
])

EXPECTED_DISTANCE_ARRAY = np.array([
    [0.10, 1000000],
    [0.10, 1000000],
    [0.19, 1000000],
    [0.23, 0.30],
    [0.19, 0.20],
    [0.20, 0.21]
])

DISTANCE_ARRAY = np.array([
    [0.00, 0.10, 0.20, 0.25],
    [0.00, 0.10, 0.21, 0.22],
    [0.00, 0.19, 0.20, 0.22],
    [0.00, 0.23, 0.30, 0.40],
    [0.00, 0.19, 0.20, 0.25],
    [0.00, 0.20, 0.21, 0.27]
])


def test_parc_run_umap_hnsw():
    iris = datasets.load_iris()
    x_data = iris.data
    y_data = iris.target

    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    parc_model.run_parc()

    graph = parc_model.create_knn_graph()
    x_umap = parc_model.run_umap_hnsw(x_data, graph)
    assert x_umap.shape == (150, 2)


@pytest.mark.parametrize(
    "dataset_name, neighbor_array, distance_array, l2_std_factor",
    [
        (
            "iris_data", NEIGHBOR_ARRAY, DISTANCE_ARRAY, 0.5
        )
    ]
)
@pytest.mark.parametrize(
    "do_prune_local, expected_neighbor_array, expected_distance_array",
    [
        (False, NEIGHBOR_ARRAY, DISTANCE_ARRAY),
        (True, EXPECTED_NEIGHBOR_ARRAY, EXPECTED_DISTANCE_ARRAY),
        (None, EXPECTED_NEIGHBOR_ARRAY, EXPECTED_DISTANCE_ARRAY)
    ]
)
def test_parc_prune_local(
    request, dataset_name, neighbor_array, distance_array, l2_std_factor,
    do_prune_local, expected_neighbor_array, expected_distance_array
):
    x_data, y_data = request.getfixturevalue(dataset_name)

    parc_model = PARC(
        x_data=x_data, y_data_true=y_data, l2_std_factor=l2_std_factor,
        do_prune_local=do_prune_local
    )

    csr_array = parc_model.prune_local(
        NearestNeighborsCollection(
            neighbors_collection=neighbor_array,
            distances_collection=distance_array
        )
    )

    nearest_neighbors_collection = NearestNeighborsCollection(csr_array=csr_array)

    np.testing.assert_array_equal(
        nearest_neighbors_collection.get_neighbors(as_type="array"),
        expected_neighbor_array
    )
    np.testing.assert_array_equal(
        np.round(nearest_neighbors_collection.get_distances(as_type="array"), decimals=3),
        expected_distance_array
    )


@pytest.mark.parametrize(
    "dataset_name, large_community_factor, knn, f1_mean, f1_accumulated, run_time",
    [
        ("iris_data", 0.4, 30, 0.9, 0.9, 1),
        ("forest_data", 0.019, 30, 0.6, 0.7, 10),
        ("forest_data", 0.4, 30, 0.6, 0.7, 10),
        ("forest_data", 0.4, 100, 0.5, 0.6, 20)
    ]
)
@pytest.mark.parametrize(
    "targets_exist",
    [
        (True),
        (False)
    ]
)
def test_parc_run_parc(
    request, dataset_name, large_community_factor, knn, f1_mean, f1_accumulated,
    run_time, targets_exist
):
    x_data, y_data = request.getfixturevalue(dataset_name)
    if targets_exist:
        parc_model = PARC(
            x_data=x_data, y_data_true=y_data, large_community_factor=large_community_factor,
            knn=knn
        )
    else:
        parc_model = PARC(x_data=x_data, large_community_factor=large_community_factor, knn=knn)
    start_time = time.time()
    parc_model.run_parc()
    end_time = time.time()
    assert end_time - start_time < run_time
    if targets_exist:
        assert parc_model.f1_mean >= f1_mean
        assert parc_model.f1_accumulated >= f1_accumulated
    else:
        assert parc_model.f1_mean == 0
        assert parc_model.f1_accumulated == 0
    assert len(parc_model.y_data_pred) == y_data.shape[0]
