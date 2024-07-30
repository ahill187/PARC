import pytest
from sklearn import datasets
import numpy as np
import igraph
import scipy
import hnswlib
import time
from parc._parc import PARC
from parc.logger import get_logger
from tests.variables import NEIGHBOR_ARRAY_L2, NEIGHBOR_ARRAY_COSINE

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


@pytest.mark.parametrize(
    "dataset_name, knn, distance_metric, expected_neighbor_array",
    [
        ("iris_data", 2, "l2", NEIGHBOR_ARRAY_L2),
        ("iris_data", 2, "cosine", NEIGHBOR_ARRAY_COSINE)
    ]
)
def test_parc_make_knn_struct(
    request, dataset_name, knn, distance_metric, expected_neighbor_array
):
    x_data, y_data = request.getfixturevalue(dataset_name)
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    knn_struct = parc_model.make_knn_struct(x_data=x_data, distance_metric=distance_metric)
    assert isinstance(knn_struct, hnswlib.Index)
    neighbor_array, distance_array = knn_struct.knn_query(x_data, k=knn)
    assert neighbor_array.shape == (x_data.shape[0], knn)
    assert distance_array.shape == (x_data.shape[0], knn)
    n_diff = np.count_nonzero(expected_neighbor_array != neighbor_array)
    assert n_diff / x_data.shape[0] < 0.1


@pytest.mark.parametrize(
    "dataset_name, knn, l2_std_factor, n_edges",
    [
        ("iris_data", 100, 3.0, 14625),
        ("iris_data", 100, 100.0, 14850),
        ("iris_data", 100, -100.0, 0)
    ]
)
def test_parc_prune_local(
    request, dataset_name, knn, l2_std_factor, n_edges
):
    x_data, y_data = request.getfixturevalue(dataset_name)
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    knn_struct = parc_model.make_knn_struct(x_data=x_data)
    neighbor_array, distance_array = knn_struct.knn_query(x_data, k=knn)
    csr_array = parc_model.prune_local(neighbor_array, distance_array, l2_std_factor)
    input_nodes, output_nodes = csr_array.nonzero()
    edges = list(zip(input_nodes, output_nodes))
    assert isinstance(csr_array, scipy.sparse.csr_matrix)
    assert len(edges) == n_edges


@pytest.mark.parametrize(
    "dataset_name, knn, jac_threshold_type, jac_std_factor, jac_weighted_edges, n_edges",
    [
        ("iris_data", 100, "median", 0.3, True, 3679),
        ("iris_data", 100, "mean", 0.3, False, 3865),
        ("iris_data", 100, "mean", -1000, False, 0),
        ("iris_data", 100, "mean", 1000, False, 8558)
    ]
)
def test_parc_prune_global(
    request, dataset_name, knn, jac_threshold_type, jac_std_factor, jac_weighted_edges, n_edges
):
    x_data, y_data = request.getfixturevalue(dataset_name)
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    knn_struct = parc_model.make_knn_struct()
    neighbor_array, distance_array = knn_struct.knn_query(x_data, k=knn)
    csr_array = parc_model.prune_local(neighbor_array, distance_array)
    graph_pruned = parc_model.prune_global(
        csr_array=csr_array,
        jac_threshold_type=jac_threshold_type,
        jac_std_factor=jac_std_factor,
        jac_weighted_edges=jac_weighted_edges,
        n_samples=x_data.shape[0]
    )
    assert isinstance(graph_pruned, igraph.Graph)
    assert graph_pruned.ecount() == n_edges
    assert graph_pruned.vcount() == x_data.shape[0]


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
