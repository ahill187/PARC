import pytest
from sklearn import datasets
import time
from parc._parc import PARC


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
