import pytest
from sklearn import datasets
from parc._parc import PARC


@pytest.fixture
def iris_data():
    iris = datasets.load_iris()
    x_data = iris.data
    y_data = iris.target
    return x_data, y_data


def test_parc_run_umap_hnsw(iris_data):
    x_data, y_data = iris_data
    parc_model = PARC(x_data, true_label=y_data)
    parc_model.run_PARC()

    graph = parc_model.knngraph_full()
    x_umap = parc_model.run_umap_hnsw(x_data, graph)
    assert x_umap.shape == (150, 2)


def test_parc_run_PARC(iris_data):
    x_data, y_data = iris_data
    parc_model = PARC(x_data, true_label=y_data)
    parc_model.run_PARC()

    assert len(parc_model.labels) == 150
