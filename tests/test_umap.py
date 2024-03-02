import pytest
from sklearn import datasets
from parc import run_umap_hnsw, PARC


@pytest.fixture
def iris_data():
    iris = datasets.load_iris()
    x_data = iris.data
    y_data = iris.target
    return x_data, y_data


def test_parc_run_umap_hnsw(iris_data):
    x_data, y_data = iris_data
    parc_model = PARC(x_data, y_data_true=y_data)
    parc_model.run_parc()

    graph = parc_model.create_knn_graph()
    x_umap = run_umap_hnsw(x_data, graph)
    assert x_umap.shape == (150, 2)