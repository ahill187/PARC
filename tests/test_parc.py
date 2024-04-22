import pytest
from sklearn import datasets
import time
from parc._parc import PARC
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


@pytest.mark.parametrize(
    "dataset_name, f1_mean, f1_accumulated, run_time",
    [
        ("iris_data", 0.9, 0.9, 1),
        ("forest_data", 0.6, 0.6, 10)
    ]
)
@pytest.mark.parametrize(
    "targets_exist",
    [
        (True),
        (False)
    ]
)
def test_parc_run_parc(request, dataset_name, f1_mean, f1_accumulated, run_time, targets_exist):
    x_data, y_data = request.getfixturevalue(dataset_name)
    if targets_exist:
        parc_model = PARC(x_data, y_data_true=y_data)
    else:
        parc_model = PARC(x_data)
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
