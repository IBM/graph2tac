import pytest

# add an option to pytest to overwrite expected data for a test
def pytest_addoption(parser):
    parser.addoption("--overwrite", action="store_true", default=False, help="Overwrite test data with new results")
    parser.addoption("--retrain", action="store_true", default=False, help="Retrain saved models")

@pytest.fixture(scope='session')
def overwrite(request):
    return request.config.getoption("--overwrite")

@pytest.fixture(scope='session')
def retrain(request):
    return request.config.getoption("--retrain")