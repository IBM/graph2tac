import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force tests to run on CPU

import pytest
from pathlib import Path
import tensorflow as tf

from tests.integration.pipeline import run_tfgnn_training, run_hmodel_training, run_predict_server, assert_results_match_expected

# this helps with determinism
tf.config.experimental.enable_op_determinism()

REL_ERROR_TOLERANCE = 1e-4

# automatically find parameters for tests to run
TESTDIR = Path(__file__).resolve().parent.parent
TESTDATADIR = TESTDIR / "data"
DATASET_PARAMS_PAIRS = [(d.parent.parent.name, d.name) for d in TESTDATADIR.glob("*/params/*")]

@pytest.mark.filterwarnings("ignore:Converting sparse IndexedSlices")  # have pytest ignore this warning
@pytest.mark.filterwarnings("ignore:NumPy will stop allowing conversion of out-of-bound Python integers to integer arrays")  # have pytest ignore this warning
@pytest.mark.parametrize("dataset,params", DATASET_PARAMS_PAIRS)
def test_pipeline(tmp_path: Path, dataset: str, params: str, overwrite: bool):
    """
    Test the full training pipeline

    :param tmp_path: Temporary path provided by pytest
    :param dataset: Name of the dataset to test
    :param params: Name of the directory where the params and expected results are stored for this test
    :param overwrite: Whether to overwrite the expected test data
    """
    params_dir = TESTDATADIR / dataset / "params" / params
    
    # train model and test results
    data_dir = TESTDATADIR / dataset / "dataset"
    if (params_dir / "hmodel.yml").exists():
        # train hmodel
        training_results = run_hmodel_training(tmp_path, data_dir, params_dir)
    else:
        # train tfgnn model
        training_results = run_tfgnn_training(tmp_path, data_dir, params_dir)
    assert_results_match_expected(
        results=training_results,
        expected_results_file = params_dir / "expected.yaml",
        results_type="Training",
        rel_error_tolerance=REL_ERROR_TOLERANCE,
        overwrite=overwrite
    )

    # predict
    record_file = params_dir / "record_file.bin"
    if not record_file.exists(): 
        return
    predict_server_results = run_predict_server(tmp_path, record_file, params_dir)
    assert_results_match_expected(
        results=predict_server_results,
        expected_results_file = params_dir / "expected_predict_server.yaml",
        results_type="Predict Server",
        rel_error_tolerance=REL_ERROR_TOLERANCE,
        overwrite=overwrite
    )
