import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force tests to run on CPU

import json
from pathlib import Path
import pytest
import sys
from unittest.mock import patch
import warnings

import graph2tac.tfgnn.train

REL_ERROR_TOLERANCE = 1e-3

# automatically find parameters for tests to run
TESTDIR = Path(__file__).resolve().parent.parent
TESTDATADIR = TESTDIR / "data"
DATASET_PARAMS_PAIRS = [(d.parent.parent.name, d.name) for d in TESTDATADIR.glob("*/params/*")]

def overwrite_expected_results(results: dict, path: Path):
    warnings.warn(f"Overwriting expected results in {path}")
    with path.open("w") as f:
        json.dump(results, f)

def get_approx_expected_results(path: Path):
    with path.open("r") as f:
        expected_results = json.load(f)
    # pytest.approx only works one level deep
    return {k:pytest.approx(v, rel=REL_ERROR_TOLERANCE) for k,v in expected_results.items()}
    
@pytest.mark.filterwarnings("ignore:Converting sparse IndexedSlices")  # have pytest ignore this warning
@pytest.mark.parametrize("dataset,params", DATASET_PARAMS_PAIRS)
def test_training(tmp_path: Path, dataset: str, params: str, overwrite: bool):
    """
    Test the full training pipeline

    :param tmp_path: Temporary path provided by pytest
    :param dataset: Name of the dataset to test
    :param params: Name of the directory where the params and expected results are stored for this test
    :param overwrite: Whether to overwrite the expected test data
    """
    
    params_dir = TESTDATADIR / dataset / "params" / params

    training_args = ["<program>",
        "--data-dir", TESTDATADIR / dataset / "dataset",
        "--dataset-config", params_dir / "dataset_config.yml",
        "--prediction-task-config", params_dir / "global_argument_prediction.yml",
        "--definition-task-config", params_dir / "definition_task.yml",
        "--trainer-config", params_dir / "trainer_config.yml",
        "--run-config", params_dir / "run_config.yml",
        "--log", tmp_path / "log",
    ]
    # use context manager to pass command line arguments to our main method
    with patch.object(sys, 'argv', [str(a) for a in training_args]):
        history = graph2tac.tfgnn.train.main()
    # remove results which are not stable or not serializable
    results = {k:v for k,v in history.history.items() if k not in ["epoch_duration", "learning_rate"]}

    expected_results_file = params_dir / "expected.json"

    if overwrite:
        if not expected_results_file.exists():
            overwrite_expected_results(results, expected_results_file)
        else:
            approx_expected_results = get_approx_expected_results(expected_results_file)
            if results != approx_expected_results:
                overwrite_expected_results(results, expected_results_file)

    assert expected_results_file.exists(), "No expected results file. To create one, rerun this test with `--overwrite`."
    approx_expected_results = get_approx_expected_results(expected_results_file)
    assert results == approx_expected_results, (
        "Training statistics don't match expected results. To overwrite expected results, rerun this test with `--overwrite`."
    )