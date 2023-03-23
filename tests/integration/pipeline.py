"""
Supporting code for testing the training and benchmarking pipeline
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force tests to run on CPU

import json
from pathlib import Path
import pytest
import sys
from unittest.mock import patch
import tensorflow as tf
from typing import Any
import warnings

# this helps with determinism
tf.config.experimental.enable_op_determinism()

def overwrite_expected_results(results: dict, path: Path):
    warnings.warn(f"Overwriting expected results in {path}")
    with path.open("w") as f:
        json.dump(results, f)

def get_expected_results(path: Path):
    with path.open("r") as f:
        expected_results = json.load(f)
    return expected_results

def approximate(results: dict | list | float | Any, rel_error_tolerance: float) -> dict | list | float | Any:
    """Recursively go through JSON compatible object and replace floats with pytest.approx"""
    if isinstance(results, dict):
        return {k : approximate(v, rel_error_tolerance) for k,v in results.items()}
    elif isinstance(results, list):
        return [approximate(v, rel_error_tolerance)  for v in results]
    elif isinstance(results, float):
        return pytest.approx(results, rel=rel_error_tolerance)
    else: # bool, str, int, None, etc
        return results 

def run_tfgnn_training(tmp_path: Path, data_dir: Path, params_dir: Path) -> dict:
    """Run training and return results for comparison"""
    import graph2tac.tfgnn.train  # put import here so it doesn't break other tests if it crashes

    training_args = ["<program>",
        "--data-dir", data_dir,
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
    return results

def run_predict_server(tmp_path: Path, record_file: Path) -> dict:
    """Run training and return results for comparison"""
    import graph2tac.loader.predict_server  # put import here so it doesn't break other tests if it crashes

    benchmark_args = ["<program>",
        "--arch", "tfgnn",
        "--log_level", "info",
        "--tf_log_level", "critical",
        "--tactic_expand_bound", "8",
        "--total_expand_bound", "10",
        "--search_expand_bound", "4",
        "--update_all_definitions",
        "--model", tmp_path / "log",
        "--replay", record_file
    ]
    # use context manager to pass command line arguments to our main method
    with patch.object(sys, 'argv', [str(a) for a in benchmark_args]):
        history = graph2tac.loader.predict_server.main()
    return history.data

def assert_results_match_expected(results: dict, expected_results_file: Path, results_type: str, rel_error_tolerance: float, overwrite: bool):
    approx_results = approximate(results, rel_error_tolerance=rel_error_tolerance)
    if overwrite:
        if not expected_results_file.exists():
            overwrite_expected_results(results, expected_results_file)
        else:
            expected_results = get_expected_results(expected_results_file)
            approx_expected_results = approximate(expected_results, rel_error_tolerance=rel_error_tolerance)
            if approx_results != approx_expected_results:
                overwrite_expected_results(results, expected_results_file)

    assert expected_results_file.exists(), f"No {results_type} expected results file. To create one, rerun this test with `--overwrite`."
    expected_results = get_expected_results(expected_results_file)
    approx_expected_results = approximate(expected_results, rel_error_tolerance=rel_error_tolerance)
    assert approx_results == approx_expected_results, (
        f"{results_type} statistics don't match expected results. To overwrite expected results, rerun this test with `--overwrite`."
    )