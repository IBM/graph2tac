import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force tests to run on CPU

import pytest
from pathlib import Path
import tensorflow as tf

from tests.integration.pipeline import ExpectedResults, Pipeline, ParamSets

# this helps with determinism
tf.config.experimental.enable_op_determinism()

REL_ERROR_TOLERANCE = 1e-4



@pytest.mark.filterwarnings("ignore:Converting sparse IndexedSlices")  # have pytest ignore this warning
@pytest.mark.parametrize("dataset,params", ParamSets.params_for_step("hmodel"))
def test_training_hmodel(request: pytest.FixtureRequest, tmp_path: Path, dataset: str, params: str, overwrite: bool):
    """
    Test the full pipeline for training an hmodel

    :param request: Metadata provided by pytest
    :param tmp_path: Temporary path provided by pytest
    :param dataset: Name of the dataset to test
    :param params: Name of the directory where the params and expected results are stored for this test
    :param overwrite: Whether to overwrite the expected test data
    """
    params_dir = ParamSets.params_dir(dataset, params)
    data_dir = ParamSets.dataset_dir(dataset)
    results, _ = Pipeline.run_training(
        tmp_path=tmp_path,
        data_dir=data_dir,
        params_dir=params_dir,
        cache=request.config.cache,
        use_cached_results=False,
        retrain=False
    )
    ExpectedResults.assert_results_match_expected(
        results=results,
        expected_results_file = params_dir / "expected.yaml",
        results_type="Training",
        rel_error_tolerance=REL_ERROR_TOLERANCE,
        overwrite=overwrite
    )

@pytest.mark.filterwarnings("ignore:Converting sparse IndexedSlices")  # have pytest ignore this warning
@pytest.mark.parametrize("dataset,params", ParamSets.params_for_step("tfgnn"))
def test_training_tfgnn(request: pytest.FixtureRequest, tmp_path: Path, dataset: str, params: str, overwrite: bool):
    """
    Test the full pipeline for training the tfgnn model

    :param request: Metadata provided by pytest
    :param tmp_path: Temporary path provided by pytest
    :param dataset: Name of the dataset to test
    :param params: Name of the directory where the params and expected results are stored for this test
    :param overwrite: Whether to overwrite the expected test data
    """
    params_dir = ParamSets.params_dir(dataset, params)
    data_dir = ParamSets.dataset_dir(dataset)
    results, _ = Pipeline.run_training(
        tmp_path=tmp_path,
        data_dir=data_dir,
        params_dir=params_dir,
        cache=request.config.cache,
        use_cached_results=False,
        retrain=False
    )
    ExpectedResults.assert_results_match_expected(
        results=results,
        expected_results_file = params_dir / "expected.yaml",
        results_type="Training",
        rel_error_tolerance=REL_ERROR_TOLERANCE,
        overwrite=overwrite
    )

@pytest.mark.filterwarnings("ignore:Converting sparse IndexedSlices")  # have pytest ignore this warning
@pytest.mark.parametrize("dataset,params", ParamSets.params_for_step("load_previous_model"))
def test_load_previous_model(request: pytest.FixtureRequest, tmp_path: Path, dataset: str, params: str, retrain: bool):
    params_dir = ParamSets.params_dir(dataset, params)
    data_dir = ParamSets.dataset_dir(dataset)
    _, model_dir = Pipeline.run_training(
        tmp_path=tmp_path,
        data_dir=data_dir,
        params_dir=params_dir,
        cache=request.config.cache,
        use_cached_results=False,
        retrain=retrain
    )
    Pipeline.load_model(
        tmp_path=model_dir,
        params_dir=params_dir,
        model_dir=model_dir
    )

@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Converting sparse IndexedSlices")  # have pytest ignore this warning
@pytest.mark.parametrize("dataset,train_params,predict_params", ParamSets.params_for_step("predict_server"))
async def test_predict_server(request: pytest.FixtureRequest, tmp_path: Path, dataset: str, train_params: str, predict_params: str, overwrite: bool):
    """
    Test the full pipeline for running the predict server

    The training results from running the model with be cached.

    :param request: Metadata provided by pytest
    :param tmp_path: Temporary path provided by pytest
    :param dataset: Name of the dataset to test
    :param train_params: Name of the directory where the params and expected results are stored for training the model
    :param predict_params: Name of the directory where the params and expected results are stored for this test
    :param overwrite: Whether to overwrite the expected test data
    """
    # train
    train_params_dir = ParamSets.params_dir(dataset, train_params)
    data_dir = ParamSets.dataset_dir(dataset)
    _, model_dir = Pipeline.run_training(
        tmp_path=tmp_path,
        data_dir=data_dir,
        params_dir=train_params_dir,
        cache=request.config.cache,
        use_cached_results=True,
        retrain=False
    )
    # predict
    predict_server_params_dir = ParamSets.params_dir(dataset, predict_params)
    record_file = predict_server_params_dir / "record_file.bin"
    predict_server_results = await Pipeline.run_predict_server(
        tmp_path=tmp_path,
        record_file=record_file,
        params_dir=predict_server_params_dir,
        model_dir=model_dir
    )
    ExpectedResults.assert_results_match_expected(
        results=predict_server_results,
        expected_results_file = predict_server_params_dir / "expected_predict_server.yaml",
        results_type="Predict Server",
        rel_error_tolerance=REL_ERROR_TOLERANCE,
        overwrite=overwrite
    )
