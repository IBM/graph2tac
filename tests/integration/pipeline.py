"""
Supporting code for testing the training and benchmarking pipeline
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force tests to run on CPU

from pathlib import Path
import pytest
import shutil
import sys
from unittest.mock import patch
import tensorflow as tf
from typing import Any, Union, List, Tuple
import warnings
import yaml

# this helps with determinism
tf.config.experimental.enable_op_determinism()

# automatically find parameters for tests to run
TESTDIR = Path(__file__).resolve().parent.parent
TESTDATADIR = TESTDIR / "data"

class ParamSets:
    @staticmethod
    def get_pipeline_step(param_dir: Path):
        if (param_dir / "model").exists():
            return "load_previous_model"
        if (param_dir / "hmodel.yml").exists():
            return "hmodel"
        elif (param_dir / "run_config.yml").exists():
            return "tfgnn"
        elif (param_dir / "predict_server.yml").exists():
            return "predict_server"
        else:
            raise Exception("Test parameter directory param_dir not properly configured.")
    
    @staticmethod
    def _param_train(param_dir: Path) -> Tuple[str, str]:
        return (param_dir.parent.parent.name, param_dir.name)
    
    @staticmethod
    def _param_predict_server(param_dir: Path) -> Tuple[str, str, str]:
        # find where to get the trained model from
        with (param_dir / "dependencies.yml").open("r") as f:
            dependencies = yaml.safe_load(f)
        data_param = param_dir.parent.parent.name
        train_param = dependencies["model"]
        predict_server_param = param_dir.name

        return (data_param, train_param, predict_server_param)
    
    @classmethod
    def params_for_step(cls, pipeline_step: str) -> List[tuple]:
        if pipeline_step == "predict_server":
            params = sorted([
                cls._param_predict_server(d)
                for d in TESTDATADIR.glob("*/params/*")
                if cls.get_pipeline_step(d) == pipeline_step
            ])
            # skip a particular test which has issues right now:
            params = [
                param if param != ("propchain_dep", "hmodel", "predict_server_hmodel") else
                pytest.param(
                    *param,
                    marks=pytest.mark.skip(reason="This hmodel test isn't properly testing the server anymore.")
                )
                for param in params
            ]
            return params
        elif pipeline_step in ["hmodel", "tfgnn", "load_previous_model"]:
            return sorted([
                cls._param_train(d)
                for d in TESTDATADIR.glob("*/params/*")
                if cls.get_pipeline_step(d) == pipeline_step
            ])
        else:
            raise ValueError(f"Unexpected value for `pipeline_step`: {pipeline_step}")

    @staticmethod
    def params_dir(dataset: str, params: str) -> Path:
        return TESTDATADIR / dataset / "params" / params
    
    @staticmethod
    def dataset_dir(dataset: str) -> Path:
        return TESTDATADIR / dataset / "dataset"


class ExpectedResults:
    @classmethod
    def overwritten_results(cls, old_results, old_approx_results, new_results, new_approx_results):
        """Overwrite expected results using old results unless they differ.

        This only basically matters for floating point numbers.
        """
        if old_approx_results == new_approx_results:
            return old_results
        
        if type(old_results) != type(new_results):
            return new_results
        
        if isinstance(new_results, dict):
            shared_keys = old_results.keys() & new_results.keys()
            new_keys = new_results.keys() - old_results.keys()
            shared_dict = {
                k: cls.overwritten_results(old_results[k], old_approx_results[k], new_results[k], new_approx_results[k])
                for k in shared_keys
            }
            new_dict = {k: new_results[k] for k in new_keys}
            shared_dict.update(new_dict)
            return shared_dict
            
        if isinstance(new_results, list):
            if len(new_results) != len(old_results):
                return new_results
            
            return [
                cls.overwritten_results(old, old_approx, new, new_approx)
                for (old, old_approx, new, new_approx) in zip(old_results, old_approx_results, new_results, new_approx_results)
            ]
    
        return new_results

    @staticmethod
    def overwrite_expected_results(results: dict, path: Path):
        warnings.warn(f"Overwriting expected results in {path}")
        with path.open("w") as f:
            yaml.safe_dump(results, f)

    @staticmethod
    def get_expected_results(path: Path):
        with path.open("r") as f:
            expected_results = yaml.safe_load(f)
        return expected_results

    @classmethod
    def approximate(cls, results: Union[dict, list, float, Any], rel_error_tolerance: float) -> Union[dict, list, float, Any]:
        """Recursively go through JSON compatible object and replace floats with pytest.approx"""
        if isinstance(results, dict):
            return {k : cls.approximate(v, rel_error_tolerance) for k,v in results.items()}
        elif isinstance(results, list):
            return [cls.approximate(v, rel_error_tolerance)  for v in results]
        elif isinstance(results, float):
            return pytest.approx(results, rel=rel_error_tolerance)
        else: # bool, str, int, None, etc
            return results 
    
    @classmethod
    def assert_results_match_expected(cls, results: dict, expected_results_file: Path, results_type: str, rel_error_tolerance: float, overwrite: bool):
        approx_results = cls.approximate(results, rel_error_tolerance=rel_error_tolerance)
        if overwrite:
            if not expected_results_file.exists():
                cls.overwrite_expected_results(results, expected_results_file)
            else:
                expected_results = cls.get_expected_results(expected_results_file)
                approx_expected_results = cls.approximate(expected_results, rel_error_tolerance=rel_error_tolerance)
                if approx_results != approx_expected_results:
                    results = cls.overwritten_results(
                        old_results=expected_results,
                        old_approx_results=approx_expected_results,
                        new_results=results,
                        new_approx_results=approx_results
                    )
                    cls.overwrite_expected_results(results, expected_results_file)

        assert expected_results_file.exists(), f"No {results_type} expected results file. To create one, rerun this test with `--overwrite`."
        expected_results = cls.get_expected_results(expected_results_file)
        approx_expected_results = cls.approximate(expected_results, rel_error_tolerance=rel_error_tolerance)
        assert approx_results == approx_expected_results, (
            f"{results_type} statistics don't match expected results. To overwrite expected results, rerun this test with `--overwrite`."
        )

class Pipeline:
    @staticmethod
    def _run_tfgnn_training(tmp_path: Path, data_dir: Path, params_dir: Path) -> Tuple[dict, Path]:
        """Run training and return results for comparison"""
        import graph2tac.tfgnn.train  # put import here so it doesn't break other tests if it crashes

        model_dir = (tmp_path / "log").resolve()
        training_args = ["<program>",
            "--data-dir", data_dir,
            "--dataset-config", params_dir / "dataset_config.yml",
            "--prediction-task-config", params_dir / "global_argument_prediction.yml",
            "--definition-task-config", params_dir / "definition_task.yml",
            "--trainer-config", params_dir / "trainer_config.yml",
            "--run-config", params_dir / "run_config.yml",
            "--log", model_dir,
        ]
        # use context manager to pass command line arguments to our main method
        with patch.object(sys, 'argv', [str(a) for a in training_args]):
            history = graph2tac.tfgnn.train.main_with_return_value()
        
        # remove results which are not stable or not serializable
        results = {k:v for k,v in history.history.items() if k not in ["epoch_duration", "learning_rate"]}
        return results, model_dir

    @staticmethod
    def _run_hmodel_training(tmp_path: Path, data_dir: Path, params_dir: Path) -> Tuple[dict, Path]:
        """Run training and return results for comparison"""
        import graph2tac.loader.hmodel  # put import here so it doesn't break other tests if it crashes

        model_dir = (tmp_path / "log").resolve()
        training_args = ["<program>",
            data_dir,
            "--output_dir", model_dir
        ]

        # read additional arguments from the hmodel.yml file
        with (params_dir / "hmodel.yml").open() as f:
            hmodel_params = yaml.safe_load(f)
        for key, value in hmodel_params.items():
            training_args.append(f"--{key}")
            if value is not None:
                training_args.append(value)

        # use context manager to pass command line arguments to our main method
        with patch.object(sys, 'argv', [str(a) for a in training_args]):
            model_results = graph2tac.loader.hmodel.main_with_return_value()
        
        # sample first 10 hashs (lexicographically) and format actions in JSON compatible format
        hashes = sorted(model_results["data"].keys())[:10]
        data = {}
        for hsh in hashes:
            data[hsh] = []
            action = model_results["data"][hsh]
            for tactic, args in action:
                data[hsh].append({"tactic": tactic, "args": [{"arg_type": int(kind), "arg_index": int(index)} for kind, index in args]})

        return data, model_dir

    @classmethod
    def _run_training(cls, tmp_path: Path, data_dir: Path, params_dir: Path) -> Tuple[dict, Path]:
        training_type = ParamSets.get_pipeline_step(params_dir)
        if training_type == "hmodel":
            return cls._run_hmodel_training(tmp_path=tmp_path, data_dir=data_dir, params_dir=params_dir)
        elif training_type == "tfgnn":
            return cls._run_tfgnn_training(tmp_path=tmp_path, data_dir=data_dir, params_dir=params_dir)
        else:
            raise ValueError(f"Unexpected value for training_type: {training_type}")

    @classmethod
    def run_training(cls, tmp_path: Path, data_dir: Path, params_dir: Path, cache: pytest.Cache, use_cached_results: bool = False, retrain: bool = False) -> Tuple[dict, Path]:
        # check for saved pretrained models
        if (params_dir / "model").exists():
            pretrained_model_dir = params_dir / "model"
            if not retrain:
                print(f"Using pretrained saved model: {pretrained_model_dir}.  Use --retrain to retrain this model.")
                return {}, pretrained_model_dir  # don't return results for pretrained model
            else:
                warnings.warn(f"Deleting and retraining saved model: {pretrained_model_dir}")
                # delete model
                shutil.rmtree(pretrained_model_dir)  
                _, tmp_model_dir = cls._run_training(tmp_path=tmp_path, data_dir=data_dir, params_dir=params_dir)
                # replace model with the one just trained
                shutil.copytree(tmp_model_dir, pretrained_model_dir)
                # remove unneeded binary files
                for tensorboard_file in pretrained_model_dir.glob("**/events.out.tfevents*"):
                    tensorboard_file.unlink()
                for zeroth_checkpoint_file in pretrained_model_dir.glob("ckpt/ckpt-0*"):
                    zeroth_checkpoint_file.unlink()
                return {}, pretrained_model_dir  # don't return results for pretrained model
        
        # check for cached models and results
        key = "graph2tac/" + str(params_dir)
        cached_results = cache.get(key, None)
        if use_cached_results and (cached_results is not None) and Path(cached_results[1]).exists():
            results, cached_model_dir = cached_results
            cached_model_dir = Path(cached_model_dir)
            print(f"Using cached model: {cached_model_dir}")
            return results, cached_model_dir
        
        # run model and cache results
        results, model_dir = cls._run_training(tmp_path, data_dir, params_dir)
        cache.set(key, [None, str(model_dir)])
        return results, model_dir

    @staticmethod
    def load_model(tmp_path: Path, params_dir: Path, model_dir: Path) -> bool:
        """Load model (and do nothing else)"""
        # figure out what kind of model it is
        if (params_dir / "hmodel.yml").exists():
            from graph2tac.loader.hmodel import HPredict
            HPredict(
                checkpoint_dir=model_dir,
                debug_dir=None
            )
            return True
        else:
            from graph2tac.tfgnn.predict import TFGNNPredict  # put import here so it doesn't break other tests if it crashes
            from tensorflow.python.eager import context
            # clear any tensorflow settings from tensorflow imports earlier in the pipeline
            # without this we can run into issues of setting tf.config twice
            context._context = None
            context._create_context()
            # a seed has to be set after tensorflow is imported.
            tf.keras.utils.set_random_seed(1)

            TFGNNPredict(
                log_dir=model_dir,
                debug_dir=None,
                checkpoint_number=None,
                exclude_tactics=None,
                tactic_expand_bound=20,
                search_expand_bound=20
            )
            return True


    @staticmethod
    async def run_predict_server(tmp_path: Path, record_file: Path, params_dir: Path, model_dir: Path) -> dict:
        """Run training and return results for comparison"""
        import graph2tac.loader.predict_server  # put import here so it doesn't break other tests if it crashes
        from tensorflow.python.eager import context
        # clear any tensorflow settings from tensorflow imports earlier in the pipeline
        # without this we can run into issues of setting tf.config twice
        context._context = None
        context._create_context()
        # a seed has to be set after tensorflow is imported.
        # Also it makes the tests deterministic.
        # (The predict server is usually deterministic except when using --update-no-definitions)
        tf.keras.utils.set_random_seed(1)

        server_args = ["<program>",
            "--model", model_dir,
            "--replay", record_file
        ]
        # read additional arguments from the predict_server.yml file
        with (params_dir / "predict_server.yml").open() as f:
            predict_server_params = yaml.safe_load(f)
        for key, value in predict_server_params.items():
            server_args.append(f"--{key}")
            if value is not None:
                server_args.append(value)

        # use context manager to pass command line arguments to our main method
        with patch.object(sys, 'argv', [str(a) for a in server_args]):
            history = await graph2tac.loader.predict_server.main_with_return_value()

        # clean up the results to be standardized and easy for the testing system
        responses =  history.data["responses"]
        for response in responses:
            if response["_type"] == "TacticPredictionsGraph":
                # sort the predictions in the response into a standard order:
                # first sort by confidence (higher first)
                # then sort by the base tactic ident
                # then sort by arguments to the tactic
                response["contents"]["predictions"] = sorted(
                    response["contents"]["predictions"],
                    key=lambda p: (-p["confidence"], p["ident"], p["arguments"])
                )

        return history.data["responses"]


