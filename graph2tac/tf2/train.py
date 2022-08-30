"""
The main training loop for TF code.
"""
import argparse
from datetime import datetime, timedelta
import numpy as np
import random
import os
import pickle
import psutil
import logging
from pathlib import Path
import socket
import subprocess
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import uuid
import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import yaml

# Turn off TF log messages besides errors before loading tensorflow (may get turned back on below)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import tensor2tensor

from graph2tac.loader.data_server import DataServer

from graph2tac.tf2.model_params import ModelDatasetConstants, ModelParams
from graph2tac.tf2.model import ModelWrapper,\
       StateActionTensor, GraphDefTensor, np_to_tensor, np_to_tensor_def
from graph2tac.tf2.stop_watch import StopWatch, get_times, print_times
from graph2tac.tf2.datatypes import RaggedPair

from graph2tac.tf2.graph_nn_batch import make_flat_batch_np, make_flat_batch_np_empty
from graph2tac.tf2.graph_nn_def_batch import make_flat_def_batch_np, make_flat_def_batch_np_empty
from graph2tac.tf2.batches import Batches, Repeat
from graph2tac.tf2.predict import Predict
from graph2tac.tf2.segments import segment_all

@dataclass_json
@dataclass
class DataParams:
    split_seed: int = 0  # The seed to use for splitting train and valid.  Doesn't effect test set.
    split: tuple[int, int, int] = (8, 1, 1)  # Train, validate, test split ratios
    cv_fold: int = 0  # The cross validation fold to use
    shuffle_def: bool = True
    max_subgraph_size: int = 1024
    bfs_option: bool = True
    restrict_to_spine: bool = False


@dataclass_json
@dataclass
class LossWeightParams:
    tactic_base: float = 1.0  # how much to weight the base tactic in the loss
    tactic_args: float = 1.0  # how much to weight the tactic arguments in the loss
    def_task: float = 0.1     # how much to weight the definition task in the loss
    # these next two values say how much to push either definition embedding to the other
    # in the definition task.  The two embs are the emb which comes from the definition id
    # and the embedding which is computed from the (nodes in) the body of the definition
    def_id_to_body: float = 1.0  # how much to push the emb from the id to the emb from the def body
    def_body_to_id: float = 1.0  # how much to push the emb from the def body to the emb from the id


@dataclass_json
@dataclass
class OptimizerParams:
    optimizer: str = "adam"
    learning_rate: Optional[float] = None  # learning rate for optimizer (None = optimizer default)
    clipvalue: Optional[float] = None  # gradient clipping bound (None = off)
    clipnorm: Optional[float] = None  # norm clipping bound (None = off)
    global_clipnorm: Optional[float] = None  # global_norm clipping bound (None = off)
    #label_smoothing: bool = False
    loss_weights: LossWeightParams = LossWeightParams()
    def_task: bool = True  # use the auxillary definition task

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict())


class StatisticsJSONEncoder(json.JSONEncoder):
    """
    JSON Encoder which also handles a few other common types like Paths.
    """
    # overload method default
    def default(self, obj):

        # Handle special types here
        if isinstance(obj, Path):
            return str(obj)

        # Call the default method for other types
        return json.JSONEncoder.default(self, obj)


class StatisticsRecorder:
    """
    A wrapper around whatever mechanism we want to use to log/record statistics for later plotting.

    By keeping everything in this wrapped object, it is easy to change how we record
    statistics in the future.  (As of the creation of this note, we just print to the STDERR,
    but soon we will either save to a file, or log to Weights & Biases or a similar service.)
    """
    def __init__(self, file: Optional[Path]):
        self._file = file
        if self._file is not None:
            self._file.write_text("")  # make new empty file

    def _log_record(self, record: dict):
        if self._file is None:
            return

        with self._file.open(mode="a") as f:
            f.write(json.dumps(record, cls=StatisticsJSONEncoder) + "\n")

    def record_parameter(self, record: dict):
        """
        Record one time statistics, such as model parameters.

        record: a JSON-serializable dictionary, e.g. {"activation": "relu", "hops": 10}
        """
        record = record.copy()
        record["_type"] = "parameter"

        # log datapoint
        self._log_record(record)


    def record_results(self, record: dict):
        """
        Record the metrics corresponding to a time event, like an epoch

        Some project metadata and machine statistics are added to the record.

        record: a JSON-serializable dictionary, e.g. {"loss": 1.2, "accuracy": .5}
        """
        record = record.copy()
        record["_type"] = "results"

        # add memory statistics to the record
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        record["mem_physical_gb"] = mem.rss / 10**9
        record["mem_total_gb"] = mem.rss / 10**9

        # log datapoint
        self._log_record(record)


# def segment_loss_cross_entropy(labels, predictions):
#     predictions, ids = predictions
#     logprobs = segment_logsoftmax(predictions, ids, len(labels))
#     return -tf.math.reduce_sum(tf.gather(logprobs, labels))
# def segment_get_predictions(logits_ids, bs):
#     logits, ids = logits_ids
#     return segment_argmax(logits, ids, bs)


class Checkpointer:
    """
    Saves checkpoints for both the model and the optimizer.
    """
    def __init__(
        self,
        checkpoint_dir: Optional[Path],
        model_name: str,
        model_wrapper: ModelWrapper,
        optimizer,
        optimizer_params: OptimizerParams,
    ):
        self.model_wrapper = model_wrapper
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(exist_ok = True)

    def make_checkpoint(self, epoch: int):
        if self.checkpoint_dir is None:
            return

        # store all files for a given checkpoint in its own subdirectory
        subdir = self.checkpoint_dir / f"{self.model_name}__epoch{epoch}"
        subdir.mkdir(exist_ok=True)

        self.model_wrapper.save_model_checkpoint(subdir)

        # save optimizer weights  # TODO(jrute): This hasn't worked right.
        # The weights can't be loaded, and it assumes adam optimizer.
        symbolic_weights = getattr(self.optimizer, 'weights')
        weight_values = keras.backend.batch_get_value(symbolic_weights)
        with (subdir / "optim_weights").open("wb") as f:
            pickle.dump(weight_values, f)

        param_yaml_path = subdir / "optim_params.yaml"
        param_yaml_path.write_text(self.optimizer_params.to_yaml())

    def test_load_from_checkpoint(self, epoch: int):
        # TODO(jrute): Remove when done testing this
        subdir = self.checkpoint_dir / f"{self.model_name}__epoch{epoch}"
        new_model_wrapper = ModelWrapper(checkpoint=subdir)
    def load_from_checkpoint(self, epoch: int):
        subdir = self.checkpoint_dir / f"{self.model_name}__epoch{epoch}"
        self.model_wrapper = ModelWrapper(checkpoint=subdir)

    def test_predict(self, epoch: int, batch):
        # TODO(jrute): Remove when done testing this
        subdir = self.checkpoint_dir / f"{self.model_name}__epoch{epoch}"
        predictor = Predict(checkpoint_dir=subdir)
        state = batch[0][0]
        cxt = state[3]
        tactic = batch[0][1][0]
        arg_mask = batch[0][1][2]
        arg_cnt = arg_mask.sum()
        print("tac_result", predictor.predict_tactic_logits(state).shape)
        r = predictor.predict_arg_logits(state, tactic)
        print("arg_result", r.shape)
        print("arg_mask", arg_mask)
        print("expected_arg_size:", (arg_cnt, len(cxt)))
        assert r.shape == (arg_cnt, len(cxt))


class FingerPrinter():
    """
    Used to make test fingerprints for integration tests
    """
    def __init__(self, fingerprint_fname: Optional[Path]):
        self.data = []
        self.fname = fingerprint_fname

    @staticmethod
    def round4(x: float) -> str:
        return "{:.4E}".format(x)

    def append(self, point: List[float]):
        assert type(point) == list
        self.data.append(self.round4(x) for x in point)

    def record(self):
        if self.fname is not None:
            with open(self.fname, 'w') as finger_file:
                print(' '.join([element for f_point in self.data  for element in f_point]),
                    file=finger_file)

class Metrics:
    """
    Stores TensorFlow metrics into a dictionary for easy access and reporting

    All metrics should have reset_state, and result attributes.
    The result attribute is assumed to return a float.
    """
    def __init__(self, metrics: Dict[str, Any]):  # where any is one of the metrics in tf.keras.metrics
        self.metric_dict = metrics  # a dictionary of metrics like tf.keras.metrics.Mean

    def reset_states(self):
        for m in self.metric_dict.values():
            m.reset_states()

    def __getitem__(self, metric_id):
        return self.metric_dict[metric_id]

    def all_results(self) -> Dict[str, float]:
        return {k: float(metric.result()) for k, metric in self.metric_dict.items()}

def print_for_user(message: str):
    print(f"TRAIN | {message}", flush=True)

def git_data():
    """
    Return a dictionary of relevant git data.

    Unfortunately, this must be run from within the git repo.
    """
    def run_cmd(cmd) -> str:
        """Run a command and catch any errors"""
        try:
            return subprocess.check_output(cmd).decode('utf-8').strip()
        except subprocess.CalledProcessError as e:
            return e.stdout.decode('utf-8').strip()

    return {
        # to make sure we are in the correct repo
        "git_origin_url": run_cmd(["git", "remote", "get-url", "origin"]),

        # this is *usually* the git branch, but it can also be a git tag
        # it is *a branch* you are on but not necessarily *the branch* you have checked out
        # if two branch heads are at the same location
        # it will end in "-dirty" if there are uncommited changes on the branch
        # it is useful for a quick human-readable id, but the commit hash is more precise
        "git_branch": run_cmd(["git", "describe", "--all", "--dirty"]),

        # this is <tag>-<num of commits after tag>-<hash> and -dirty if uncommitted code
        # without any tags in the origin, it is just the commit hash
        "git_describe": run_cmd(["git", "describe", "--always", "--dirty"]),

        # git commits
        "git_commit_short": run_cmd(["git", "rev-parse", "--short", "HEAD"]),
        "git_commit_long": run_cmd(["git", "rev-parse", "HEAD"]),
    }

def uuid_data():
    """
    Return a dictionary with a uuid and related data

    Note, this uses the random number generator.
    """
    this_uuid = uuid.uuid1()  # this is determined by the time, the machine ip, and a random index

    # make a time stamp from the time used in the uuid
    ms = this_uuid.time // 10  # milliseconds from 00:00:00.00, 15 October 1582
    dt = datetime(1582, 10, 15) + timedelta(microseconds=ms)
    timestamp = str(dt)  # timestamp which is both human readable and readable by pandas

    # get hostname
    hostname = socket.gethostname()

    return {
        # to make sure we are in the correct repo
        "uuid": str(this_uuid),
        "timestamp": timestamp,
        "hostname": hostname,
    }

@dataclass_json
@dataclass
class TrainingParams:
    data_params: DataParams = DataParams()
    optimizer_params: OptimizerParams = OptimizerParams()
    model_params: ModelParams = ModelParams()

    tf_seed: int = 42  # seed for TF (also used for python random module just in case)
    tf_eager: bool = False  # run TF in eager mode (compiles faster but runs slower, for testing)
    enable_op_determinism: bool = False
    batch_size: int = 50
    num_epochs: int = 10  # number of passes through the full training dataset
    num_proc: int = None


    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml_file(yaml_file: Path) -> "TrainingParams":
        with yaml_file.open() as f:
            param_dict=yaml.load(f, Loader=yaml.FullLoader)
            if param_dict is None:  # empty file
                param_dict = {}
            return TrainingParams.from_dict(param_dict)


class TrainingLoop:
    # training loop settings
    batch_size: int
    num_epochs: int
    # objects for passing and recording data
    data_server: DataServer
    metrics: Metrics
    checkpointer: Checkpointer
    recorder: StatisticsRecorder
    fingerprinter: FingerPrinter
    from_checkpoint: Optional[int]

    # TODO(jrute): Remove dependence on test parameter since not being used anymore to restrict to a single file
    def __init__(
        self,
        data_dir: Path,
        work_dir: Path,
        params: TrainingParams,
        output_dir: Optional[Path],
        fingerprint: Optional[Path],
        logging_level: str,
        use_cache: bool,
        max_steps: float,  # for debugging, max steps before stopping training (default is inf)
        from_checkpoint: Optional[int],
    ):
        print_for_user("Setting up training...")

        # set up directories for output
        if output_dir is None:
            print_for_user("No output directory.  Model weights and results will not be saved.")
            records_file = None
            checkpoint_dir = None
        else:
            output_dir.mkdir(exist_ok=True)
            records_file = output_dir / "records.jsonl"
            checkpoint_dir = output_dir / "weights"
            checkpoint_dir.mkdir(exist_ok=True)
            print_for_user(f"Results will be saved in: {records_file.resolve()}")
            print_for_user(f"Checkpoints will be saved in: {checkpoint_dir.resolve()}")
        print_for_user(params.to_yaml())
        self.recorder = StatisticsRecorder(records_file)
        self.fingerprinter = FingerPrinter(None if fingerprint is None else Path(fingerprint))

        # store a unique id for the run data
        # this uses the random number generator, but it is ok since we set the seed after this
        self.recorder.record_parameter(uuid_data())

        # record function inputs
        self.recorder.record_parameter({
            "data_dir": str(data_dir),
            "params_dict": params.to_dict(),
            "params_yaml": params.to_yaml(),
            "output_dir": str(output_dir),
            "fingerprint": fingerprint,
            "logging_level": logging_level
        })

        # record git data
        # this only gives meaningful results if run from inside the git repo
        self.recorder.record_parameter(git_data())

        # Handle loggers and warnings
        LOGGING_LEVELS = {"DEBUG": '0', "INFO": '1', "WARNING": '2', "ERROR": '3'}
        assert logging_level in LOGGING_LEVELS, \
            f"Logging level {logging_level} not one of {list(LOGGING_LEVELS.keys())}."
        # TF sometimes uses the warnings module.
        # Here we redirect them to the "py.warnings" logger.
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging_level)
        # We use environment variable here so that tf warnings are turned on/off before tensorflow used in the next line.
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = LOGGING_LEVELS[logging_level]
        tf.get_logger().setLevel(logging_level)
        logging.getLogger().setLevel(logging_level)
        # This next line is needed for TF warnings which come through the warning module to work for some reason.
        logging.warning("Warnings are enabled.")  # Only visible if indeed warnings are enabled.

        # Tensorflow settings
        tf.config.run_functions_eagerly(params.tf_eager)  # turns off @tf.function compilation.  Used for testing.
        # tf.config.threading.set_inter_op_parallelism_threads(4)
        # tf.config.threading.set_intra_op_parallelism_threads(1)
        random.seed(params.tf_seed)
        tf.random.set_seed(params.tf_seed)
        if params.enable_op_determinism:
            tf.config.experimental.enable_op_determinism()

        # training loop parameters
        self.batch_size = params.batch_size
        self.num_epochs = params.num_epochs
        self.max_steps = max_steps

        # load dataset
        print_for_user(f"Data source dir: {data_dir.resolve()}")
        print_for_user(f"Loading data...")

        self.data_server = DataServer(
            data_dir,
            max_subgraph_size=params.data_params.max_subgraph_size,
            split=tuple(params.data_params.split),
            bfs_option=params.data_params.bfs_option,
            split_random_seed=params.data_params.split_seed,
            restrict_to_spine=params.data_params.restrict_to_spine
        )
        print_for_user(f"Loading data complete...")

        graph_constants = self.data_server.graph_constants()

        print_for_user(f"Loading graph constants complete")

        # extract constants needed for building the model
        model_params = params.model_params
        assert model_params.dataset_consts is None, (
            f"Expected param dataset_consts == None for training.  "
            f"Found: {model_params.dataset_consts}"
        )
        model_params.dataset_consts = ModelDatasetConstants(
            tactic_num=graph_constants.tactic_num,
            tactic_max_arg_num=max(graph_constants.tactic_index_to_numargs).item(),
            edge_label_num=graph_constants.edge_label_num,
            base_node_label_num=graph_constants.base_node_label_num,
            node_label_num=graph_constants.node_label_num,
            # convert tactic_index_to_numargs to list[int] so can serialize this hyperparam to yaml

            tactic_index_to_numargs=graph_constants.tactic_index_to_numargs.tolist(),
            tactic_index_to_hash=graph_constants.tactic_index_to_hash.tolist(),
            node_label_to_name=graph_constants.label_to_names,
            node_label_in_spine=graph_constants.label_in_spine,
            global_context=graph_constants.global_context.tolist(),
            max_subgraph_size=graph_constants.max_subgraph_size
        )
        # TODO(jrute): Move everything which depends on this (namely flat_batch_np)
        # into ModelWrapper and then we don't need to expose it.
        self.dataset_consts = model_params.dataset_consts

        # build model
        model_wrapper = ModelWrapper(model_params)

        # build metrics for recording
        primary_task_metrics = {
            "tac_loss": tf.keras.metrics.Mean,
            "arg_loss": tf.keras.metrics.Mean,
            "act_log_prob": tf.keras.metrics.Mean,  # the log prob of the whole tactic
            "tac_acc": tf.keras.metrics.Accuracy,
            "arg_acc": tf.keras.metrics.Accuracy,
            "acc": tf.keras.metrics.Mean,
            # "strict_acc" is accuracy where Nones are considered inaccurate even if in ground truth
            "strict_acc": tf.keras.metrics.Mean,
            # "no_nones" is the ratios of ground truth tactics without nones
            "no_nones": tf.keras.metrics.Mean,
            # accuracy on arguments with the gold label local / none / global
            "arg_local_acc": tf.keras.metrics.Mean,
            "arg_none_acc": tf.keras.metrics.Mean,
            "arg_global_acc": tf.keras.metrics.Mean,
        }
        other_training_metrics = {
            "act_loss": tf.keras.metrics.Mean,  # weighted loss for both parts of the tactic
            "def_loss": tf.keras.metrics.Mean,
            "loss": tf.keras.metrics.Mean,
            "def_body_norm": tf.keras.metrics.Mean,
            "def_id_norm": tf.keras.metrics.Mean,
            "def_diff_norm": tf.keras.metrics.Mean,
        }
        metrics_dict = {}
        metrics_dict.update({
            prefix + "_" + name : constructor(name=prefix + "_" + name)
            for name, constructor in primary_task_metrics.items()
            for prefix  in ["train", "valid"]
        })
        metrics_dict.update({
            "train_" + name : constructor(name="train_" + name)
            for name, constructor in other_training_metrics.items()
        })
        self.metrics = Metrics(metrics_dict)

        def record_metrics_in_step(
            prefix: str,
            batch: StateActionTensor,
            model_outputs  # ([bs], [bs, (args)] (ragged)), ([bs], [bs])
        ):
            """
            Record the primary task metrics during the train_step or valid_step
            prefix: "train" or "valid"
            """
            (tactic_pred, arg_pred), (tactic_loss, arg_loss) = \
                model_outputs  # ([bs], [bs, (args)] (ragged)), ([bs], [bs])

            mask_args = batch.actions.mask_args
            arg_ids = tf.boolean_mask(
                tf.tile(
                    tf.expand_dims(tf.range(tf.shape(mask_args)[0]), 1),
                    (1, tf.shape(mask_args)[1])
                ),
                mask_args,
            )
            arg_labels = RaggedPair(arg_ids, batch.actions.arg_labels)

            self.metrics[prefix + "_tac_loss"](tactic_loss)
            self.metrics[prefix + "_arg_loss"](arg_loss)
            self.metrics[prefix + "_act_log_prob"](tactic_loss + arg_loss)
            self.metrics[prefix + "_tac_acc"](batch.actions.tactic_labels, tactic_pred)
            self.metrics[prefix + "_arg_acc"](arg_labels.values, arg_pred.values)  # mean per argument accuracy

            bs = tf.shape(batch.states.roots)[0]
            tactic_accurate = tf.math.equal(batch.actions.tactic_labels, tactic_pred) # [bs]  dtype=bool
            args_accurate = tf.math.equal(arg_labels.values, arg_pred.values)
            all_args_accurate = segment_all(
                args_accurate,
                arg_labels.indices,
                bs,
            ) # [bs]  dtype=bool
            all_accurate = tactic_accurate & all_args_accurate
            self.metrics[prefix + "_acc"](all_accurate)

            # "strict_acc" is the accuracy where anything with a None is counted as incorrect.
            # The "none" index is equal to the number of elements in the context.
            # This way of getting the index of the none value (# of elmts in cxt) is really hacky.
            # Hopefully we move away from indexing tricks like this
            context_cnt = tf.math.unsorted_segment_sum(
                tf.ones_like(batch.states.context.indices),
                batch.states.context.indices,
                bs,
            )
            context_cnt = tf.reduce_sum(tf.gather(context_cnt, arg_labels.indices))
            label_cnt = tf.shape(arg_labels.indices)[0]
            none_ix=tf.range(context_cnt, context_cnt + label_cnt, dtype=tf.int32)
            is_local = arg_labels.values < context_cnt
            is_none = (arg_labels.values == none_ix)
            is_global = arg_labels.values >= context_cnt+label_cnt
            has_no_none = segment_all(  # [bs]  dtype=bool
                ~is_none,
                arg_labels.indices,
                bs,
            )
            self.metrics[prefix + "_strict_acc"](all_accurate & has_no_none)
            self.metrics[prefix + "_no_nones"](has_no_none)
            self.metrics[prefix + "_arg_local_acc"](tf.boolean_mask(args_accurate, is_local))
            self.metrics[prefix + "_arg_none_acc"](tf.boolean_mask(args_accurate, is_none))
            self.metrics[prefix + "_arg_global_acc"](tf.boolean_mask(args_accurate, is_global))

        # build optimizer
        # TODO(jrute): add back in label smoothing
        def get_optimizer_cls(optimizer_type: str):
            if optimizer_type == "adadelta":
                return tf.keras.optimizers.Adadelta
            elif optimizer_type == "adafactor":
                return tensor2tensor.utils.adafactor.AdafactorOptimizer
            elif optimizer_type == "adagrad":
                return tf.keras.optimizers.Adagrad
            elif optimizer_type == "adam":
                return tf.keras.optimizers.Adam
            elif optimizer_type == "adamax":
                return tf.keras.optimizers.Adamax
            elif optimizer_type == "ftrl":
                return tf.keras.optimizers.Ftrl
            elif optimizer_type == "nadam":
                return tf.keras.optimizers.Nadam
            elif optimizer_type == "rmsprop":
                return tf.keras.optimizers.RMSprop
            elif optimizer_type == "sgd":
                return tf.keras.optimizers.SGD
            else:
                raise Exception(f"Unknown optimizer: {optimizer_type}")
        optimizer_cls = get_optimizer_cls(
            params.optimizer_params.optimizer
        )
        opt_kwargs = {}
        if params.optimizer_params.learning_rate is not None:
            opt_kwargs["learning_rate"] = params.optimizer_params.learning_rate

        if params.optimizer_params.clipvalue is not None:
            opt_kwargs["clipvalue"] = params.optimizer_params.clipvalue

        if params.optimizer_params.clipnorm is not None:
            opt_kwargs["clipnorm"] = params.optimizer_params.clipnorm

        if params.optimizer_params.global_clipnorm is not None:
            opt_kwargs["global_clipnorm"] = params.optimizer_params.global_clipnorm

        optimizer = optimizer_cls(**opt_kwargs)

        # build checkpointer
        self.checkpointer = Checkpointer(
            # TODO(jrute): Use better location
            checkpoint_dir=checkpoint_dir,
            model_name="checkpoint",
            model_wrapper=model_wrapper,
            optimizer=optimizer,
            optimizer_params=params.optimizer_params,
        )
        self.from_checkpoint = from_checkpoint
        if from_checkpoint is not None:
            self.checkpointer.load_from_checkpoint(from_checkpoint)
            model_wrapper=self.checkpointer.model_wrapper

        loss_weights = params.optimizer_params.loss_weights
        use_def_task = params.optimizer_params.def_task

        def squared_norm_diff(
            emb1,  # [batch, dim]
            emb2,  # [batch, dim]
        ):  # -> [batch]
            diff = emb1 - emb2  # [batch, dim]
            norm_squared = tf.math.reduce_sum(tf.square(diff), axis=-1)  # [batch]
            return norm_squared  # [batch]

        combined_model = model_wrapper.combined_model

        @tf.function(input_signature=(model_wrapper.input_spec, model_wrapper.def_input_spec))
        def train_step(batch_sa: StateActionTensor, batch_gd: GraphDefTensor):
            with tf.GradientTape() as tape:
                tactic_logits, arg_cnt, arg_logits = combined_model.model(batch_sa, training=True)
                model_outputs = model_wrapper.get_predictions_and_losses(
                    tactic_logits=tactic_logits,
                    tactic_labels=batch_sa.actions.tactic_labels,
                    arg_cnt=arg_cnt,
                    arg_logits=arg_logits,
                    arg_labels=batch_sa.actions.arg_labels,
                )
                _, (tactic_loss, arg_loss) = model_outputs  # _, ([bs], [bs])
                action_loss = (
                    loss_weights.tactic_base * tactic_loss + loss_weights.tactic_args * arg_loss
                )
                # average the loss over the batch
                # note, we use tf.nn.compute_average_loss to prepare for
                # distributed training on multiple GPUs
                # We just use the current batch size (max with 1 to handle empty batch)
                action_loss = tf.nn.compute_average_loss(  # scalar
                    action_loss,
                    global_batch_size=tf.maximum(tf.shape(action_loss)[0], 1)
                )

                if use_def_task:
                    def_body_embs, def_id_embs = model_wrapper.model_def(batch_gd, training=True)
                    # note, we use stop gradients to give more control over whether to push
                    # the def_body_embs to def_id_embs or the other direction.
                    # If loss_weights.def_val = loss_weights.def_body = 1.0
                    # then the gradient is equal to the usual mean squared error
                    # (even though the loss is now twice as large.)
                    def_id_vs_body_loss = squared_norm_diff(  # [roots in batch]
                        tf.stop_gradient(def_body_embs.values),
                        def_id_embs.values,
                    )
                    def_body_vs_id_loss = squared_norm_diff(  # [roots in batch]
                        def_body_embs.values,
                        tf.stop_gradient(def_id_embs.values),
                    )
                    defs_loss = (  # [roots in batch]
                        loss_weights.def_body_to_id * def_body_vs_id_loss
                        + loss_weights.def_id_to_body * def_id_vs_body_loss
                    )
                    # average the loss over the batch
                    # note, we use tf.nn.compute_average_loss to prepare for
                    # distributed training on multiple GPUs
                    # also note, the defs_loss array has more elements than the batch
                    # size since some graphs have multiple roots
                    # We just use the current batch size (max with 1 to handle empty batch)
                    defs_loss = tf.nn.compute_average_loss(  # scalar
                        defs_loss,
                        global_batch_size=tf.maximum(tf.shape(defs_loss)[0], 1)
                    )
                else:
                    defs_loss = tf.constant(0.0)

                weighted_defs_loss = loss_weights.def_task * defs_loss  # scalar

                loss = action_loss + weighted_defs_loss

            trainable_variables = combined_model.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            record_metrics_in_step("train", batch=batch_sa, model_outputs=model_outputs)

            # training specific metrics
            self.metrics["train_act_loss"](action_loss)
            self.metrics["train_loss"](action_loss + weighted_defs_loss)
            if use_def_task:
                self.metrics["train_def_loss"](defs_loss)
                self.metrics["train_def_body_norm"](tf.norm(def_body_embs.values, axis=-1))
                self.metrics["train_def_id_norm"](tf.norm(def_id_embs.values, axis=-1))
                self.metrics["train_def_diff_norm"](tf.norm(def_body_embs.values - def_id_embs.values, axis=-1))

        self.train_step = train_step  # TODO(jrute): Make this a method directly instead of this hack

        self._shuffle_def = params.data_params.shuffle_def
        @tf.function(input_signature=(model_wrapper.input_spec,))
        def valid_step(batch: StateActionTensor):
            tactic_logits, arg_cnt, arg_logits = model_wrapper.model(batch)
            model_outputs = model_wrapper.get_predictions_and_losses(
                tactic_logits=tactic_logits,
                tactic_labels=batch.actions.tactic_labels,
                arg_cnt=arg_cnt,
                arg_logits=arg_logits,
                arg_labels=batch.actions.arg_labels,
            )

            record_metrics_in_step("valid", batch=batch, model_outputs=model_outputs)

        self.valid_step = valid_step  # TODO(jrute): Make this a method directly instead of this hack

        # log model summary
        def get_model_summary_string(model):
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            short_model_summary = "\n".join(stringlist)
            return short_model_summary
        self.recorder.record_parameter({"model_summary": get_model_summary_string(model_wrapper.model)})
        self.recorder.record_parameter({"model_def_summary": get_model_summary_string(model_wrapper.model_def)})
        print(get_model_summary_string(model_wrapper.model))
        print(get_model_summary_string(model_wrapper.model_def))

        print_for_user(f"Compiling TF model...")

        with StopWatch("Compilation"):
            flat_batch_np = make_flat_batch_np_empty(self.dataset_consts)
            flat_batch_def_np = make_flat_def_batch_np_empty(self.dataset_consts)
            flat_batch = np_to_tensor(flat_batch_np)
            flat_batch_def = np_to_tensor_def(flat_batch_def_np)

            with StopWatch("Training"):
                self.train_step(flat_batch, flat_batch_def)
            with StopWatch("Validation"):
                self.valid_step(flat_batch)

        print_for_user("TF model compilation  complete")

    def run_training_loop(self):
        max_args_num = max(self.dataset_consts.tactic_index_to_numargs)
        global_context_size = len(self.dataset_consts.global_context)
        steps = 0

        data_gd = self.data_server.def_cluster_subgraphs()
        batches_gd = Repeat(data_gd, shuffled=self._shuffle_def)
        batches_gd = Batches(batches_gd, self.batch_size)

        if self.from_checkpoint is None:
            self.checkpointer.make_checkpoint(epoch=0)
            self.run_validation(0)
            epoch_range = range(self.num_epochs)
        else:
            epoch_range = range(self.from_checkpoint, self.from_checkpoint+self.num_epochs)
        for epoch in epoch_range:
            if steps >= self.max_steps:
                print_for_user(f"exiting epoch loop early")
                break

            with StopWatch("Epoch"):

                self.metrics.reset_states()

                with StopWatch("Training"):
                    data_train = self.data_server.data_train(shuffled=True)
                    batches_sa = Batches(data_train, self.batch_size)
                    progress_bar = tqdm(batches_sa, desc=f"TRAIN | Epoch {epoch}/{self.num_epochs} | train", leave=False)
                    for batch_sa in progress_bar:
                        if steps >= self.max_steps:
                            print_for_user(f"exiting training loop early after {steps} total steps")
                            break
                        steps += 1
                        batch_gd = next(batches_gd)

                        with StopWatch("Step"):
                            flat_batch_np = make_flat_batch_np(batch_sa, global_context_size, max_args_num)
                            flat_batch_def_np = make_flat_def_batch_np(batch_gd)
                            flat_batch = np_to_tensor(flat_batch_np)
                            flat_batch_def = np_to_tensor_def(flat_batch_def_np)

                            self.train_step(flat_batch, flat_batch_def)

                    print_for_user(f"Epoch {epoch}/{self.num_epochs} | train | " + ", ".join([
                        f"tac_loss: {self.metrics['train_tac_loss'].result():#.3g}",
                        f"arg_loss: {self.metrics['train_arg_loss'].result():#.3g}",
                        f"act_loss: {self.metrics['train_act_loss'].result():#.3g}",
                        f"def_loss: {self.metrics['train_def_loss'].result():#.3g}",
                        f"loss: {self.metrics['train_loss'].result():#.3g}",
                        f"log_prob: {self.metrics['train_act_log_prob'].result():#.3g}",
                        f"tac_acc: {self.metrics['train_tac_acc'].result():.02f}",
                        f"arg_acc: {self.metrics['train_arg_acc'].result():.02f}",
                        f"acc: {self.metrics['train_acc'].result():.02f}",
                    ]))

                self.run_validation(epoch+1)

            # record results
            results: Dict[str, Any] = {"Epoch": epoch}
            results.update(self.metrics.all_results())
            results.update(get_times())
            self.recorder.record_results(results)

            self.fingerprinter.append([
                self.metrics["train_tac_acc"].result(),
                self.metrics["train_arg_acc"].result(),
                self.metrics["valid_tac_acc"].result(),
                self.metrics["train_tac_loss"].result(),
                self.metrics["train_arg_loss"].result(),
                self.metrics["valid_tac_loss"].result()
            ])

            self.checkpointer.make_checkpoint(epoch=epoch+1)

        self.fingerprinter.record()
        print_for_user(f"Training complete.")

    def run_validation(self, epoch):
        max_args_num = max(self.dataset_consts.tactic_index_to_numargs)
        global_context_size = len(self.dataset_consts.global_context)

        with StopWatch("Validation"):
            data_valid = self.data_server.data_valid()
            batches = Batches(data_valid, self.batch_size)
            progress_bar = tqdm(batches, desc=f"TRAIN | Epoch {epoch}/{self.num_epochs} | valid", leave=False)
            for batch in progress_bar:
                with StopWatch("Step"):
                    flat_batch_np = make_flat_batch_np(batch, global_context_size, max_args_num)
                    self.valid_step(np_to_tensor(flat_batch_np))
            print_for_user(f"Epoch {epoch}/{self.num_epochs} | valid | " + ", ".join([
                f"tac_loss: {self.metrics['valid_tac_loss'].result():#.3g}",
                f"arg_loss: {self.metrics['valid_arg_loss'].result():#.3g}",
                f"log_prob: {self.metrics['valid_act_log_prob'].result():#.3g}",
                f"tac_acc: {self.metrics['valid_tac_acc'].result():.02f}",
                f"arg_acc: {self.metrics['valid_arg_acc'].result():.02f}",
                f"acc: {self.metrics['valid_acc'].result():.02f}",
            ]))

class DefaultParamsAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        header=(
            "# Default training parameters YAML:\n"
            "\n"
            "# The YAML parameter file passed to g2t-train --params\n"
            "# can be any subset of these parameters, including an empty file.\n"
            "\n"
            "# See the TrainingParam class for documentation of each parameter.\n"
            "\n"
        )
        yaml_file = header + TrainingParams().to_yaml()
        print(yaml_file)
        exit()


def main():
    #fire.Fire(_main)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Location of the data"
    )
    parser.add_argument(
        "params",
        type=Path,
        help="YAML parameter file (can be an empty file for all default values)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path('.'),
        help=("directory to store checkpoints, logs, and other training outputs "
              "generated per each trainning run"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path('.'),  # for work_dir we can't have None as we must provide a directory
                            # as a workspace to store our intermediate representation
                            # we can think what is a better default a more frequent workflow
        help=("the work directory for intermediate data representation "
              "hash tables and indices (generated once per imported dataset) "
              "default is the current directory")
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="use cached data if available instead of loading the data from source"
    )
    parser.add_argument(
        "--fingerprint",
        type=Path,
        help="file to store fingerprint (used for testing)"
    )
    parser.add_argument(
        "--default-params",
        action=DefaultParamsAction,  # prints default YAML and exits
        help="prints a YAML file with default parameter values and exits"
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        default="ERROR",
        help="logging level: ERROR, WARNING, INFO, DEBUG"
    )
    parser.add_argument(
        "--max-steps",
        type=float,  # to support a default of infinity, we make this a float
        default=float("inf"),
        help="a debugging parameter which ends training this number of training steps"
    )
    parser.add_argument(
        "--from-checkpoint",
        type=int,
        default=None,
        help="loads a trained model and trains from that point on (with their parameters)"
    )
    args = parser.parse_args()

    # check inputs
    assert args.data_dir.is_dir(), f"{args.data_dir} must be an existing directory"
    assert args.work_dir.is_dir(), f"{args.work_dir} must be an existing directory"
    assert args.params.is_file(), f"{args.params} must be a yaml file"
    assert args.logging_level in ["ERROR", "WARNING", "INFO", "DEBUG"], \
        f"{args.logging_level} must be a a valid logging level"

    loop = TrainingLoop(
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        params=TrainingParams.from_yaml_file(args.params),
        output_dir=args.output_dir,
        fingerprint=args.fingerprint,
        logging_level=args.logging_level,
        use_cache=args.use_cache,
        max_steps=args.max_steps,
        from_checkpoint=args.from_checkpoint,
    )
    print_for_user("Training loop initialized")
    loop.run_training_loop()

if __name__ == "__main__":
    main()
