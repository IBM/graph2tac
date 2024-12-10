from abc import ABC, abstractmethod
import argparse
from collections import Counter, defaultdict
import dataclasses
import itertools
import json
import gzip
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Literal, TypedDict
import numpy as np
import numpy.typing as npt
import tqdm
import tensorflow as tf

NameId = int
TacticId = int
DataId = int

class DataPoint(TypedDict):
    """Metadata for a proofstate-tactic pair (implemented as a Python dictionary)"""
    tactic_id: TacticId
    name_id: NameId
    prev_indices: list[DataId]
    """Indices of the previous thousand (or fewer) datapoints"""

class DataProcessor:
    """Processes the extracted knn data stored in .jsonl or .jsonl.gz files."""
    data: list[DataPoint]

    def __init__(self, data_path: Path, limit: int|None):
        """Initialize DataProcessor

        :param data_path: File (.jsonl or .jsonl.gz) or directory of such files
        :param limit: Maximum number of datapoints to retreive (useful for testing), else None to retrieve all data.
        """
        self.data_path = data_path
        self.counter: DataId = 0
        self.data = []
        self.name_i_to_indices: dict[NameId, list[DataId]] = defaultdict(list)
        self.name_i_to_previous_n: dict[NameId, list[DataId]] = defaultdict(list)
        self.bad_examples: set[NameId] = set()
        self.limit = limit
    
    def _reached_limit(self):
        return (self.limit is not None and self.counter >= self.limit)  

    def _get_most_recent_n_examples(self, name_i_to_indices: dict[NameId, list[DataId]], global_cxt: list[NameId], size: int) -> list[DataId]:
        global_cxt_ = reversed(sorted(global_cxt))
        output = (x for name_i in global_cxt_ for x in reversed(name_i_to_indices[name_i]))
        output = list(itertools.islice(output, size))  # take first n elements of generator where n=size
        if len(output) > size:
            output = output[:size]
        elif len(output) < size:
            output.extend([-1] * (size - len(output))) 
        assert len(output) == size, len(output)
        return output

    def _process_line(self, line: str|bytes):
        if self._reached_limit():
            return

        i = self.counter
        self.counter += 1

        datapoint: dict[str, Any] = json.loads(line)
        global_cxt: list[NameId] = datapoint["global_context"]
        name_id: NameId = datapoint["metadata_name_id"]
        
        if name_id in self.name_i_to_previous_n:
            # global_cxt only depends on the theorem, not the proof step
            last_thousand = self.name_i_to_previous_n[name_id]
        else:
            last_thousand = self._get_most_recent_n_examples(self.name_i_to_indices, global_cxt, 1000)
            self.name_i_to_previous_n[name_id] = last_thousand
        
        self.data.append({
            "tactic_id": datapoint["tactic_id"],
            "name_id": name_id,
            "prev_indices": last_thousand
        })
        self.name_i_to_indices[name_id].append(i)

        if name_id != max(datapoint["global_context"], default=0) + 1:
            self.bad_examples.add(name_id)

    def _process_jsonl_file(self, data_jsonl: Path, disable_progress_bar: None|bool = None):
        with data_jsonl.open() as f:
            for line in tqdm.tqdm(f, disable=disable_progress_bar):
                self._process_line(line)
    
    def _process_jsonl_gz_file(self, data_jsonl_gz: Path, disable_progress_bar: None|bool = None):
        with gzip.open(data_jsonl_gz) as f:
            for line in tqdm.tqdm(f, leave=False, disable=disable_progress_bar):
                self._process_line(line)

    def _process_path(self, data_path: Path, disable_progress_bar: None|bool = None):
        if self._reached_limit():
            return

        if data_path.is_dir():
            # process files in order by there index
            # files are of format data123.jsonl or data123.jsonl.gz
            data_files = sorted(data_path.glob("data*.jsonl*"), key=lambda p: int(p.stem[4:].split(".")[0]))
            for data_file in tqdm.tqdm(data_files, disable=disable_progress_bar):
                self._process_path(data_file, disable_progress_bar=True)
        elif data_path.is_file() and data_path.suffix == ".jsonl":
            self._process_jsonl_file(data_path, disable_progress_bar=disable_progress_bar)
        elif data_path.is_file() and data_path.suffix == ".gz":
            self._process_jsonl_gz_file(data_path, disable_progress_bar=disable_progress_bar)
        else:
            raise ValueError(f"Incorrect file: {data_path}")
    
    def process(self):
        """Process data

        :return: Processed data
        """
        self._process_path(self.data_path)

    @staticmethod
    def process_data(data_dir: Path, limit: None | int) -> list[DataPoint]:
        """Process data

        :param data_path: File (.jsonl or .jsonl.gz) or directory of such files
        :param limit: Maximum number of datapoints to retreive (useful for testing), else None to retrieve all data.
        :return: Processed data
        """
        data_processor = DataProcessor(data_dir, limit=limit)
        data_processor.process()
        print("ID not at end of global context", len(data_processor.bad_examples))
        return data_processor.data


def process_context_names(name_jsonl: Path) -> list[str]:
    """Process the names of all declarations

    :param name_jsonl: Path of context_names.jsonl
    :return: List of all names indexed by name_id
    """
    with name_jsonl.open() as f:
        names = [name for name in f]
    return names

def process_tactics(tactic_jsonl: Path) -> list[str]:
    """Process the names of all tactics

    :param name_jsonl: Path of tactic_names.jsonl
    :return: List of all tactics indexed by name_id
    """
    with tactic_jsonl.open() as f:
        tactics = [tactic for tactic in f]
    return tactics

def process_all_embeddings(emb_dir: Path) -> npt.NDArray:
    """Process the proof state embeddings

    :param emb_dir: Path to directory with embeddings in .npy files
    :return: Array with dimensions [data_ids, hdim] of type float32
    """
    data = []
    emb_files = sorted(emb_dir.glob("embeddings*.npy"), key=lambda p: int(p.stem[len("embeddings"):].split(".")[0]))
    for emb_file in emb_files:
        embs = np.load(emb_file)
        data.append(embs[:, 0])
    return np.concatenate(data, axis=0)

def check_if_tactic_is_in_history(data: list[DataPoint]):
    checks = 0
    count = 0
    for d in tqdm.tqdm(data):
        tactic_id = d["tactic_id"]
        if any(i != -1 and data[i]["tactic_id"] == tactic_id for i in d["prev_indices"]):
            checks += 1
        count += 1
    print("Tactics in history:", checks, "Total data size:", count)

def check_libraries(data: list[DataPoint], names: list[str]):
    prefixes = Counter()
    double_prefixes = Counter()
    for d in data:
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        prefixes[prefix] += 1
        double_prefix = ".".join(names[name_id].split(".")[:2])
        double_prefixes[double_prefix] += 1
    print("Num of prefixes: ", prefixes, "Number of double prefixes:", double_prefixes)

def count_proofstates_with_new_tactics(data: list[DataPoint], names: list[str]):
    good = Counter()
    bad = Counter()
    coq_tactics = set()
    for d in data:
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        if prefix != "Coq":
            continue 
        tactic_id = d["tactic_id"]
        coq_tactics.add(tactic_id)

    for d in data:
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        if prefix == "Coq":
            continue 
        tactic_id = d["tactic_id"]

        if tactic_id in coq_tactics:
            good[prefix] += 1
        else:
            bad[prefix] += 1
    print("Proofstates which have tactics not in Coq prefix:", sum(bad[x] for x in bad), bad)

def train_valid_datasets(data: list[DataPoint], names: list[str], embeddings):
    embeddings = np.concatenate([embeddings, np.ones([1, embeddings.shape[1]])])
    train_query = []
    train_keys = []
    train_correct = []
    valid_query = []
    valid_keys = []
    valid_correct = []
    for i, d in enumerate(data):
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        query = embeddings[i]
        keys = embeddings[d["prev_indices"]]
        tactic_id = d["tactic_id"]
        correct = np.array([j != -1 and data[j]["tactic_id"] == tactic_id for j in d["prev_indices"]])
        correct_sum = sum(correct)
        if correct_sum == 0:
            continue
        correct = [c / correct_sum for c in correct]

        if prefix == "Coq":
            train_query.append(query)
            train_keys.append(keys)
            train_correct.append(correct)
        else:
            valid_query.append(query)
            valid_keys.append(keys)
            valid_correct.append(correct)
    return train_query, train_keys, train_correct, valid_query, valid_keys, valid_correct
        
def train_valid_datasets2(data: list[DataPoint], names: list[str]):
    train_query = []
    train_keys = []
    train_correct = []
    valid_query = []
    valid_keys = []
    valid_correct = []
    for i, d in enumerate(tqdm.tqdm(data)):
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        tactic_id = d["tactic_id"]
        correct = np.array([j != -1 and data[j]["tactic_id"] == tactic_id for j in d["prev_indices"]])
        correct_sum = sum(correct)
        if correct_sum == 0:
            continue
        correct = [c / correct_sum for c in correct]
        if prefix == "Coq":
            train_query.append(i)
            train_keys.append(d["prev_indices"])
            train_correct.append(correct)
        else:
            valid_query.append(i)
            valid_keys.append(d["prev_indices"])
            valid_correct.append(correct)
    return np.array(train_query), np.array(train_keys), np.array(train_correct), np.array(valid_query), np.array(valid_keys), np.array(valid_correct)


class Embeddings(tf.keras.layers.Layer):
    def __init__(self, embeddings: npt.NDArray):
        super().__init__(name="proofstate_embeddings")
        # add element at the end for case where key does not exist
        self.hdim = embeddings.shape[1]
        embeddings = np.concatenate([embeddings, np.ones([1, self.hdim])])
        self.embeddings = tf.constant(embeddings, dtype=tf.float32)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embeddings": None,
        })
        return config
    
    def call(self, query_id: tf.Tensor, key_ids: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        query = tf.gather(self.embeddings, query_id % len(self.embeddings))
        keys = tf.gather(self.embeddings, key_ids % len(self.embeddings))
        return query, keys


class Temp(tf.keras.layers.Layer):
    def __init__(self, init_temp: float, name="temp"):
        super().__init__(name=name)
        self.temp = tf.Variable(initial_value=init_temp, trainable=True)
    
    def call(self, logits: tf.Tensor):
        return logits / self.temp


class QueryKeyMult(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="query_key_mult")

    def call(self, query, keys):
        #tf.print("tf_query", x)
        #tf.print("tf_keys", y)
        x = query
        y = keys
        # normalize
        x = x / tf.norm(x, keepdims=True, axis=-1)
        y = y / tf.norm(y, keepdims=True, axis=-1)
        #tf.print("tf_query_norm", x)
        #tf.print("tf_keys_norm", y)
        # multiply
        logits = tf.einsum("ik,ijk->ij", x, y)
        # temp
        #logits = logits / temp
        #tf.print("tf_logits", logits)
        # softmax
        #logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        #probs = tf.exp(logits)
        #logits = logits - tf.math.log(tf.reduce_sum(probs, axis=-1, keepdims=True))
        
        # output
        return logits


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, hdim: int, residual: bool, dropout: bool, name: str):
        super().__init__(name=name)
        self.layers = []
        for _ in range(num_layers):
            if dropout:
                self.layers.append(tf.keras.layers.Dropout(rate=0.1))
            self.layers.append(tf.keras.layers.Dense(hdim, activation="relu"))
        self.residual = residual
    
    def call(self, inputs: tf.Tensor):
        x0 = inputs
        x = inputs
        for layer in self.layers:
            x: tf.Tensor = layer(x)
        if self.residual:
            x = x + x0
        return x


class SoftmaxCollapse(tf.keras.layers.Layer):
    def __init__(self, num_tactics):
        super().__init__(name="softmax_collapse")
        # add one for special null tactic id
        self.num_tactics = num_tactics + 1
    
    def call(
            self, 
            tactic_logits: tf.Tensor,  # [batch, history]
            tactic_ids: tf.Tensor,     # [batch, history]
    ):
        batch_size = tf.shape(tactic_logits)[0]
        history_size = tf.shape(tactic_logits)[1]

        # flatten the arrays
        # and make tactic ids distinct across batch ids
        # so that when we combine like tactic ids, we don't combine across batch ids
        # [batch, 1]
        offset = tf.expand_dims(tf.range(batch_size, dtype=tf.int64) * self.num_tactics, axis=-1)
        # [batch * history]
        tactic_ids = tf.reshape(tf.cast(tactic_ids, dtype=tf.int64) + offset, shape=[batch_size * history_size])
        tactic_logits = tf.reshape(tactic_logits, shape=[batch_size * history_size])
        
        # [batch * all_tactics]
        maxs = tf.math.unsorted_segment_max(tactic_logits, tactic_ids, batch_size * self.num_tactics)
        # [batch * history]
        maxs_ = tf.gather(maxs, tactic_ids)
        tactic_probs = tf.exp(tactic_logits - maxs_)
        # [batch * all_tactics]
        tactic_probs = tf.math.unsorted_segment_sum(tactic_probs, tactic_ids, batch_size * self.num_tactics)
        tactic_logits = tf.math.log(tactic_probs) + maxs

        # [batch, all_tactics]
        predictions = tf.reshape(tactic_logits, shape=[batch_size, self.num_tactics])
        return predictions
    

class SoftmaxCombine(tf.keras.layers.Layer):
    def __init__(self, ratio):
        super().__init__(name="softmax_combine")
        # add one for special null tactic id
        self.ratio = ratio
    
    def call(
        self, 
        knn_predictions: tf.Tensor,   # [batch, tactics+1]
        cls_predictions: tf.Tensor, # [batch, tactics+1]
    ):
        # make log probs
        knn_predictions = tf.math.log_softmax(knn_predictions)
        cls_predictions = tf.math.log_softmax(cls_predictions)

        # scale by ratio
        knn_predictions = knn_predictions + tf.math.log(self.ratio)
        cls_predictions = cls_predictions + tf.math.log(1-self.ratio)

        # subtract max for numerical stability
        maxs = tf.maximum(knn_predictions, cls_predictions)
        knn_predictions = knn_predictions - maxs
        cls_predictions = cls_predictions - maxs

        # add in prob space
        predictions = tf.math.log(tf.exp(knn_predictions) + tf.exp(cls_predictions))

        # add back maxs
        predictions = predictions + maxs

        return predictions


class ModelBuilder:
    def __init__(
        self,
        num_tactics: int,
        embeddings: Embeddings,
        query_emb_layer: None | FeedForward, 
        key_emb_layer: None | FeedForward, 
        class_emb_layer: None | FeedForward,
        class_layer: None | tf.keras.layers.Dense,
        knn_class_prob_ratio: float,
        temp_layer: None | Temp,
    ):
        """Build model

        :param embeddings: Layer wrapping the embeddings.
        :param query_emb_layer: The trainable query layer used for the knn model.  If None, use initial query embeddings.
        :param key_emb_layer: The trainable query layer used for the knn model.  If None, use initial key embeddings.
        :param class_emb_layer: The trainable classifier embedding layer.  If None, use initial query embeddings.
        :param class_layer: Classification dense layer.  (If None, don't use classification.)
        :param knn_class_prob_ratio: Ratio to mix probs for knn and classifer.  If 1.0 only train knn.  If 0.0 only train classifier.
        :param temp_layer: The tempature parameter layer.
        
        knn_classifer_weight
        """
        self.num_tactics = num_tactics
        self.embeddings = embeddings
        self.query_emb_layer = query_emb_layer
        self.key_emb_layer = key_emb_layer
        self.class_emb_layer = class_emb_layer
        self.class_layer = class_emb_layer
        self.temp_layer = temp_layer
        self.knn_class_prob_ratio = knn_class_prob_ratio
        self.class_layer = class_layer
        
        assert 0.0 <= self.knn_class_prob_ratio and self.knn_class_prob_ratio <= 1.0, self.knn_class_prob_ratio
        self.use_knn = (self.knn_class_prob_ratio > 0.0)
        self.use_class = (self.knn_class_prob_ratio < 1.0)
        if self.use_class:
            assert class_layer is not None
            assert class_layer.units == self.num_tactics + 1

        if self.use_knn:
            assert temp_layer is not None
    
    def build_prediction_model(self) -> tf.keras.Model:
        hdim = self.embeddings.hdim
        # training model store embeddings explicitly to reduce memory in training 
        query = tf.keras.layers.Input(shape=(hdim,), dtype=tf.float32)
        keys = tf.keras.layers.Input(shape=(1000, hdim), dtype=tf.float32)
        tactic_ids = tf.keras.layers.Input(shape=(1000,), dtype=tf.int32)
        
        if self.use_knn and self.query_emb_layer is not None:
            knn_query = self.query_emb_layer(query)
        else:
            knn_query = query
        
        if self.use_knn and self.key_emb_layer is not None:
            knn_keys = self.key_emb_layer(keys)
        else:
            knn_keys = keys
        
        if self.use_class and self.class_emb_layer is not None:
            class_query = self.class_emb_layer(query)
        else:
            class_query = query

        if self.use_knn:
            assert self.temp_layer is not None
            logits = QueryKeyMult()(knn_query, knn_keys)
            logits = self.temp_layer(logits)
            knn_prediction = SoftmaxCollapse(num_tactics)(logits, tactic_ids)
        
        if self.use_class:
            assert self.class_layer is not None
            class_prediction = self.class_layer(class_query)
        
        if not self.use_knn:
            prediction = class_prediction
        elif not self.use_class:
            prediction = knn_prediction
        else:
            SoftmaxCombine(ratio=self.knn_class_prob_ratio)(knn_prediction, class_prediction)
            prediction = self.knn_class_prob_ratio * knn_prediction + (1-self.knn_class_prob_ratio) * class_prediction

        model = tf.keras.Model(inputs=[query, keys, tactic_ids], outputs=prediction)

        return model

    def build_model(self) -> tuple[tf.keras.Model, tf.keras.Model]:
        # training model store embeddings explicitly to reduce memory in training 
        query_id = tf.keras.layers.Input(shape=tuple(), dtype=tf.int32)
        key_ids = tf.keras.layers.Input(shape=(1000,), dtype=tf.int32)
        tactic_ids = tf.keras.layers.Input(shape=(1000,), dtype=tf.int32)

        query, keys = self.embeddings(query_id, key_ids)
        
        pred_model = self.build_prediction_model()
        prediction = pred_model([query, keys, tactic_ids])

#        if self.use_knn and self.query_emb_layer is not None:
#            knn_query = self.query_emb_layer(query)
#        else:
#            knn_query = query
#        
#        if self.use_knn and self.key_emb_layer is not None:
#            knn_keys = self.key_emb_layer(keys)
#        else:
#            knn_keys = keys
#        
#        if self.use_class and self.class_emb_layer is not None:
#            class_query = self.class_emb_layer(query)
#        else:
#            class_query = query
#
#        if self.use_knn:
#            assert self.temp_layer is not None
#            logits = QueryKeyMult()(knn_query, knn_keys)
#            logits = self.temp_layer(logits)
#            knn_prediction = SoftmaxCollapse(num_tactics)(logits, tactic_ids)
#        
#        if self.use_class:
#            assert self.class_layer is not None
#            class_prediction = self.class_layer(class_query)
#        
#        if not self.use_knn:
#            prediction = class_prediction
#        elif not self.use_class:
#            prediction = knn_prediction
#        else:
#            SoftmaxCombine(ratio=self.knn_class_prob_ratio)(knn_prediction, class_prediction)
#            prediction = self.knn_class_prob_ratio * knn_prediction + (1-self.knn_class_prob_ratio) * class_prediction

        model = tf.keras.Model(inputs=[query_id, key_ids, tactic_ids], outputs=prediction)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003, weight_decay=0.1, global_clipnorm=1.0),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True), tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        return model, pred_model


@dataclasses.dataclass
class DataGroup:
    query: npt.NDArray
    keys: npt.NDArray
    tactic_ids: npt.NDArray
    correct: npt.NDArray

    def sub_sample(self, size: int) -> "DataGroup":
        num_samples = len(self.query)
        if num_samples > size:
            rng = np.random.RandomState(0)
            subsample = rng.choice(np.arange(num_samples), size=size, replace=False)
            return DataGroup(
                query = self.query[subsample],
                keys = self.keys[subsample],
                tactic_ids = self.tactic_ids[subsample],
                correct = self.correct[subsample],
            )
        else:
            return self
        
class DataSplitter:
    def __init__(self, prefixes: list[str], train_size: int, valid_size: int):
        self.prefixes = prefixes
        self.train_size=train_size
        self.valid_size=valid_size
    
    @staticmethod
    def is_train_thm(name: str) -> bool:
        prefix = name.split(".")[0]
        return (prefix == "Coq")
    
    def train_valid_datasets(self, data: list[DataPoint], names: list[str], num_tactics: int) -> tuple[DataGroup, DataGroup]:
        train_query = []
        train_keys = []
        train_tactic_ids = []
        train_correct = []
        valid_query = []
        valid_keys = []
        valid_tactic_ids = []
        valid_correct = []
        print("Split data:")
        for i, d in enumerate(tqdm.tqdm(data)):
            name_id = d["name_id"]
            name = names[name_id]
            tactic_id = d["tactic_id"]
            tactic_ids = np.array([num_tactics if j == -1 else data[j]["tactic_id"] for j in d["prev_indices"]])
            if all(id != tactic_id for id in tactic_ids):
                continue
            if self.is_train_thm(name):
                train_query.append(i)
                train_keys.append(d["prev_indices"])
                train_tactic_ids.append(tactic_ids)
                train_correct.append(tactic_id)
            else:
                valid_query.append(i)
                valid_keys.append(d["prev_indices"])
                valid_tactic_ids.append(tactic_ids)
                valid_correct.append(tactic_id)
        
        train_data = DataGroup(
            query = np.array(train_query), 
            keys = np.array(train_keys), 
            tactic_ids = np.array(train_tactic_ids),
            correct = np.array(train_correct)
        )
        valid_data = DataGroup(
            query = np.array(valid_query), 
            keys = np.array(valid_keys), 
            tactic_ids = np.array(valid_tactic_ids),
            correct = np.array(valid_correct)
        )

        train_data = train_data.sub_sample(self.train_size)
        valid_data = valid_data.sub_sample(self.valid_size)

        return train_data, valid_data


class ModelTrainer:
    def __init__(self, model: tf.keras.Model, train_data: DataGroup, valid_data: DataGroup):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data

    def train_model(self, epochs: int, batch_size: int = 128):

        print("Learnable in model", {layer.name: [var.name for var in layer.trainable_variables] for layer in self.model.layers})

        self.model.fit(
            x=[self.train_data.query, self.train_data.keys, self.train_data.tactic_ids],
            y=self.train_data.correct,
            epochs=epochs,
            validation_data=([self.valid_data.query, self.valid_data.keys, self.valid_data.tactic_ids], self.valid_data.correct),
            batch_size=batch_size
        )

        for layer in self.model.layers:
            print(layer.name, layer.get_config(), layer.get_weights())
        

class AccuracyMeasurement(ABC):
    def __init__(
        self,
        data: list[DataPoint],
        embeddings: None | npt.NDArray,
        names: list[str],
        limit: None | int = None
    ):
        self.data = data
        if embeddings is None:
            embeddings = np.zeros([1, 1])
        # Add extra dummy logit on the end which is far away from the other logits
        # so that when prev_indices = -1 it selects that logit
        self.embeddings = np.concatenate([embeddings, np.ones([1, embeddings.shape[1]])])
        self.names = names
        self.limit = limit
    
    @abstractmethod
    def _calc_logits(self, query: npt.NDArray, keys: npt.NDArray, prev_tactic_ids: list[TacticId]) -> tuple[npt.NDArray, list[TacticId]]:
        """Calculate logits for a given single query and list of keys

        :param query: Array of shape [hdim]
        :param keys: Array of shape [prev_tactics, hdim] where prev is usually 1000
        :return: Array of logits with shape [returned_tactics] and the tactic ids of those logits (len: returned_tactics)
        """
        raise NotImplementedError("Must subclass")
    
    def _calc_logits_from_ids(self, query_id: DataId, key_ids: list[DataId], prev_tactic_ids: list[TacticId]) -> tuple[npt.NDArray, npt.NDArray]:
        """Calculate logits for a given single query and list of keys

        Overridden when evaluating a tf model where the embeddings are stored directly in the model

        :param query_id: Array of shape [hdim]
        :param key_ids: Keys of previous data points, length prev_tactics (usually 1000), using -1 for missing items.
        :param prev_tactic_ids: List of previous tactic ids, length prev_tactics (usually 1000), using -1 for missing items.
        :return: Array of logits with shape [returned_tactics] and the tactic ids of those logits (len: returned_tactics)
        """
        query = embeddings[query_id]
        keys = embeddings[key_ids] # if key_id is -1 it selects dummy key at the end
        logits, tactic_ids = self._calc_logits(query, keys, prev_tactic_ids)
        
        # filter out any dummy -1 tactic ids
        tactic_ids = np.array(tactic_ids)
        assert logits.shape == tactic_ids.shape
        mask = (tactic_ids != -1)
        logits = logits[mask]
        tactic_ids = tactic_ids[mask]

        return logits, tactic_ids
    
    @staticmethod
    def safe_divide(num: int, dom: int) -> float:
        if dom:
            return num / dom
        else:
            return 0.0
    
    def _sample_data_ids(self) -> Iterable[DataId]:
        if self.limit is None or self.limit >= len(data):
            return range(len(data))
        
        rng = np.random.RandomState(0)
        data_ids = rng.choice(len(self.data), size=self.limit, replace=False)
        return data_ids

    def measure_accuracy(self):
        train_correct = 0
        train_count = 0
        valid_correct = 0
        valid_count = 0
        novel_correct = 0
        novel_count = 0

        coq_tactics: set[TacticId] = set()
        for d in data:
            name_id = d["name_id"]
            prefix = names[name_id].split(".")[0]
            if prefix != "Coq":
                continue 
            tactic_id = d["tactic_id"]
            coq_tactics.add(tactic_id)
        
        for data_id in tqdm.tqdm(self._sample_data_ids()):
            d = data[data_id]
            query_id = data_id
            key_ids = d["prev_indices"]
            prev_tactic_ids = [data[i]["tactic_id"] if i != -1 else -1 for i in d["prev_indices"]]

            logits, tactic_ids = self._calc_logits_from_ids(query_id, key_ids, prev_tactic_ids)
            if len(logits):
                max_tactic_id: TacticId = tactic_ids[np.argmax(logits)]
                is_correct = (max_tactic_id == d["tactic_id"])
            else:
                is_correct = False
            
            name_id = d["name_id"]
            prefix = names[name_id].split(".")[0]
            if prefix == "Coq":
                train_correct += is_correct
                train_count += 1
            else:
                valid_correct += is_correct
                valid_count += 1
            
            if d["tactic_id"] not in coq_tactics:
                novel_correct += is_correct
                novel_count += 1
        
        print(
            "Accuracy Results",
            train_correct,
            train_count,
            self.safe_divide(train_correct, train_count),
            valid_correct,
            valid_count,
            self.safe_divide(valid_correct, valid_count),
            novel_correct,
            novel_count,
            self.safe_divide(novel_correct, novel_count),
        )


class RawLogitAccuracyMeasurement(AccuracyMeasurement):
    def __init__(
        self,
        data: list[DataPoint],
        embeddings: npt.NDArray,
        names: list[str],
        dist: Literal["cosign", "inner", "euclidean"],
        limit: None | int = None,
    ):
        super().__init__(
            data=data,
            embeddings=embeddings,
            names=names,
            limit=limit,
        )
        self.dist = dist
    
    def _calc_raw_logits(self, query: npt.NDArray, keys: npt.NDArray) -> npt.NDArray:
        """Calculate raw logits for a given single query and a list of keys

        :param query: Array of shape [hdim]
        :param keys: Array of shape [prev_tactics, hdim] where prev is usually 1000
        :return: Array of logits with shape [prev_tactics]
        """
        if self.dist == "cosign":
            query = query / np.sqrt(np.einsum("i,i->", query, query))
            keys = keys / np.sqrt(np.einsum("ji,ji->j", keys, keys))[:, np.newaxis]
        
        if self.dist == "inner" or self.dist == "cosign":
            logits = np.einsum("i,ji->j", query, keys)
        elif self.dist == "euclidean":
            # -(x - y)^2 == -x^2 + 2xy - y^2
            logits = -np.einsum("i,i->", query, query) + 2 * np.einsum("i,ji->j", query, keys) - np.einsum("ji,ji->j", keys, keys)
        else:
            raise Exception(f"unknown dist: {self.dist}")
        
        return logits


class MaxLogitAccuracyMeasurement(RawLogitAccuracyMeasurement):
    def _calc_logits(self, query: npt.NDArray, keys: npt.NDArray, prev_tactic_ids: list[TacticId]) -> tuple[npt.NDArray, list[TacticId]]:
        logits = self._calc_raw_logits(query, keys)
        return logits, prev_tactic_ids
    
def max_accuracy(data: list[DataPoint], embeddings, dist: str = "cosign"):
    embeddings = np.concatenate([embeddings, np.ones([1, embeddings.shape[1]])])

    train_correct = 0
    train_count = 0
    valid_correct = 0
    valid_count = 0
    novel_correct = 0
    novel_count = 0

    coq_tactics = set()
    for d in data:
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        if prefix != "Coq":
            continue 
        tactic_id = d["tactic_id"]
        coq_tactics.add(tactic_id)
    
    for i, d in enumerate(tqdm.tqdm(data)):
        query = embeddings[i]
        keys = embeddings[d["prev_indices"]]

        if dist == "cosign":
            query = query / np.sqrt(np.einsum("i,i->", query, query))
            keys = keys / np.sqrt(np.einsum("ji,ji->j", keys, keys))[:, np.newaxis]
        
        if dist == "inner" or dist == "cosign":
            logits = np.einsum("i,ji->j", query, keys)
        elif dist == "euclidean":
            # -(x - y)^2 == -x^2 + 2xy - y^2
            logits = -np.einsum("i,i->", query, query) + 2 * np.einsum("i,ji->j", query, keys) - np.einsum("ji,ji->j", keys, keys)
        else:
            raise Exception(f"unknown dist: {dist}")
        
        logits = np.einsum("i,ji->j", query, keys)
        max_i = d["prev_indices"][np.argmax(logits)]
        max_tactic_id = data[max_i]["tactic_id"]

        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        if prefix == "Coq":
            if max_tactic_id == d["tactic_id"]:
                train_correct += 1
            train_count += 1
        else:
            if max_tactic_id == d["tactic_id"]:
                valid_correct += 1
            valid_count += 1
        
        if d["tactic_id"] not in coq_tactics:
            if max_tactic_id == d["tactic_id"]:
                novel_correct += 1
            novel_count += 1

    print(
        train_correct,
        train_count,
        AccuracyMeasurement.safe_divide(train_correct, train_count),
        valid_correct,
        valid_count,
        AccuracyMeasurement.safe_divide(valid_correct, valid_count),
        novel_correct,
        novel_count,
        AccuracyMeasurement.safe_divide(novel_correct, novel_count),
    )

class SoftMaxLogitAccuracyMeasurement(RawLogitAccuracyMeasurement):
    def __init__(
        self,
        data: list[DataPoint],
        embeddings: npt.NDArray,
        names: list[str],
        dist: Literal["cosign", "inner", "euclidean"],
        temp: float,
        limit: None | int = None,
    ):
        super().__init__(
            data=data,
            embeddings=embeddings,
            names=names,
            dist=dist,
            limit=limit,
        )
        self.temp = temp
    
    def _calc_logits(self, query: npt.NDArray, keys: npt.NDArray, prev_tactic_ids: list[TacticId]) -> tuple[npt.NDArray, list[TacticId]]:
        logits = self._calc_raw_logits(query, keys)

        logits = logits / self.temp
        logits_max = np.max(logits)
        logits = logits - logits_max
        probs = np.exp(logits)

        combined_probs = Counter()
        for i, p in enumerate(probs):
            tactic_id = prev_tactic_ids[i]
            combined_probs[tactic_id] += p
        probs = []
        tactic_ids = []

        for tactic_id, p in combined_probs.items():
            probs.append(p)
            tactic_ids.append(tactic_id)

        return np.array(probs), tactic_ids
    
def softmax_accuracy(data: list[DataPoint], embeddings, names, temp, dist: str = "cosign"):
    embeddings = np.concatenate([embeddings, np.ones([1, embeddings.shape[1]])])

    train_correct = 0
    train_count = 0
    valid_correct = 0
    valid_count = 0
    novel_correct = 0
    novel_count = 0

    coq_tactics = set()
    for d in data:
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        if prefix != "Coq":
            continue 
        tactic_id = d["tactic_id"]
        coq_tactics.add(tactic_id)
    
    for i, d in enumerate(tqdm.tqdm(data)):
        query = embeddings[i]
        keys = embeddings[d["prev_indices"]]

        if dist == "cosign":
            query = query / np.sqrt(np.einsum("i,i->", query, query))
            keys = keys / np.sqrt(np.einsum("ji,ji->j", keys, keys))[:, np.newaxis]
        
        if dist == "inner" or dist == "cosign":
            logits = np.einsum("i,ji->j", query, keys)
        elif dist == "euclidean":
            # -(x - y)^2 == -x^2 + 2xy - y^2
            logits = -np.einsum("i,i->", query, query) + 2 * np.einsum("i,ji->j", query, keys) - np.einsum("ji,ji->j", keys, keys)
        else:
            raise Exception(f"unknown dist: {dist}")

        logits = logits / temp
        logits_max = np.max(logits)
        logits = logits - logits_max
        probs = np.exp(logits)
        combined_probs = Counter()
        for i, p in enumerate(probs):
            data_id = d["prev_indices"][i]
            if data_id == -1:
                continue
            tactic_id = data[data_id]["tactic_id"]
            combined_probs[tactic_id] += p
        max_tactic_id = max(combined_probs, key=lambda t: combined_probs[t], default=-1)
        
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        if prefix == "Coq":
            if max_tactic_id == d["tactic_id"]:
                train_correct += 1
            train_count += 1
        else:
            if max_tactic_id == d["tactic_id"]:
                valid_correct += 1
            valid_count += 1
        
        if d["tactic_id"] not in coq_tactics:
            if max_tactic_id == d["tactic_id"]:
                novel_correct += 1
            novel_count += 1

    print(
        train_correct,
        train_count,
        AccuracyMeasurement.safe_divide(train_correct, train_count),
        valid_correct,
        valid_count,
        AccuracyMeasurement.safe_divide(valid_correct, valid_count),
        novel_correct,
        novel_count,
        AccuracyMeasurement.safe_divide(novel_correct, novel_count),
    )

class FrequencyAccuracyMeasurement(AccuracyMeasurement):
    def _calc_logits(self, query: npt.NDArray, keys: npt.NDArray, prev_tactic_ids: list[TacticId]) -> tuple[npt.NDArray, list[TacticId]]:
        counts = Counter(prev_tactic_ids)

        probs = []
        tactic_ids = []
        for tactic_id, p in counts.items():
            probs.append(p)
            tactic_ids.append(tactic_id)

        return np.array(probs), tactic_ids
    
def frequency_accuracy(data: list[DataPoint], embeddings):
    train_correct = 0
    train_count = 0
    valid_correct = 0
    valid_count = 0
    novel_correct = 0
    novel_count = 0

    coq_tactics = set()
    for d in data:
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        if prefix != "Coq":
            continue 
        tactic_id = d["tactic_id"]
        coq_tactics.add(tactic_id)
    
    for i, d in enumerate(tqdm.tqdm(data)):
        combined_probs = Counter()
        for i, _ in enumerate(d["prev_indices"]):
            data_id = d["prev_indices"][i]
            if data_id == -1:
                continue
            tactic_id = data[data_id]["tactic_id"]
            combined_probs[tactic_id] += 1
        max_tactic_id = max(combined_probs, key=lambda t: combined_probs[t], default=-1)
        
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        if prefix == "Coq":
            if max_tactic_id == d["tactic_id"]:
                train_correct += 1
            train_count += 1
        else:
            if max_tactic_id == d["tactic_id"]:
                valid_correct += 1
            valid_count += 1

        if d["tactic_id"] not in coq_tactics:
            if max_tactic_id == d["tactic_id"]:
                novel_correct += 1
            novel_count += 1

    print(
        train_correct,
        train_count,
        AccuracyMeasurement.safe_divide(train_correct, train_count),
        valid_correct,
        valid_count,
        AccuracyMeasurement.safe_divide(valid_correct, valid_count),
        novel_correct,
        novel_count,
        AccuracyMeasurement.safe_divide(novel_correct, novel_count),
    )

class ModelAccuracyMeasurement(AccuracyMeasurement):
    def __init__(
        self,
        data: list[DataPoint],
        names: list[str],
        model: tf.keras.Model,
        num_tactics: TacticId,
        limit: None | int = None,
    ):
        super().__init__(
            data=data,
            embeddings=None,  # embeddings are in the model already
            names=names,
            limit=limit,
        )
        self.model = model
        self.num_tactics = num_tactics
        self.all_tactic_ids = np.arange(self.num_tactics)

    def _calc_logits(self, query: npt.NDArray, keys: npt.NDArray, prev_tactic_ids: list[TacticId]) -> tuple[npt.NDArray, list[TacticId]]:
        raise NotImplementedError("Not used for tf models.  Use _calc_logits_from_ids.")

    def _calc_logits_from_ids(self, query_id: int, key_ids: list[int], prev_tactic_ids: list[int]) -> tuple[npt.NDArray, npt.NDArray]:
        tactic_ids = np.array([self.num_tactics if tactic_id == -1 else tactic_id for tactic_id in prev_tactic_ids])
        logits = self.model([np.array([query_id]), np.array([key_ids]), np.array(tactic_ids)])[0].numpy()
        # remove last tactics which is out of bounds
        logits = logits[:-1]
        assert len(logits) == self.num_tactics
        return logits, self.all_tactic_ids

def softmax_accuracy_from_model(data: list[DataPoint], names, model, embeddings, num_tactics):
    embeddings = np.concatenate([embeddings, np.ones([1, embeddings.shape[1]])])

    train_correct = 0
    train_count = 0
    valid_correct = 0
    valid_count = 0
    novel_correct = 0
    novel_count = 0

    coq_tactics = set()
    for d in data:
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        if prefix != "Coq":
            continue 
        tactic_id = d["tactic_id"]
        coq_tactics.add(tactic_id)
    
    all_results = []
    for i, d in enumerate(tqdm.tqdm(data)):
        query_id = [i]
        key_ids = [d["prev_indices"]]
        tactic_ids = np.array([num_tactics if j == -1 else data[j]["tactic_id"] for j in d["prev_indices"]])

        logits = model([np.array(query_id), np.array(key_ids), np.array(tactic_ids)])[0].numpy()
        
        max_tactic_id = np.argmax(logits)

        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        if prefix == "Coq":
            if max_tactic_id == d["tactic_id"]:
                train_correct += 1
            train_count += 1
        else:
            if max_tactic_id == d["tactic_id"]:
                valid_correct += 1
            valid_count += 1
        result = {"id": i, "prediction": max_tactic_id, "correct": max_tactic_id == d["tactic_id"]}
        all_results.append(result)

        if d["tactic_id"] not in coq_tactics:
            if max_tactic_id == d["tactic_id"]:
                novel_correct += 1
            novel_count += 1

    print(
        train_correct,
        train_count,
        AccuracyMeasurement.safe_divide(train_correct, train_count),
        valid_correct,
        valid_count,
        AccuracyMeasurement.safe_divide(valid_correct, valid_count),
        novel_correct,
        novel_count,
        AccuracyMeasurement.safe_divide(novel_correct, novel_count),
    )

def experiment_softmax(data, names, embeddings: Embeddings, num_tactics, train_size, model_weights_path: Path):
    print("====")
    print("Softmax")
    print("====")
    temp_layer = Temp(init_temp=0.03)

    model, pred_model = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=None,
        key_emb_layer=None,
        class_emb_layer=None,
        class_layer=None,
        knn_class_prob_ratio=1.0,
        temp_layer=temp_layer,
    ).build_model()

    train_data, valid_data = DataSplitter(
        prefixes=["Coq"],
        train_size=train_size,
        valid_size=5000,
    ).train_valid_datasets(data=data, names=names, num_tactics=num_tactics)

    ModelTrainer(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
    ).train_model(
        epochs=10,
        batch_size=128
    )

    print("softmax hidden")
    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

def experiment_separate(data, names, embeddings: Embeddings, num_tactics, train_size, model_weights_path: Path):
    print("====")
    print("trained seperately")
    print("====")
    dim=128
    query_layer = FeedForward(hdim=dim, num_layers=3, residual=True, dropout=True, name="tactic_knn_query")
    key_layer = query_layer  # FeedForward(hdim=dim, num_layers=3, residual=True, dropout=True, name="tactic_knn_key")
    class_emb_layer = FeedForward(hdim=dim, num_layers=3, residual=True, dropout=True, name="tactic_class_emb")
    class_layer = tf.keras.layers.Dense(num_tactics+1)
    temp_layer = Temp(init_temp=0.03)

    class_model, class_pred_model = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=None,
        key_emb_layer=None,
        temp_layer=None,
        class_emb_layer=class_emb_layer,
        class_layer=class_layer,
        knn_class_prob_ratio=0.0,
    ).build_model()

    knn_model, knn_pred_model = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=query_layer,
        key_emb_layer=key_layer,
        temp_layer=temp_layer,
        class_emb_layer=None,
        class_layer=None,
        knn_class_prob_ratio=1.0,
    ).build_model()

    train_data, valid_data = DataSplitter(
        prefixes=["Coq"],
        train_size=train_size,
        valid_size=5000,
    ).train_valid_datasets(data=data, names=names, num_tactics=num_tactics)

    print("Train classifier")
    ModelTrainer(
        model=class_model,
        train_data=train_data,
        valid_data=valid_data,
    ).train_model(
        epochs=30,
        batch_size=128,
    )
    
    print("Train knn")
    ModelTrainer(
        model=knn_model,
        train_data=train_data,
        valid_data=valid_data,
    ).train_model(
        epochs=10,
        batch_size=128,
    )

    # test save and load pred model
    print("Save and reload model")
    knn_pred_model.save_weights(model_weights_path / "knn_model.h5")
    knn_pred_model.load_weights(model_weights_path / "knn_model.h5")

    print("classifier as trained")
    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=class_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("knn as trained")
    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=knn_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("knn query only")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=query_layer,
        key_emb_layer=query_layer,
        temp_layer=temp_layer,
        class_emb_layer=None,
        class_layer=None,
        knn_class_prob_ratio=1.0,
    ).build_model()

    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("knn key only")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=key_layer,
        key_emb_layer=key_layer,
        temp_layer=temp_layer,
        class_emb_layer=None,
        class_layer=None,
        knn_class_prob_ratio=1.0,
    ).build_model()

    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("knn class_emb only")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=class_emb_layer,
        key_emb_layer=class_emb_layer,
        temp_layer=temp_layer,
        class_emb_layer=None,
        class_layer=None,
        knn_class_prob_ratio=1.0,
    ).build_model()

    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("mixed - three layers")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=query_layer,
        key_emb_layer=key_layer,
        temp_layer=temp_layer,
        class_emb_layer=class_emb_layer,
        class_layer=class_layer,
        knn_class_prob_ratio=0.5,
    ).build_model()
    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("mixed - class_emb_layer everywhere")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=class_emb_layer,
        key_emb_layer=class_emb_layer,
        temp_layer=temp_layer,
        class_emb_layer=class_emb_layer,
        class_layer=class_layer,
        knn_class_prob_ratio=0.5,
    ).build_model()
    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("mixed - hidden for knn and class_emb for classifier")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=None,
        key_emb_layer=None,
        temp_layer=temp_layer,
        class_emb_layer=class_emb_layer,
        class_layer=class_layer,
        knn_class_prob_ratio=0.5,
    ).build_model()
    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

def experiment_mixed(data, names, embeddings: Embeddings, num_tactics, train_size, model_weights_path: Path):
    print("====")
    print("trained together")
    print("====")
    dim=128
    query_layer = FeedForward(hdim=dim, num_layers=3, residual=True, dropout=True, name="tactic_knn_query")
    key_layer = query_layer  # FeedForward(hdim=dim, num_layers=3, residual=True, dropout=True, name="tactic_knn_key")
    class_emb_layer = FeedForward(hdim=dim, num_layers=3, residual=True, dropout=True, name="tactic_class_emb")
    class_layer = tf.keras.layers.Dense(num_tactics+1)
    temp_layer = Temp(init_temp=0.03)

    model, pred_model = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=query_layer,
        key_emb_layer=key_layer,
        temp_layer=temp_layer,
        class_emb_layer=class_emb_layer,
        class_layer=class_layer,
        knn_class_prob_ratio=0.5,
    ).build_model()

    train_data, valid_data = DataSplitter(
        prefixes=["Coq"],
        train_size=train_size,
        valid_size=5000,
    ).train_valid_datasets(data=data, names=names, num_tactics=num_tactics)

    print("Train mixed model")
    ModelTrainer(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
    ).train_model(
        epochs=10,
        batch_size=128
    )

    print("model as trained")
    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("knn key and query")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=key_layer,
        key_emb_layer=query_layer,
        temp_layer=temp_layer,
        class_emb_layer=None,
        class_layer=None,
        knn_class_prob_ratio=1.0,
    ).build_model()

    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("knn query only")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=query_layer,
        key_emb_layer=query_layer,
        temp_layer=temp_layer,
        class_emb_layer=None,
        class_layer=None,
        knn_class_prob_ratio=1.0,
    ).build_model()

    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("knn key only")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=key_layer,
        key_emb_layer=key_layer,
        temp_layer=temp_layer,
        class_emb_layer=None,
        class_layer=None,
        knn_class_prob_ratio=1.0,
    ).build_model()

    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("knn class_emb only")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=class_emb_layer,
        key_emb_layer=class_emb_layer,
        temp_layer=temp_layer,
        class_emb_layer=None,
        class_layer=None,
        knn_class_prob_ratio=1.0,
    ).build_model()

    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("classifier only")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=None,
        key_emb_layer=None,
        temp_layer=None,
        class_emb_layer=class_emb_layer,
        class_layer=class_layer,
        knn_class_prob_ratio=0.0,
    ).build_model()
    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("mixed - class_emb_layer everywhere")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=class_emb_layer,
        key_emb_layer=class_emb_layer,
        temp_layer=temp_layer,
        class_emb_layer=class_emb_layer,
        class_layer=class_layer,
        knn_class_prob_ratio=0.5,
    ).build_model()
    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

    print("mixed - hidden for knn and class_emb for classifier")
    temp_model, _ = ModelBuilder(
        num_tactics=num_tactics,
        embeddings=embeddings,
        query_emb_layer=None,
        key_emb_layer=None,
        temp_layer=temp_layer,
        class_emb_layer=class_emb_layer,
        class_layer=class_layer,
        knn_class_prob_ratio=0.5,
    ).build_model()
    ModelAccuracyMeasurement(
        data=data,
        names=names,
        model=temp_model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()

# TODO: train matrix
# TODO: breakdown on train/valid
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='graph2tac knn model trainer')

    parser.add_argument(
        "--data-dir", "--data_dir",
        type=Path,
        required=True,
        help="Location of the data"
    )

    parser.add_argument(
        "--output-dir", "--output_dir",
        type=Path,
        required=True,
        help="Location of the output"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many datapoints to use"
    )


if __name__ == "__main__":
    data_dir = Path(sys.argv[1])
    names = process_context_names(data_dir / "context_names.jsonl")
    tactics = process_tactics(data_dir / "tactic_names.jsonl")

    # load data
    data = DataProcessor.process_data(data_dir / "data", limit=10000000)  # 10000000
    # perform checks and analysis
    check_if_tactic_is_in_history(data)
    check_libraries(data, names)
    count_proofstates_with_new_tactics(data, names)

    # load embeddings
    if (data_dir / "embeddings").is_dir():
        embeddings = process_all_embeddings(data_dir / "embeddings")
    else:
        embeddings = np.load(data_dir / "embeddings.npy")
    print("Embeddings shape:", embeddings.shape)
    
    num_tactics = len(tactics)
    embeddings_layer = Embeddings(embeddings)
    experiment_softmax(data=data, names=names, embeddings=embeddings_layer, num_tactics=num_tactics, train_size=100000, model_weights_path=data_dir / "models")
    experiment_separate(data=data, names=names, embeddings=embeddings_layer, num_tactics=num_tactics, train_size=100000, model_weights_path=data_dir / "models")
    experiment_mixed(data=data, names=names, embeddings=embeddings_layer, num_tactics=num_tactics, train_size=100000, model_weights_path=data_dir / "models")
    """
    model = train_model(data, names, embeddings, num_tactics)
    for layer in model.layers: print(layer.get_config(), layer.get_weights())
    #softmax_accuracy_from_model(data, names, model, embeddings, num_tactics)
    ModelAccuracyMeasurement(
        data=data,
        embeddings=embeddings,
        names=names,
        model=model,
        num_tactics=num_tactics,
        limit=10000,
    ).measure_accuracy()
    
    #max_accuracy(data, embeddings)
    MaxLogitAccuracyMeasurement(
        data=data,
        embeddings=embeddings,
        names=names,
        dist="cosign",
        limit=10000,
    ).measure_accuracy()
    #frequency_accuracy(data, embeddings)
    FrequencyAccuracyMeasurement(
        data=data,
        embeddings=embeddings,
        names=names,
        limit=10000,
    ).measure_accuracy()
    #for temp in [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9]:
    #for temp in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    #for temp in [1e-1, 6e-2, 3e-2, 1e-2, 6e-3, 3e-3, 1e-3]:
    for temp in [3e-2]:
        print(temp)
        #softmax_accuracy(data, embeddings, names, temp)
        SoftMaxLogitAccuracyMeasurement(
            data=data,
            embeddings=embeddings,
            names=names,
            dist="cosign",
            temp=temp,
            limit=10000,
        ).measure_accuracy()
    """
