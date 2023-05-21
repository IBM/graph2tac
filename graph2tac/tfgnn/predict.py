from typing import Tuple, List, Union, Iterable, Callable, Optional

import re
import yaml
import json
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from dataclasses import dataclass
from pathlib import Path

from graph2tac.loader.data_classes import DataConfig, GraphConstants, LoaderGraph, ProofstateMetadata, ProofstateContext, LoaderAction, LoaderActionSpec, LoaderProofstate, LoaderProofstateSpec, LoaderDefinition, LoaderDefinitionSpec
from graph2tac.loader.data_server import DataToTFGNN
from graph2tac.tfgnn.tasks import PredictionTask, TacticPrediction, DefinitionTask, GLOBAL_ARGUMENT_PREDICTION
from graph2tac.tfgnn.models import GraphEmbedding, LogitsFromEmbeddings
from graph2tac.tfgnn.train import Trainer
from graph2tac.common import logger
from graph2tac.predict import Predict, predict_api_debugging, cartesian_product, NUMPY_NDIM_LIMIT
from graph2tac.tfgnn.graph_schema import vectorized_definition_graph_spec, proofstate_graph_spec, batch_graph_spec
from graph2tac.tfgnn.stack_graph_tensors import stack_graph_tensors


class BeamSearch(tf.keras.layers.Layer):
    def __init__(self, logits_fn, stopping_fn, eos_id: int):
        super().__init__()
        self.logits_fn = logits_fn
        self.stopping_fn = stopping_fn
        self.eos_id = eos_id
        #self.max_beam_size = max_beam_size,
        # this needs to be a python int to make sure that the beam search knows how many steps of the loop to compile
    
    @staticmethod
    def expand_dims_none(x: tf.Tensor, axis: int):
        "Like tf.expand_dims, but sets the new axis dimension to be None instead of 1"
        x = tf.expand_dims(x, axis=axis)
        # set axis to None with this trick
        return tf.reshape(x, shape=tf.tensor_scatter_nd_update(tf.shape(x), [[axis % tf.rank(x)]], [-1]))
    
    def one_beam_step(
        self,
        ids: tf.Tensor,  # [batch_size, beam_size0, seq_length]
        scores: tf.Tensor,  # [batch_size, beam_size0]
        cache: dict[str, tf.Tensor],  # dict with values: [batch_size, beam_size0, ...]
        static_data: dict[str, tf.Tensor | tf.RaggedTensor],
        i: tf.Tensor,  # int32
        max_beam_size: tf.Tensor,  # int32
    ) -> tuple[
        tf.Tensor,  # ids: [batch_size, beam_size, seq_length+1]
        tf.Tensor,  # scores: [batch_size, beam_size]
        dict[str, tf.Tensor],  # cache with values: [batch_size, beam_size, ...]
    ]:
        batch_size = tf.shape(ids)[0]
        beam_size0 = tf.shape(ids)[1]

        # [batch, beam_size0, vocabulary], dict of [batch_size, beam_size0, ...]
        token_scores, cache = self.logits_fn(ids, i, cache, static_data)
        vocabulary = tf.shape(token_scores)[2]
        
        # don't return full beam_size if not enough vocabulary to support it yet
        beam_size = tf.minimum(max_beam_size, beam_size0 * vocabulary)

        all_scores = tf.expand_dims(scores, axis=-1) + token_scores  # [batch_size, beam_size0, vocabulary]
        all_scores = tf.reshape(all_scores, shape=[batch_size, beam_size0*vocabulary])  # [batch_size, beam_size0*vocabulary]
        
        top_k = tf.math.top_k(all_scores, k=beam_size, sorted=True)
        top_k_scores = top_k.values  # [batch_size, beam_size]
        top_k_ixs = top_k.indices  # [batch_size, beam_size]
        top_k_beams = top_k_ixs // vocabulary  # [batch_size, beam_size]
        top_k_tokens = top_k_ixs % vocabulary  # [batch_size, beam_size]

        # reorder the beams and add on the new tokens
        ids = tf.gather(ids, top_k_beams, batch_dims=1)  # [batch_size, beam_size, seq_length]
        ids = tf.concat([ids, tf.expand_dims(top_k_tokens, axis=-1)], axis=-1)  # [batch_size, beam_size, seq_length+1]
        
        # reorder cache
        cache = {
            k: tf.gather(v, top_k_beams, batch_dims=1)  # [batch_size, beam_size, ...]}
            for k, v in cache.items()  # v: [batch_size, beam_size0, ...]
        }

        return ids, top_k_scores, cache
    
    def one_beam_step_with_early_stopping(
        self,
        is_finished: tf.Tensor, # bool
        ids: tf.Tensor,  # [batch_size, beam_size0, seq_length]
        scores: tf.Tensor,  # [batch_size, beam_size0]
        cache: dict[str, tf.Tensor],  # dict with values: [batch_size, beam_size0, ...]
        static_data: dict[str, tf.Tensor | tf.RaggedTensor],
        i: tf.Tensor,  # int32
        max_beam_size: tf.Tensor,  # int32                           
    ) -> tuple[
        tf.Tensor,  # ids: [batch_size, beam_size, seq_length+1]
        tf.Tensor,  # scores: [batch_size, beam_size]
        dict[str, tf.Tensor],  # cache with values: [batch_size, beam_size, ...]
    ]:
        if is_finished:
            return (is_finished, (ids, scores, cache))
        
        # check which are finished
        is_finished, cache = self.stopping_fn(ids, i, cache, static_data)  # [batch_size, beam_size0]
        is_finished = tf.reduce_all(is_finished)

        # wait until all are finished
        if is_finished:
            return (is_finished, (ids, scores, cache))
        
        return (is_finished, self.one_beam_step(ids, scores, cache, static_data, i, max_beam_size))

    def call(
        self,
        initial_ids: tf.Tensor,  # [batch_size, initial_seq_length]
        initial_cache: dict[str, tf.Tensor],  # [batch_size, ...]
        static_data: dict[str, tf.Tensor | tf.RaggedTensor],
        beam_size: tf.Tensor,  # int32
        max_decode_length: tf.Tensor,  # int32
    ) -> tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(initial_ids)[0]
        
        # start with a beam_size of 1 for the input to the first step first step
        ids = self.expand_dims_none(initial_ids, axis=1)  # batch_size, beam_size0, initial_seq_length
        ids = tf.reshape(ids, shape=[tf.shape(ids)[0], tf.shape(ids)[1], -1])  # make sure sequence length is None
        cache = {
            k: self.expand_dims_none(v, axis=1)  # [batch_size, beam_size0, ...]
            for k, v in initial_cache.items()  # v: [batch_size, ...]
        }
        scores = self.expand_dims_none(tf.zeros(shape=[batch_size]), axis=-1)  # [batch_size, beam_size0]

        # iterate over the sequence
        is_finished = False
        # this loop is automatically converted into tf.while_loop
        for i in range(max_decode_length):
            # ids: [batch_size, beam_size, initial_seq_length]
            # scores: [batch_size, beam_size]
            # cache: dict with values [batch_size, beam_size, ...]
            is_finished, (ids, scores, cache) = self.one_beam_step_with_early_stopping(is_finished, ids, scores, cache, static_data, i, beam_size)

        return ids, scores


class Selection:
    def __init__(self, tactic_index_to_numargs: list[int]):
        self.tactic_index_to_numargs = tf.cast(tf.constant(tactic_index_to_numargs), tf.int32)
        self.eos_id = 0

        self.beam_search = BeamSearch(
            logits_fn=self.token_probability_fn,
            stopping_fn=self.stopping_fn,
            eos_id=self.eos_id,  
        )

        @tf.function(input_signature=[
            {
                "tactic": tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                "tactic_logits": tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                "local_arguments_logits": tf.RaggedTensorSpec(shape=[None, None, None], dtype=tf.float32),
                "global_arguments_logits": tf.RaggedTensorSpec(shape=[None, None, None], dtype=tf.float32),
            },
            tf.TensorSpec(shape=[], dtype=tf.int32),
        ])
        def select_results(inference_output, search_expand_bound):
            return self._select_results(inference_output, search_expand_bound)
    
        self.select_results = select_results

    def token_probability_fn(
        self,
        partial_seqs: tf.Tensor,  # [batch, beam_size, seq_length] dtype: int
        pos: tf.Tensor,  # scalar dtype: int32,
        cache: dict[str, tf.Tensor],  # value shapes: [batch, beam_size, ...]
        static_data: dict[str, tf.Tensor | tf.RaggedTensor]
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:  # output shapes [batch, beam_size, vocab], dict of [batch, beam_size, ...]
        if pos == 0:
            # look up tactic logits
            batch_ix = cache["batch_ix"]  # [batch, beam_size]
            tactic_logits = static_data["tactic_logits"]  # [batch, vocab]
            tactic_logits = tf.gather(tactic_logits, batch_ix)  # [batch, beam_size, vocab]
            return tactic_logits, cache

        # look up argument logits
        arg_pos = tf.cast(pos - 1, tf.int32)
        arg_logits = static_data["arg_logits"]  # [batch, tactic, arg, vocab]    
        batch_ix = cache["batch_ix"]  # [batch, beam_size]  
        # first token is the tactic id.  It may be a stop token but that is accounted for
        tactic_id = self.get_tactic(partial_seqs)  # [batch, beam_size]
        indices = tf.stack([batch_ix, tactic_id, tf.ones_like(batch_ix) * arg_pos], axis=-1)  # [batch, beam_size]    
        logits = tf.gather_nd(arg_logits, indices)
        return logits, cache

    @staticmethod
    def get_tactic(partial_seqs):
        if tf.size(partial_seqs) == 0:
            return tf.zeros(tf.shape(partial_seqs)[:2], tf.int32)
        else:
            return tf.reduce_max(partial_seqs[:, :, :1], axis=-1)
    
    def stopping_fn(
        self,
        partial_seqs: tf.Tensor,  # [batch, beam_size, seq_length] dtype: int
        pos: tf.Tensor,  # scalar dtype: int32,
        cache: dict[str, tf.Tensor],  # value shapes: [batch, beam_size, ...]
        static_data: dict[str, tf.Tensor | tf.RaggedTensor]
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:  # output shapes [batch, beam_size, vocab], dict of [batch, beam_size, ...]
        if pos == 0:
            is_finished = tf.zeros(shape=tf.shape(partial_seqs)[:2], dtype=bool)  # [batch, beam_size]
            return is_finished, cache
        else:
            arg_pos = tf.cast(pos - 1, tf.int32)
            # first token is the tactic id.  It may be a stop token but that is accounted for
            tactic_id = self.get_tactic(partial_seqs)  # [batch, beam_size]
            batch_ix = cache["batch_ix"]  # [batch, beam_size]    
            arg_lengths = static_data["arg_lengths"]  # [batch, tactic, arg, vocab]  
            indices = tf.stack([batch_ix, tactic_id], axis=-1)  # [batch, beam_size, 2]
            arg_lengths = tf.gather_nd(arg_lengths, indices)  # [batch, beam_size]
            
            is_finished = (arg_pos >= arg_lengths)
            return is_finished, cache

    def _select_best_results(
        self,
        tactics: tf.Tensor,  # [batch, tactics] dtype:int32
        arg_counts: tf.Tensor,  # [batch, tactics] dtype:int32
        tactic_logits: tf.Tensor,  # [batch, tactics]
        local_arg_logits: tf.RaggedTensor,  # [batch * tactics, None(args), global_cxt] 
        global_arg_logits: tf.RaggedTensor,  # [batch * tactics, None(args), local_cxt] 
        beam_width: tf.Tensor,  # int32
    ) -> tuple[
        tf.RaggedTensor,  # log_probs [batch, None(options)]
        tf.RaggedTensor,  # log_probs [batch, None(options), None(tactic+args), 2]
    ]:
        batch_size = tf.shape(tactics)[0]
        tactic_cnt = tf.shape(tactics)[1]

        # arg encoding step 1: combine the global and local_arguments together
        arg_logits = tf.concat([local_arg_logits, global_arg_logits], axis=-1)  # [batch*tactic, None(args), cxt]
        arg_logits = tf.ragged.map_flat_values(tf.math.log_softmax, arg_logits, axis=-1)
        local_arg_count = tf.shape(local_arg_logits)[-1]

        # arg encoding step 2: restrict to the top-k arguments
        # [batch-tactic-args, small_cxt], [batch-tactic-args, small_cxt]
        arg_logits_values, pre_topk_arg_indices = \
            tf.math.top_k(arg_logits.values, k=tf.minimum(beam_width, tf.shape(arg_logits)[-1]))
        pre_topk_arg_indices = arg_logits.with_values(pre_topk_arg_indices)  # [batch*tactic, None(args), small_cxt]
        arg_logits = arg_logits.with_values(arg_logits_values)  # [batch*tactic, None(args), small_cxt]
        
        # encode step 3: add padding for args and tactics
        # for args padding:
        # if valid argument postion, the probability is 0 (log prob = -inf)
        # [batch*tactic, None(args), 1+cxt]
        arg_logits = tf.ragged.map_flat_values(tf.pad, arg_logits, paddings=[[0, 0], [1, 0]], constant_values=-np.inf)
        # if invalid argument postion, the probability is 1 (log prob = 0)
        log_prob_pad_token = tf.math.log(tf.one_hot(0, tf.shape(arg_logits)[-1]))  # [1+cxt]
        # [batch*tactic, max(args), 1+cxt]
        arg_logits = arg_logits.to_tensor(default_value=log_prob_pad_token)
        # pad tactic dimension as well.  the probability is always 0 (log prob = -inf) for the tactic pad token
        # [batch, tactic, max(args), 1+cxt]
        arg_logits = tf.reshape(arg_logits, shape=[batch_size, tactic_cnt, tf.shape(arg_logits)[1], tf.shape(arg_logits)[2]])
        # [batch, 1+tactic, max(args), 1+cxt]
        arg_logits = tf.pad(arg_logits, paddings=[[0,0], [1, 0], [0,0], [0,0]], constant_values=-np.inf)
        # [batch, 1+tactic]
        tactic_logits = tf.pad(tactic_logits, paddings=[[0,0], [1, 0]], constant_values=-np.inf)
        # [batch, 1+tactic]
        arg_counts = tf.pad(arg_counts, paddings=[[0,0], [1, 0]], constant_values = 0)

        # encode 4: compute sequences where
        # - the tactic id (or padding) is in the first position
        # - arguments (or padding) are in later positions
        # [batch, beam_size, sequence_length], [batch, beam_size]
        sequences, log_probs = self.beam_search(
            initial_ids=tf.zeros([batch_size, 0], tf.int32),  # [batch, 0]
            initial_cache={"batch_ix": tf.range(batch_size)},  # [batch],
            static_data={
                "arg_logits": arg_logits, # [batch, 1+tactic, max(args), vocab]
                "arg_lengths": arg_counts,  # [batch, 1+tactic]
                "tactic_logits": tactic_logits,  # [batch, 1+tactics]
            },
            beam_size=beam_width,
            max_decode_length=tf.reduce_max(arg_counts) + 1  # +1 for the tactic
        )
        
        # decode all encoding steps in reverse

        # decode step 4 (make sequences)

        # make ragged to remove bad sequences
        # [batch, None(beam_size), sequence_length]
        sequences = tf.RaggedTensor.from_tensor(sequences, ragged_rank=1)
        log_probs = tf.RaggedTensor.from_tensor(log_probs, ragged_rank=1)  # [batch, None(beam)]

        # split tactics and args and record batch index
        tactic_token = tf.cast(sequences.values[:, 0], tf.int64)  # [batch-beam]
        arg_tokens = sequences.values[:, 1:]  # [batch-beam, max(args)]
        batch_ix = sequences.value_rowids()  # [batch-beam]

        # filter sequences for positive probability and a valid tactics
        is_good = (log_probs.values != -np.inf) & (tactic_token != 0)  # [batch-beam]
        batch_ix = tf.boolean_mask(batch_ix, is_good)  # [batch-beam]
        log_probs_values = tf.boolean_mask(log_probs.values, is_good)  # [batch-beam]
        tactic_token = tf.boolean_mask(tactic_token, is_good)  # [batch-beam]
        arg_tokens = tf.boolean_mask(arg_tokens, is_good)  # [batch-beam, max(args)]

        # filter arguments for trailing padding.
        batch_tactic_token_nd_ix = tf.stack([batch_ix, tactic_token], axis=-1)  # [batch-beam, 2]
        arg_lengths = tf.gather_nd(arg_counts, batch_tactic_token_nd_ix)  # [batch-beam]
        arg_tokens = tf.RaggedTensor.from_tensor(arg_tokens, lengths=arg_lengths, ragged_rank=1)  # [batch-beam, None(args)]

        # decode step 3 (add padding tokens)
        # note, we have already removed all padding tokens
        tf.assert_greater(arg_tokens.values, tf.constant(0, tf.int32))
        tf.assert_greater(tactic_token, tf.constant(0, tf.int64))
        # so we can shift the index
        tactic_ix = tactic_token - 1  # [batch-beam]
        arg_ix = arg_tokens - 1  # [batch-beam, None(args)]

        # decode step 2 (select top-k values)
        # we just have to reindex back into the pre-top-k index
        # args
        batch_tactic_ix = batch_ix * tf.cast(tactic_cnt, tf.int64) + tactic_ix  # [batch-beam]
        arg_indices = tf.gather(pre_topk_arg_indices, batch_tactic_ix)  # [batch-beam, None(args), small-cxt]
        arg_ix = tf.gather(arg_indices, arg_ix, batch_dims=2)  # [batch-beam, None(args)]
        
        # tactic
        batch_tactic_nd_ix = tf.stack([batch_ix, tactic_ix], axis=-1)  # [batch-beam, 2]
        final_tactic = tf.gather_nd(tactics, batch_tactic_nd_ix)  # [batch-beam]

        # decode step 1 (combine local and global)
        is_local = arg_ix < local_arg_count  # [batch-beam, None(args)]
        # 0 for local, 1 for global 
        arg_type = tf.where(is_local, 0, 1)  # [batch-beam, None(args)]
        arg_index = tf.where(is_local, arg_ix, arg_ix - local_arg_count)  # [batch-beam, None(args)]
        
        # return in desired form
        arg_decodings = tf.stack([arg_type, arg_index], axis=-1)  # [batch-beam, None(args), 2]
        tactic_decoding = tf.stack([final_tactic, final_tactic], axis=-1)  # [batch-beam, 2]
        tactic_decoding = tf.expand_dims(tactic_decoding, axis=1)  # [batch-beam, 1, 2]
        decodings = tf.concat([tactic_decoding, arg_decodings], axis=1)
        # [batch, None(beam), None(tactic+arg), 2]
        decodings = tf.RaggedTensor.from_value_rowids(
            values=decodings,  # [batch-beam, None(tactic+args), 2]
            value_rowids=batch_ix,  # [batch-beam]
            nrows=tf.cast(batch_size, tf.int64)
        )
        # [batch, None(beam)]
        log_probs = tf.RaggedTensor.from_value_rowids(
            values=log_probs_values,  # [batch-beam]
            value_rowids=batch_ix,  # [batch-beam]
            nrows=tf.cast(batch_size, tf.int64)
        )
        
        return log_probs, decodings
    
    @staticmethod
    def make_context_dense(
        logits: tf.RaggedTensor,  # [batch * top_k_tactics, None(args), None(cxt)]
        batch_size: tf.Tensor,  #int 
    ) -> tf.Tensor:  # [batch * top_k_tactics, None(args), max(cxt)]
        # the cxt size is a function of the batch dim, so if the batch size is one, we can optimize this step
        if batch_size == 1:
            # the size of the cxt is constant so we can use tf.shape to make it a tensor
            logits_values_size = tf.cast(tf.shape(logits.values)[0], tf.int64)
            cxt_size = tf.maximum(tf.reduce_max(logits.values.row_lengths()), 0)  # 0 is in case logits.values is empty
            # [1-top_k_tactics-args, cxt]
            logits_values = tf.reshape(logits.values.values, shape=[logits_values_size, cxt_size])
            # [1-top_k_tactics, None(args), cxt]
            logits = logits.with_values(
                logits_values
            )
        else:
            # [batch * top_k_tactics, None(args), cxt]    
            logits = logits.with_values(logits.values.to_tensor(default_value=-np.inf))
        return logits

    def _select_results(
        self,
        inference_output: dict[str, tf.Tensor | tf.RaggedTensor],
        search_expand_bound: tf.Tensor, # int32
    ):
        tactics = inference_output["tactic"]  # [batch, top_k_tactics]
        tactic_logits = inference_output["tactic_logits"]  # [batch, top_k_tactics]
        local_arguments_logits = inference_output["local_arguments_logits"]  # [batch * top_k_tactics, None(args), None(local_cxt)]
        global_arguments_logits = inference_output["global_arguments_logits"]  # [batch * top_k_tactics, None(args), None(global_cxt)]

        # find arg counts
        arg_counts = tf.gather(self.tactic_index_to_numargs, tactics) # [batch, top_k_tactics]
        
        # make last dimension dense
        # [batch * top_k_tactics, None(args), max(local_cxt)]             
        local_arguments_logits = self.make_context_dense(local_arguments_logits, batch_size=tf.shape(tactics)[0])
        # [batch * top_k_tactics, None(args), max(global_cxt)]
        global_arguments_logits = self.make_context_dense(global_arguments_logits, batch_size=tf.shape(tactics)[0])

        # [batch, None(results)], [batch, None(results), None(args)]
        log_probs, actions = self._select_best_results(
            tactics=tactics,
            arg_counts=arg_counts,
            tactic_logits=tactic_logits, 
            global_arg_logits=global_arguments_logits, 
            local_arg_logits=local_arguments_logits, 
            beam_width=search_expand_bound
        )
        return log_probs, actions


class Inference:
    """
    Container class for a single inference for a given proof-state.
    Subclasses should implement the following methods:
        - `numpy`: converts the inference into numpy format for Vasily's evaluation framework
        - `evaluate`: returns whether the inference is correct or not
    """
    value: float
    numpy: Callable[[], np.ndarray]
    evaluate: Callable[[int, tf.Tensor, tf.Tensor], bool]


@dataclass
class TacticInference(Inference):
    """
    Container class for a single base tactic inference for a given proof-state.
    """
    value: float
    tactic_id: int

    def numpy(self) -> np.ndarray:
        return np.array([[self.tactic_id, self.tactic_id]], dtype=np.uint32)

    def evaluate(self, tactic_id: int, local_arguments: tf.Tensor, global_arguments: tf.Tensor) -> bool:
        return tactic_id == self.tactic_id


@dataclass
class LocalArgumentInference(Inference):
    """
    Container class for a single base tactic and local arguments inference for a given proof-state.
    """
    value: float
    tactic_id: int
    local_arguments: tf.Tensor

    def numpy(self) -> np.ndarray:
        top_row = np.insert(np.zeros_like(self.local_arguments), 0, self.tactic_id)
        bottom_row = np.insert(self.local_arguments, 0, self.tactic_id)
        return np.stack([top_row, bottom_row], axis=-1).astype(np.uint32)

    def evaluate(self, tactic_id: int, local_arguments: tf.Tensor, global_arguments: tf.Tensor) -> bool:
        return tactic_id == self.tactic_id and tf.reduce_all(local_arguments == self.local_arguments)


@dataclass
class GlobalArgumentInference(Inference):
    """
    Container class for a single base tactic and local+global arguments inference for a given proof-state.
    """
    value: float
    tactic_id: int
    local_arguments: tf.Tensor
    global_arguments: tf.Tensor

    def numpy(self) -> np.ndarray:
        top_row = np.insert(np.where(self.global_arguments == -1, 0, 1), 0, self.tactic_id)
        bottom_row = np.insert(np.where(self.global_arguments == -1, self.local_arguments, self.global_arguments), 0, self.tactic_id)
        return np.stack([top_row, bottom_row], axis=-1).astype(np.uint32)

    def evaluate(self, tactic_id: int, local_arguments: tf.Tensor, global_arguments: tf.Tensor) -> bool:
        return tactic_id == self.tactic_id and tf.reduce_all(local_arguments == self.local_arguments) and tf.reduce_all(global_arguments == self.global_arguments)


@dataclass
class PredictOutput:
    """
    Container class for a list of predictions for a given proof-state.
    """
    state: Optional[LoaderProofstate]
    predictions: List[Inference]

    def p_total(self) -> float:
        """
        Computes the total probability captured by all the predictions for this proof-state.
        """
        return sum(np.exp(pred.value) for pred in self.predictions)

    def sort(self) -> None:
        """
        Sorts all the predictions in descending order according to their value (log of probability).
        """
        self.predictions.sort(key=lambda prediction: -prediction.value)

    def numpy(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Converts all predictions to numpy format for interaction with Vasily's evaluation framework.
        """
        self.sort()
        return [pred.numpy() for pred in self.predictions], np.array([np.exp(pred.value) for pred in self.predictions])

    def _evaluate(self,
                  tactic_id: int,
                  local_arguments: tf.Tensor,
                  global_arguments: tf.Tensor,
                  search_expand_bound: Optional[int] = None):
        self.sort()
        predictions = self.predictions[:search_expand_bound] if search_expand_bound is not None else self.predictions
        return any(inference.evaluate(tactic_id, local_arguments, global_arguments) for inference in predictions)

    def evaluate(self, action: LoaderAction, search_expand_bound: Optional[int] = None) -> bool:
        """
        Evaluate an action in the loader format
        """
        local_context_ids = self.state.context.local_context
        local_context_length = tf.shape(local_context_ids, out_type=tf.int64)[0]

        tactic_id = tf.cast(action.tactic_id, dtype=tf.int64)
        arguments_array = tf.cast(action.args, dtype=tf.int64)
        local_args = tf.cast(action.local_args, dtype=tf.int64)
        global_args = tf.cast(action.global_args, dtype=tf.int64)

        return self._evaluate(tactic_id, local_args, global_args, search_expand_bound=search_expand_bound)


class TFGNNPredict(Predict):
    def __init__(self,
                 log_dir: Path,
                 tactic_expand_bound: int,
                 debug_dir: Optional[Path] = None,
                 checkpoint_number: Optional[int] = None,
                 exclude_tactics: Optional[List[str]] = None,
                 allocation_reserve: float = 0.5,
                 numpy_output: bool = True,
                 ):
        """
        @param log_dir: the directory for the checkpoint that is to be loaded (as passed to the Trainer class)
        @param total_expand_bound:
        @param debug_dir: set to a directory to dump pickle files for every API call that is made
        @param checkpoint_number: the checkpoint number we want to load (use `None` for the latest checkpoint)
        @param exclude_tactics: a list of tactic names to exclude from all predictions
        @param allocation_reserve: proportional size of extra allocated space when resizing the nodes embedding array
        @param numpy_output: set to True to return the predictions as a tuple of numpy arrays (for evaluation purposes)
        """

        self._exporter = DataToTFGNN()
        self._allocation_reserve = allocation_reserve

        # create dummy dataset for pre-processing purposes
        graph_constants_filepath_yaml = log_dir / 'config' / 'graph_constants.yaml'
        graph_constants_filepath_json = log_dir / 'config' / 'graph_constants.json'
        if not graph_constants_filepath_json.exists():
            logger.info(f'no json graph_constants file, trying to load yaml')
            with graph_constants_filepath_yaml.open('r') as yml_file:
                graph_constants_d = yaml.load(yml_file, Loader=yaml.SafeLoader)
            if "global_context" in graph_constants_d:
                global_context = graph_constants_d["global_context"]
                assert global_context == list(range(len(global_context)))
                del graph_constants_d["global_context"]
            with graph_constants_filepath_json.open('w') as json_file:
                json.dump(graph_constants_d, json_file)
        else:
            logger.info(f'loading json graph_constants file')
            with graph_constants_filepath_json.open('r') as json_file:
                graph_constants_d = json.load(json_file)

        graph_constants_d['data_config'] = DataConfig(**graph_constants_d['data_config'])
        graph_constants = GraphConstants(**graph_constants_d)
            

        # call to parent constructor to defines self.graph_constants
        super().__init__(
            graph_constants=graph_constants,
            tactic_expand_bound=tactic_expand_bound,
            debug_dir=debug_dir,
        )

        # to build dummy proofstates we will need to use a tactic taking no arguments
        self._dummy_tactic_id = tf.argmin(graph_constants.tactic_index_to_numargs)  # num_arguments == 0

        # the decoding mechanism currently does not support tactics with more than NUMPY_NDIM_LIMIT
        self.fixed_tactic_mask = tf.constant(np.array(graph_constants.tactic_index_to_numargs) < NUMPY_NDIM_LIMIT)

        # mask tactics explicitly excluded from predictions
        if exclude_tactics is not None:
            exclude_tactics = set(exclude_tactics)
            self.fixed_tactic_mask &= tf.constant([(tactic_name not in exclude_tactics) for tactic_name in graph_constants.tactic_index_to_string])

        # create prediction task
        prediction_yaml_filepath = log_dir / 'config' / 'prediction.yaml'
        self.prediction_task = PredictionTask.from_yaml_config(graph_constants=graph_constants,
                                                               yaml_filepath=prediction_yaml_filepath)
        self.prediction_task_type = self.prediction_task.get_config()['prediction_task_type']

        # create definition task
        definition_yaml_filepath = log_dir / 'config' / 'definition.yaml'
        if definition_yaml_filepath.is_file():
            self.definition_task = DefinitionTask.from_yaml_config(graph_embedding=self.prediction_task.graph_embedding,
                                                                   gnn=self.prediction_task.gnn,
                                                                   yaml_filepath=definition_yaml_filepath)
        else:
            self.definition_task = None

        # load training checkpoint
        checkpoint = tf.train.Checkpoint(prediction_task=self.prediction_task.checkpoint)
        if self.definition_task is not None:
            checkpoint.definition_task = self.definition_task.get_checkpoint()

        checkpoints_path = log_dir / 'ckpt'
        available_checkpoints = {int(re.search('ckpt-(\d+).index', str(ckpt)).group(1)): ckpt.with_suffix('')
                                 for ckpt in checkpoints_path.glob('*.index')}
        if checkpoint_number is None:
            checkpoint_number = max(available_checkpoints.keys())
            logger.info(f'no checkpoint number specified, using latest available checkpoint #{checkpoint_number}')
        elif checkpoint_number not in available_checkpoints.keys():
            logger.error(f'checkpoint #{checkpoint_number} is not available')
            raise ValueError(f'checkpoint number {checkpoint_number} not found')

        try:
            load_status = checkpoint.restore(save_path=str(available_checkpoints[checkpoint_number]))
        except tf.errors.OpError as error:
            logger.error(f'unable to restore checkpoint #{checkpoint_number}!')
            raise error
        else:
            load_status.expect_partial().assert_nontrivial_match().assert_existing_objects_matched().run_restore_ops()
            logger.info(f'restored checkpoint #{checkpoint_number}!')

        node_label_num = self.graph_constants.node_label_num
        extra_label_num = round(self._allocation_reserve*node_label_num)
        if extra_label_num > 0: self._allocate_definitions(node_label_num + extra_label_num)
        
        self._select_best_results = Selection(
            tactic_index_to_numargs=self.graph_constants.tactic_index_to_numargs,
        ).select_results
        self._compile_network()

    def _allocate_definitions(self, new_node_label_num) -> None: # explicit change of the network array

        logger.info(f'extending global context from {self.graph_constants.node_label_num} to {new_node_label_num} elements')
        new_graph_emb_layer = self.prediction_task.graph_embedding.extend_embeddings(new_node_label_num)
        self.prediction_task.graph_embedding = new_graph_emb_layer

        if self.definition_task is not None:
            self.definition_task._graph_embedding = new_graph_emb_layer
        self.graph_constants.node_label_num = new_node_label_num
        self.prediction_task.global_arguments_logits.update_embedding_matrix(
            embedding_matrix=self.prediction_task.graph_embedding.get_node_embeddings()
        )

    @predict_api_debugging
    def allocate_definitions(self, new_node_label_num : int) -> None:
        if self.prediction_task_type != GLOBAL_ARGUMENT_PREDICTION:
            # no need to update anything if we are not going to use the global context
            return

        if new_node_label_num <= self.graph_constants.node_label_num:
            # already have sufficient array
            return

        new_node_label_num += round(self._allocation_reserve*new_node_label_num)

        self._allocate_definitions(new_node_label_num)
        self._compile_network()

    @predict_api_debugging
    def compute_new_definitions(self, new_cluster_subgraphs: List[LoaderDefinition]) -> None:
        if self.definition_task is None:
            raise RuntimeError('cannot update definitions when a definition task is not present')

        assert len(new_cluster_subgraphs) == 1
        self._compute_and_replace_definition_embs(new_cluster_subgraphs[0])

    @tf.function(input_signature = (LoaderProofstateSpec,))
    def _make_proofstate_graph_tensor(self, state : LoaderProofstate):
        action = LoaderAction(
            self._dummy_tactic_id,
            tf.zeros(shape=(0), dtype=tf.int64),
            tf.zeros(shape=(0), dtype=tf.int64),
        )
        graph_id = tf.constant(-1, dtype=tf.int64)
        x = DataToTFGNN.proofstate_to_graph_tensor(state, action, graph_id)
        return x

    def _compile_network(self):
        @tf.function(input_signature = (LoaderDefinitionSpec,))
        def compute_and_replace_definition_embs(loader_definition):
            graph_tensor = self._exporter.definition_to_graph_tensor(loader_definition)
            definition_graph = stack_graph_tensors([graph_tensor])

            scalar_definition_graph = definition_graph.merge_batch_to_components()
            definition_embeddings = self.definition_task(scalar_definition_graph).flat_values
            defined_labels = Trainer._get_defined_labels(definition_graph).flat_values

            self.prediction_task.graph_embedding.update_node_embeddings(
                embeddings=definition_embeddings,
                indices=defined_labels
            )
        self._compute_and_replace_definition_embs = compute_and_replace_definition_embs

        inference_model_bare = self.prediction_task.create_inference_model(
            tactic_expand_bound=self._tactic_expand_bound,
            graph_constants=self.graph_constants
        )
        allowed_model_tactics_spec = tf.TensorSpec(shape=(None,), dtype=tf.int32)
        @tf.function(input_signature = (LoaderProofstateSpec, allowed_model_tactics_spec))
        def inference_model(state, allowed_model_tactics):
            tactic_mask = tf.scatter_nd(
                indices = tf.expand_dims(allowed_model_tactics, axis = 1),
                updates = tf.ones_like(allowed_model_tactics, dtype=bool),
                shape = [self.graph_constants.tactic_num]
            )
            graph_tensor_single = self._make_proofstate_graph_tensor(state)
            graph_tensor_stacked = stack_graph_tensors([graph_tensor_single])
            inference_output = inference_model_bare({
                self.prediction_task.PROOFSTATE_GRAPH: graph_tensor_stacked,
                self.prediction_task.TACTIC_MASK: tf.expand_dims(tactic_mask, axis=0),
            })
            return inference_output
        self._inference_model = inference_model

        dummy_graph = LoaderGraph(
            nodes = np.zeros([1], dtype = int),
            edges = np.zeros([0,2], dtype = int),
            edge_labels = np.zeros([0], dtype = int),
            edge_offsets = np.zeros([0], dtype = int),
        )
        dummy_loader_proofstate = LoaderProofstate(
            graph = dummy_graph,
            root = 0,
            context = ProofstateContext(
                local_context = np.zeros([0], dtype = int),
                global_context = np.zeros([1], dtype = int), # make sure at least one element in global context
            ),
            metadata = ProofstateMetadata(
                name = "",
                step = 0,
                is_faithful = False,
            ),
        )
        
        # run on a dummy input to force precompilation
        output = self._inference_model(dummy_loader_proofstate, np.zeros([1], dtype = int))
        self._select_best_results(output, 1)
        
    # Currently not used
    def _make_proofstate_batch(self, datapoints : Iterable[LoaderProofstate]):
        return stack_graph_tensors([
            self._make_proofstate_graph_tensor(x)
            for x in datapoints
        ])

    @staticmethod
    def _logits_decoder(logits: tf.Tensor, total_expand_bound: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decoding mechanism
        """
        num_arguments = tf.shape(logits)[0]
        if num_arguments == 0:
            return np.array([[]], dtype=np.uint32), np.array([0], dtype=np.float32)

        logits = tf.math.log_softmax(logits).numpy()

        expand_bound = int(total_expand_bound**(1/num_arguments))

        sorted_indices = np.argsort(-logits).astype(dtype=np.uint32)
        restricted_indices = sorted_indices[:, :expand_bound]
        arg_combinations = cartesian_product(*restricted_indices)
        first_index = np.tile(np.arange(num_arguments), (arg_combinations.shape[0], 1))
        combination_values = np.sum(logits[first_index, arg_combinations], axis=1)
        return arg_combinations, combination_values

    @classmethod
    def _expand_arguments_logits(cls,
                                 total_expand_bound: int,
                                 num_arguments: int,
                                 local_context_size: int,
                                 global_context_size: int,
                                 tactic: tf.Tensor,
                                 tactic_logits: tf.Tensor,
                                 local_arguments_logits: Optional[tf.Tensor] = None,
                                 global_arguments_logits: Optional[tf.Tensor] = None
                                 ) -> List[Inference]:
        if local_arguments_logits is None:
            # this is a base tactic prediction
            return [TacticInference(value=float(tactic_logits.numpy()), tactic_id=int(tactic.numpy()))]
        elif global_arguments_logits is None:
            # this is a base tactic plus local arguments prediction
            logits = local_arguments_logits[:num_arguments, :local_context_size]
            arg_combinations, combination_values = cls._logits_decoder(logits=logits,
                                                                       total_expand_bound=total_expand_bound)
            combination_values += tactic_logits.numpy()
            return [LocalArgumentInference(value=float(value),
                                           tactic_id=int(tactic.numpy()),
                                           local_arguments=local_arguments)
                    for local_arguments, value in zip(tf.cast(arg_combinations, dtype=tf.int64), combination_values)]
        else:
            # this is a base tactic plus local and global arguments prediction
            combined_arguments_logits = tf.concat([global_arguments_logits[:num_arguments, :], local_arguments_logits[:num_arguments, :local_context_size]], axis=-1)
            arg_combinations, combination_values = cls._logits_decoder(logits=combined_arguments_logits,
                                                                       total_expand_bound=total_expand_bound)
            combination_values += tactic_logits.numpy()
            return [GlobalArgumentInference(value=float(value),
                                            tactic_id=int(tactic.numpy()),
                                            local_arguments=tf.where(arguments < global_context_size, -1, arguments - global_context_size),
                                            global_arguments=tf.where(arguments < global_context_size, arguments, -1))
                    for arguments, value in zip(tf.cast(arg_combinations, dtype=tf.int64), combination_values)]

    @predict_api_debugging
    def ranked_predictions(self,
                           state: LoaderProofstate,
                           total_expand_bound: int,
                           available_global: Optional[np.ndarray] = None,
                           allowed_model_tactics: Optional[Iterable[int]] = None
                           ) -> Union[PredictOutput, Tuple[np.ndarray, np.ndarray]]:
        """
        Produces predictions for a single proof-state.
        """
        if available_global is not None:
            raise NotImplementedError('available_global is not supported yet')

        search_expand_bound = total_expand_bound  # temporary co-opting this value
        inference_output = self._inference_model(state, allowed_model_tactics)
        best_log_probs, best_tactics = self._select_best_results(inference_output, search_expand_bound)
        return (best_tactics.numpy()[0], np.exp(best_log_probs.numpy()[0]))

        predict_output = PredictOutput(state=None, predictions=[])

        # go over the tactic_expand_bound batches
        for proofstate_batch_output in zip(*inference_output.values()):
            # go over the individual proofstates in a batch
            inference_data = {
                output_name: output_value[0]
                for output_name, output_value in zip(inference_output.keys(), proofstate_batch_output)
            }

            num_arguments = self.graph_constants.tactic_index_to_numargs[inference_data[TacticPrediction.TACTIC]]
            predictions = self._expand_arguments_logits(total_expand_bound=total_expand_bound,
                                                        num_arguments=num_arguments,
                                                        local_context_size=len(state.context.local_context),
                                                        global_context_size=len(state.context.global_context),
                                                        **inference_data)
            predict_output.predictions.extend(filter(lambda inference: inference.value > -float('inf'), predictions))

        # fill in the states in loader format
        predict_output.state = state

        # return predictions in the appropriate format
        return predict_output.numpy()

    # (!) NOT MAINTAINED
    def _batch_ranked_predictions(self,
                                  proofstate_graph: tfgnn.GraphTensor,
                                  tactic_expand_bound: int,
                                  total_expand_bound: int,
                                  tactic_mask: tf.Tensor
                                  ) -> List[PredictOutput]:

        raise Exception("Running unmaintained code. Delete this line at your own risk.")
        
        inference_model = self._inference_model(tactic_expand_bound)

        inference_output = inference_model({self.prediction_task.PROOFSTATE_GRAPH: proofstate_graph,
                                            self.prediction_task.TACTIC_MASK: tactic_mask})

        _, local_context_sizes = proofstate_graph.context['local_context_ids'].nested_row_lengths()

        batch_size = int(proofstate_graph.total_num_components.numpy())

        predict_outputs = [PredictOutput(state=None, predictions=[]) for _ in range(batch_size)]

        # go over the tactic_expand_bound batches
        for proofstate_batch_output in zip(*inference_output.values()):
            # go over the individual proofstates in a batch
            for predict_output, proofstate_output, local_context_size in zip(predict_outputs,
                                                                             zip(*proofstate_batch_output),
                                                                             local_context_sizes):
                inference_data = {output_name: output_value for output_name, output_value in zip(inference_output.keys(), proofstate_output)}
                num_arguments = self.graph_constants.tactic_index_to_numargs[inference_data[TacticPrediction.TACTIC]]
                predictions = self._expand_arguments_logits(total_expand_bound=total_expand_bound,
                                                            num_arguments=num_arguments,
                                                            local_context_size=local_context_size,
                                                            global_context_size=global_context_size,
                                                            **inference_data)
                predict_output.predictions.extend(filter(lambda inference: inference.value > -float('inf'), predictions))
        return predict_outputs

    
    # (!) NOT MAINTAINED
    def _evaluate(self,
                  proofstate_graph_dataset: tf.data.Dataset,
                  batch_size: int,
                  tactic_expand_bound: int,
                  total_expand_bound: int,
                  search_expand_bound: Optional[int] = None,
                  allowed_model_tactics: Optional[Iterable[int]] = None
                  ) -> Tuple[float, float]:

        raise Exception("Running unmaintained code. Delete this line at your own risk.")

        tactic_mask = self._tactic_mask_from_allowed_model_tactics(allowed_model_tactics)

        predictions = []
        tactic = []
        local_arguments = []
        global_arguments = []
        names = []
        for proofstate_graph in iter(proofstate_graph_dataset.batch(batch_size)):
            scalar_proofstate_graph = proofstate_graph.merge_batch_to_components()
            batch_tactic_mask = tf.repeat(tf.expand_dims(tactic_mask, axis=0), repeats=proofstate_graph.total_num_components, axis=0)
            batch_predict_output = self._batch_ranked_predictions(proofstate_graph=proofstate_graph,
                                                                  tactic_expand_bound=tactic_expand_bound,
                                                                  total_expand_bound=total_expand_bound,
                                                                  tactic_mask=batch_tactic_mask)
            predictions.extend(batch_predict_output)
            tactic.append(scalar_proofstate_graph.context['tactic'])
            local_arguments.append(scalar_proofstate_graph.context['local_arguments'])
            global_arguments.append(scalar_proofstate_graph.context['global_arguments'])
            names.append(scalar_proofstate_graph.context['name'])

        tactic = tf.concat(tactic, axis=0)
        local_arguments = tf.concat(local_arguments, axis=0)
        global_arguments = tf.concat(global_arguments, axis=0)
        names = tf.concat(names, axis=0).numpy()

        per_proofstate = []
        per_lemma = {}
        for action, name, predict_output in zip(zip(tactic, local_arguments, global_arguments), names, predictions):
            result = predict_output._evaluate(*action, search_expand_bound=search_expand_bound)
            per_proofstate.append(result)

            per_lemma[name] = (per_lemma.get(name, True) and result)
        per_proofstate_result = np.array(per_proofstate).mean()
        per_lemma_result = np.array(list(per_lemma.values())).mean()
        return per_proofstate_result, per_lemma_result
