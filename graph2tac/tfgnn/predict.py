from typing import Tuple, List, Union, Iterable, Callable, Optional, Dict

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
from graph2tac.tfgnn.tasks import PredictionTask, DefinitionTask, GLOBAL_ARGUMENT_PREDICTION
from graph2tac.tfgnn.train import Trainer
from graph2tac.common import logger
from graph2tac.predict import Predict, predict_api_debugging, NUMPY_NDIM_LIMIT
from graph2tac.tfgnn.stack_graph_tensors import stack_graph_tensors


class BeamSearch(tf.keras.layers.Layer):
    """Beam search layer

    :param token_log_prob_fn: A tensorflow function which returns log probabilities for all the tokens.
        It also takes as input a cache dictionary which is stores hidden information for each partial sequence
        which can be used to compute the log probabilities, and it returns an updated version.  (The beam search
        will use gather to reorder this cache after each beam search step.)  Another dictionary static_data
        is used to pass in other tensorflow tensorflow objects which can be used in the computation.
        Inputs:  - partial sequences: [batch, beam_size, seq_length]
                 - beam_search_step: int32
                 - cache: dict of [batch, beam_size, ...]
                 - static_data: dict of tensors of any shape
        Outputs: - token_log_probs: [batch, beam_size, vocabulary]
                 - updated_cache: dict of [batch, beam_size, ...]
    :param stopping_fn: A tensorflow function which returns whether a beam ray is finished.
        It takes inputs and outputs similar to token_log_prob_fn.
        Inputs:  - partial sequences: [batch, beam_size, seq_length]
                 - beam_search_step: int32
                 - cache: dict of [batch, beam_size, ...]
                 - static_data: dict of tensors of any shape
        Outputs: - is_finished: [batch, beam_size, vocabulary]
                 - updated_cache: dict of [batch, beam_size, ...]
    """
    def __init__(
        self, 
        token_log_prob_fn: Callable[
            [tf.Tensor, tf.Tensor, Dict[str, tf.Tensor], Dict[str, Union[tf.Tensor, tf.RaggedTensor]]],
            Tuple[tf.Tensor, Dict[str, tf.Tensor]]
        ], 
        stopping_fn: Callable[
            [tf.Tensor, tf.Tensor, Dict[str, tf.Tensor], Dict[str, Union[tf.Tensor, tf.RaggedTensor]]],
            Tuple[tf.Tensor, Dict[str, tf.Tensor]]
        ],
    ):
        super().__init__()
        self.token_log_prob_fn = token_log_prob_fn
        self.stopping_fn = stopping_fn
    
    def one_beam_step(
        self,
        ids: tf.Tensor,  # [batch_size, beam_size0, seq_length]
        scores: tf.Tensor,  # [batch_size, beam_size0]
        cache: Dict[str, tf.Tensor],  # dict with values: [batch_size, beam_size0, ...]
        static_data: Dict[str, Union[tf.Tensor, tf.RaggedTensor]],
        i: tf.Tensor,  # int32
        max_beam_size: tf.Tensor,  # int32
    ) -> Tuple[
        tf.Tensor,  # ids: [batch_size, beam_size, seq_length+1]
        tf.Tensor,  # scores: [batch_size, beam_size]
        Dict[str, tf.Tensor],  # cache with values: [batch_size, beam_size, ...]
    ]:
        batch_size = tf.shape(ids)[0]
        beam_size0 = tf.shape(ids)[1]

        # [batch, beam_size0, vocabulary], dict of [batch_size, beam_size0, ...]
        token_scores, cache = self.token_log_prob_fn(ids, i, cache, static_data)
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
        cache: Dict[str, tf.Tensor],  # dict with values: [batch_size, beam_size0, ...]
        static_data: Dict[str, Union[tf.Tensor, tf.RaggedTensor]],
        i: tf.Tensor,  # int32
        max_beam_size: tf.Tensor,  # int32                           
    ) -> Tuple[
        tf.Tensor,  # ids: [batch_size, beam_size, seq_length+1]
        tf.Tensor,  # scores: [batch_size, beam_size]
        Dict[str, tf.Tensor],  # cache with values: [batch_size, beam_size, ...]
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
        initial_cache: Dict[str, tf.Tensor],  # [batch_size, ...]
        static_data: Dict[str, Union[tf.Tensor, tf.RaggedTensor]],
        beam_size: tf.Tensor,  # int32
        max_decode_length: tf.Tensor,  # int32
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(initial_ids)[0]

        # start with a beam_size of 1 for the input to the first step first step
        ids = tf.expand_dims(initial_ids, axis=1)  # [batch_size, beam_size0, initial_seq_length]
        cache = {
            k: tf.expand_dims(v, axis=1)  # [batch_size, beam_size0, ...]
            for k, v in initial_cache.items()  # v: [batch_size, ...]
        }
        scores = tf.expand_dims(tf.zeros(shape=[batch_size]), axis=-1)  # [batch_size, beam_size0]

        # iterate over the sequence
        is_finished = False
        # this loop is automatically converted into tf.while_loop
        shape_invariants = [
            (ids, tf.TensorShape([None, None, None])),
            (scores, tf.TensorShape([None, None])),
            (cache, {k: tf.TensorShape([None, None] + [None for i in range(len(tf.shape(v)) - 2)]) for k,v in cache.items()}),
        ]
        for i in range(max_decode_length):
            # the sizes of beam_size and seq_length can change during the loop and we need to let TF know this
            tf.autograph.experimental.set_loop_options(
                shape_invariants=shape_invariants
            )
            # ids: [batch_size, beam_size, initial_seq_length]
            # scores: [batch_size, beam_size]
            # cache: dict with values [batch_size, beam_size, ...]
            is_finished, (ids, scores, cache) = self.one_beam_step_with_early_stopping(is_finished, ids, scores, cache, static_data, i, beam_size)

        return ids, scores


class SelectBestResults(tf.keras.layers.Layer):
    """Select the best results from model.

    Returns best results given the model outputs.
    It takes as input the inference_output of the base inference model.
    It returns two things:
    - log_probabilities: the log probabilities of the returned results, shape [results]
    - results as a ragged tensor of shape [results, (sequence_length), 2]
      the sequence includes both the tactic id as the first element and the arguments
      the tactic is encoded as [tactic_id, tactic_id]
      the arguments are encoded as either [0, local_context_id] or [1, global_context_id]

    It may return less than search_expand_bound number of results,
    since it filters out results with probability 0.

    :param tactic_index_to_numargs: list tactic arg lengths for each tactic
    :param search_expand_bound: maximum number of results to return
    """
    def __init__(self, tactic_index_to_numargs: List[int], search_expand_bound: int):
        super().__init__()
        self.tactic_index_to_numargs = tf.cast(tf.constant(tactic_index_to_numargs), tf.int32)
        self.search_expand_bound = search_expand_bound

        self.beam_search = BeamSearch(
            token_log_prob_fn=self.token_log_prob_fn,
            stopping_fn=self.stopping_fn,
        )

    def token_log_prob_fn(
        self,
        partial_seqs: tf.Tensor,  # [batch, beam_size, seq_length] dtype: int
        pos: tf.Tensor,  # scalar dtype: int32,
        cache: Dict[str, tf.Tensor],  # value shapes: [batch, beam_size, ...]
        static_data: Dict[str, Union[tf.Tensor, tf.RaggedTensor]],
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:  # output shapes [batch, beam_size, vocab], dict of [batch, beam_size, ...]
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
        cache: Dict[str, tf.Tensor],  # value shapes: [batch, beam_size, ...]
        static_data: Dict[str, Union[tf.Tensor, tf.RaggedTensor]],
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:  # output shapes [batch, beam_size], dict of [batch, beam_size, ...]
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

    def _select_best_results_search(
        self,
        tactic_logits: tf.Tensor,  # [batch, tactics]
        tactic_arg_counts: tf.Tensor,  # [batch, tactics] dtype:int32
        arg_logits: tf.RaggedTensor,  # [batch * tactics, None(args), small_cxt] 
        beam_width: tf.Tensor,  # int32
    ) -> Tuple[
        tf.Tensor,  # tactic_token [batch-beam]
        tf.RaggedTensor,  # arg_token [batch-beam, None(args)]
        tf.Tensor,  # log_probs [batch-beam]
        tf.Tensor,  # batch_ix [batch-beam]
    ]:
        batch_size = tf.shape(tactic_logits)[0]
        
        # encode: compute sequences where
        # - the tactic id (or padding) is in the first position
        # - arguments (or padding) are in later positions
        # [batch, beam_size, sequence_length], [batch, beam_size]
        sequences, log_probs = self.beam_search(
            initial_ids=tf.zeros([batch_size, 0], tf.int32),  # [batch, 0]
            initial_cache={"batch_ix": tf.range(batch_size)},  # [batch],
            static_data={
                "arg_logits": arg_logits, # [batch, 1+tactic, max(args), vocab]
                "arg_lengths": tactic_arg_counts,  # [batch, 1+tactic]
                "tactic_logits": tactic_logits,  # [batch, 1+tactics]
            },
            beam_size=beam_width,
            max_decode_length=tf.reduce_max(tactic_arg_counts) + 1  # +1 for the tactic
        )
        
        # decode

        # make ragged to remove bad sequences
        # [batch, None(beam_size), sequence_length]
        sequences = tf.RaggedTensor.from_tensor(sequences, ragged_rank=1)
        log_probs = tf.RaggedTensor.from_tensor(log_probs, ragged_rank=1)  # [batch, None(beam)]

        # split tactics and args and record batch index
        tactic_token = tf.cast(sequences.values[:, 0], tf.int64)  # [batch-beam]
        arg_tokens = sequences.values[:, 1:]  # [batch-beam, max(args)]
        batch_ix = sequences.value_rowids()  # [batch-beam]

        # filter sequences for positive probability and a valid tactics
        is_good = (log_probs.values != -np.inf)  # [batch-beam]
        batch_ix = tf.boolean_mask(batch_ix, is_good)  # [batch-beam]
        log_probs = tf.boolean_mask(log_probs.values, is_good)  # [batch-beam]
        tactic_token = tf.boolean_mask(tactic_token, is_good)  # [batch-beam]
        arg_tokens = tf.boolean_mask(arg_tokens, is_good)  # [batch-beam, max(args)]

        # filter arguments for trailing padding.
        batch_tactic_token_nd_ix = tf.stack([batch_ix, tactic_token], axis=-1)  # [batch-beam, 2]
        arg_lengths = tf.gather_nd(tactic_arg_counts, batch_tactic_token_nd_ix)  # [batch-beam]
        arg_tokens = tf.RaggedTensor.from_tensor(arg_tokens, lengths=arg_lengths, ragged_rank=1)  # [batch-beam, None(args)]
        
        return tactic_token, arg_tokens, log_probs, batch_ix
    
    def _select_best_results_padding(
        self,
        tactic_logits: tf.Tensor,  # [batch, tactics]
        tactic_arg_counts: tf.Tensor,  # [batch, tactics] dtype:int32
        arg_logits: tf.RaggedTensor,  # [batch * tactics, None(args), small_cxt] 
        beam_width: tf.Tensor,  # int32
    ) -> Tuple[
        tf.Tensor,  # tactic [batch-beam]
        tf.RaggedTensor,  # arg_ix [batch, None(args)]
        tf.Tensor,  # log_probs [batch-beam]
        tf.Tensor,  # batch_ix [batch-beam]
    ]:
        batch_size = tf.shape(tactic_logits)[0]
        tactic_cnt = tf.shape(tactic_logits)[1]

        PAD = 0
        
        # encode: add padding for args and tactics
        # for args padding:
        # if valid argument postion, the probability is 0 (log prob = -inf)
        # [batch*tactic, None(args), 1+cxt]
        arg_logits = tf.ragged.map_flat_values(tf.pad, arg_logits, paddings=[[0, 0], [1, 0]], constant_values=-np.inf)
        # if invalid argument postion, the probability is 1 (log prob = 0)
        log_prob_pad_token = tf.math.log(tf.one_hot(PAD, tf.shape(arg_logits)[-1]))  # [1+cxt]
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
        tactic_arg_counts = tf.pad(tactic_arg_counts, paddings=[[0,0], [1, 0]], constant_values = 0)

        # calculate results
        # [batch-beam], [batch-beam], [batch-beam], [batch-beam, None(args)]
        tactic_token, arg_tokens, log_probs, batch_ix = self._select_best_results_search(
            tactic_logits=tactic_logits,
            tactic_arg_counts=tactic_arg_counts,
            arg_logits=arg_logits,
            beam_width=beam_width
        )

        # decode
        # note, we have already removed all padding tokens
        tf.assert_greater(arg_tokens.values, tf.constant(PAD, tf.int32))
        tf.assert_greater(tactic_token, tf.constant(PAD, tf.int64))
        # so we can shift the index
        tactic_ix = tactic_token - 1  # [batch-beam]
        arg_ix = arg_tokens - 1  # [batch-beam, None(args)]
        
        return tactic_ix, arg_ix, log_probs, batch_ix
    
    def _select_best_results_top_k(
        self,
        tactic_logits: tf.Tensor,  # [batch, tactics]
        tactics: tf.Tensor,  # [batch, tactics] dtype:int32
        tactic_arg_counts: tf.Tensor,  # [batch, tactics] dtype:int32
        arg_logits: tf.RaggedTensor,  # [batch * tactics, None(args), cxt] 
        beam_width: tf.Tensor,  # int32
    ) -> Tuple[
        tf.Tensor,  # tactic [batch-beam]
        tf.RaggedTensor,  # arg_ix [batch, None(args)]
        tf.Tensor,  # log_probs [batch-beam]
        tf.Tensor,  # batch_ix [batch-beam]
    ]:
        tactic_cnt = tf.shape(tactics)[1]

        # tactics are already restricted to top-k

        # arg encoding step: restrict to the top-k arguments
        # [batch-tactic-args, small_cxt], [batch-tactic-args, small_cxt]
        arg_logits_values, pre_topk_arg_indices = \
            tf.math.top_k(arg_logits.values, k=tf.minimum(beam_width, tf.shape(arg_logits)[-1]), sorted=False)
        pre_topk_arg_indices = arg_logits.with_values(pre_topk_arg_indices)  # [batch*tactic, None(args), small_cxt]
        arg_logits = arg_logits.with_values(arg_logits_values)  # [batch*tactic, None(args), small_cxt]

        # calculate results
        # tactic_ix and arg_ix are indices into tactic and pre_topk_arg_indices
        # [batch-beam], [batch-beam, None(args)], [batch-beam], [batch-beam]
        tactic_ix, arg_ix, log_probs, batch_ix  = self._select_best_results_padding(
            tactic_logits=tactic_logits,
            tactic_arg_counts=tactic_arg_counts,
            arg_logits=arg_logits,
            beam_width=beam_width
        )

        # decode
        # args (reindex back into the pre-top-k index)
        batch_tactic_ix = batch_ix * tf.cast(tactic_cnt, tf.int64) + tactic_ix  # [batch-beam]
        arg_indices = tf.gather(pre_topk_arg_indices, batch_tactic_ix)  # [batch-beam, None(args), small-cxt]
        arg_ix = tf.gather(arg_indices, arg_ix, batch_dims=2)  # [batch-beam, None(args)]
        
        # tactic (reindex back into tactics)
        batch_tactic_nd_ix = tf.stack([batch_ix, tactic_ix], axis=-1)  # [batch-beam, 2]
        final_tactic = tf.gather_nd(tactics, batch_tactic_nd_ix)  # [batch-beam]
        
        return final_tactic, arg_ix, log_probs, batch_ix

    def _select_best_results_local_global(
        self,
        tactic_logits: tf.Tensor,  # [batch, tactics]
        tactics: tf.Tensor,  # [batch, tactics] dtype:int32
        tactic_arg_counts: tf.Tensor,  # [batch, tactics] dtype:int32
        local_arg_logits: tf.RaggedTensor,  # [batch * tactics, None(args), global_cxt]
        global_arg_logits: tf.RaggedTensor,  # [batch * tactics, None(args), local_cxt]
        beam_width: tf.Tensor,  # int32
    ) -> Tuple[
        tf.RaggedTensor,  # log_probs [batch, None(options)]
        tf.RaggedTensor,  # log_probs [batch, None(options), None(tactic+args), 2]
    ]:
        # encode: combine the global and local_arguments together
        arg_logits = tf.concat([local_arg_logits, global_arg_logits], axis=-1)  # [batch*tactic, None(args), cxt]
        arg_logits = tf.ragged.map_flat_values(tf.math.log_softmax, arg_logits, axis=-1)  # [batch*tactic, None(args), cxt]
        global_arg_offset = tf.shape(local_arg_logits)[-1]

        # calculate results
        # [batch-beam], [batch-beam, None(args)], [batch-beam], [batch-beam]
        tactic, arg_ix, log_probs, batch_ix  = self._select_best_results_top_k(
            tactic_logits=tactic_logits,
            tactics=tactics,
            tactic_arg_counts=tactic_arg_counts,
            arg_logits=arg_logits,
            beam_width=beam_width
        )

        # decode
        is_local = arg_ix < global_arg_offset  # [batch-beam, None(args)]
        # 0 for local, 1 for global 
        arg_type = tf.where(is_local, 0, 1)  # [batch-beam, None(args)]
        arg_index = tf.where(is_local, arg_ix, arg_ix - global_arg_offset)  # [batch-beam, None(args)]

        return tactic, arg_type, arg_index, log_probs, batch_ix

    @staticmethod
    def make_context_dense(
        logits: tf.RaggedTensor,  # [batch * top_k_tactics, None(args), None(cxt)]
        batch_size: tf.Tensor,  #int 
    ) -> tf.Tensor:  # [batch * top_k_tactics, None(args), max(cxt)]
        # the cxt size is a function of the batch dim, so if the batch size is one, we can optimize this step
        if batch_size == 1:
            # the size of the cxt is constant so we can use tf.shape to make it a tensor
            logits_values_size = tf.cast(tf.shape(logits.values)[0], tf.int64)
            cxt_size = tf.maximum(tf.reduce_max(logits.values.row_lengths()), 0)
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

    def call(
        self,
        inference_output: Dict[str, Union[tf.Tensor, tf.RaggedTensor]],
    ):
        tactics = inference_output["tactic"]  # [batch, top_k_tactics]
        tactic_logits = inference_output["tactic_logits"]  # [batch, top_k_tactics]
        local_arguments_logits = inference_output["local_arguments_logits"]  # [batch * top_k_tactics, None(args), None(local_cxt)]
        global_arguments_logits = inference_output["global_arguments_logits"]  # [batch * top_k_tactics, None(args), None(global_cxt)]

        # find arg counts
        tactic_arg_counts = tf.gather(self.tactic_index_to_numargs, tactics) # [batch, top_k_tactics]
        
        # make last dimension dense
        batch_size = tf.shape(tactics)[0]
        # [batch * top_k_tactics, None(args), max(local_cxt)]             
        local_arguments_logits = self.make_context_dense(local_arguments_logits, batch_size=batch_size)
        # [batch * top_k_tactics, None(args), max(global_cxt)]
        global_arguments_logits = self.make_context_dense(global_arguments_logits, batch_size=batch_size)

        # [batch-beam], [batch-beam, None(args)], [batch-beam, None(args)], [batch-beam]
        tactic, arg_type, arg_index, log_probs, batch_ix = self._select_best_results_local_global(
            tactic_logits=tactic_logits, 
            tactics=tactics,
            tactic_arg_counts=tactic_arg_counts,
            global_arg_logits=global_arguments_logits, 
            local_arg_logits=local_arguments_logits, 
            beam_width=self.search_expand_bound
        )

        # return in desired form
        arg_decodings = tf.stack([arg_type, arg_index], axis=-1)  # [batch-beam, None(args), 2]
        tactic_decoding = tf.stack([tactic, tactic], axis=-1)  # [batch-beam, 2]
        tactic_decoding = tf.expand_dims(tactic_decoding, axis=1)  # [batch-beam, 1, 2]
        action_values = tf.concat([tactic_decoding, arg_decodings], axis=1)  # [batch-beam, None(1+args), 2]

        # [batch, None(beam), None(tactic+args), 2]
        actions = tf.RaggedTensor.from_value_rowids(
            values=action_values,  # [batch-beam, None(tactic+args), 2]
            value_rowids=batch_ix,  # [batch-beam]
            nrows=tf.cast(batch_size, tf.int64)
        )
        # [batch, None(beam)]
        log_probs = tf.RaggedTensor.from_value_rowids(
            values=log_probs,  # [batch-beam]
            value_rowids=batch_ix,  # [batch-beam]
            nrows=tf.cast(batch_size, tf.int64)
        )
        return log_probs, actions


class TFGNNPredict(Predict):
    def __init__(self,
                 log_dir: Path,
                 tactic_expand_bound: int,
                 search_expand_bound: int,
                 debug_dir: Optional[Path] = None,
                 checkpoint_number: Optional[int] = None,
                 exclude_tactics: Optional[List[str]] = None,
                 allocation_reserve: float = 0.5,
                 numpy_output: bool = True,
                 ):
        """
        @param log_dir: the directory for the checkpoint that is to be loaded (as passed to the Trainer class)
        @param tactic_expand_bound: the number of top base tactics to consider
        @param search_expand_bound: the max number of results to return
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
            search_expand_bound=search_expand_bound,
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

        # create task to select best results from prediction task
        self.select_best_results_task = SelectBestResults(
            tactic_index_to_numargs=self.graph_constants.tactic_index_to_numargs,
            search_expand_bound=self._search_expand_bound
        )

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
        available_checkpoints = {int(re.search(r'ckpt-(\d+).index', str(ckpt)).group(1)): ckpt.with_suffix('')
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
            return self.select_best_results_task(inference_output)
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
        self._inference_model(dummy_loader_proofstate, np.zeros([1], dtype = int))
        
    # Currently not used
    def _make_proofstate_batch(self, datapoints : Iterable[LoaderProofstate]):
        return stack_graph_tensors([
            self._make_proofstate_graph_tensor(x)
            for x in datapoints
        ])

    @predict_api_debugging
    def ranked_predictions(self,
                           state: LoaderProofstate,
                           available_global: Optional[np.ndarray] = None,
                           allowed_model_tactics: Optional[Iterable[int]] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:  # 
        """
        Produces predictions for a single proof-state.
        """
        if available_global is not None:
            raise NotImplementedError('available_global is not supported yet')

        best_log_probs, best_tactics = self._inference_model(state, allowed_model_tactics)
        return (best_tactics.numpy()[0], best_log_probs.numpy()[0])

    
    # (!) NOT MAINTAINED
    def _evaluate(self,
                  proofstate_graph_dataset: tf.data.Dataset,
                  batch_size: int,
                  tactic_expand_bound: int,
                  total_expand_bound: int,
                  search_expand_bound: Optional[int] = None,
                  allowed_model_tactics: Optional[Iterable[int]] = None
                  ) -> Tuple[float, float]:  # per_proofstate_passrate, per_lemma_passrate

        raise NotImplemented("Running unmaintained code. Delete this line at your own risk.")
