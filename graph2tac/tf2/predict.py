from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import tensorflow as tf
from graph2tac.tf2.datatypes import RaggedPair
from graph2tac.tf2.graph_nn_batch import make_flat_batch_np
from graph2tac.tf2.graph_nn_def_batch import make_flat_def_batch_np
from graph2tac.tf2.model import ModelWrapper, np_to_tensor, np_to_tensor_def
from graph2tac.tf2.segments import segment_lens
import time

from graph2tac.tf2.graph_nn_batch import make_flat_batch_np_empty

NUMPY_NDIM_LIMIT = 32

def cartesian_product(*arrays):
    """
    using the code from  https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    """
    la = len(arrays)

    if la > 32:
        print(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


from typing import Optional

class Predict:
    """
    This is a simple predict class.  It's just a first draft.

    Initialize with the directory containing the checkpoint files.
    This is usually not the weights directory, but a subdirectory corresponding
    to the particular epoch.
    """
    def __init__(self, checkpoint_dir: Path):
        """
        Public API
        """
        self.checkpoint_dir = checkpoint_dir
        self.params = ModelWrapper.get_params_from_checkpoint(checkpoint_dir)
        self.dataset_consts = self.params.dataset_consts
        assert self.dataset_consts is not None
        self.n_predict = 0
        self.t_predict = 0

    def get_tactic_index_to_numargs(self) -> Iterable:
        """
        Public API
        """
        return self.dataset_consts.tactic_index_to_numargs

    def get_tactic_index_to_hash(self) -> Iterable:
        """
        Public API
        """
        return self.dataset_consts.tactic_index_to_hash

    def get_node_label_to_name(self) -> Iterable:
        """
        Public API
        """
        return self.dataset_consts.node_label_to_name

    def get_node_label_in_spine(self) -> Iterable:
        """
        Public API
        """
        return self.dataset_consts.node_label_in_spine


    def initialize(self, global_context: Optional[list[int]] = None):

        self.params = ModelWrapper.get_params_from_checkpoint(self.checkpoint_dir)
        self.dataset_consts = self.params.dataset_consts
        assert self.dataset_consts is not None

        if global_context is not None:
            self.dataset_consts.global_context = global_context
        self.model_wrapper = ModelWrapper(checkpoint=self.checkpoint_dir, params=self.params)
        if global_context is not None:
            node_label_num = max(global_context)+1
            if node_label_num > self.dataset_consts.node_label_num:
                self.model_wrapper.extend_embedding_table(node_label_num)
        max_args = self.dataset_consts.tactic_max_arg_num
        self.dummy_action = (0,
                             np.zeros([max_args, 2]),
                             # np.array([False] * max_args),
        )
        self.dummy_id = 0
        self.pred_fn = self.model_wrapper.get_predict_fn()
        self.set_def_fn = self.model_wrapper.get_set_def_fn()
        self.tactic_index_to_numargs = np.array(self.dataset_consts.tactic_index_to_numargs)

        flat_batch_np = make_flat_batch_np_empty(self.dataset_consts)
        flat_batch = np_to_tensor(flat_batch_np)
        result = self.pred_fn(flat_batch)




    def compute_new_definitions(self, new_cluster_subgraphs : list) -> None:
        """
        Public API. The client is supposed to call this method for a sequence of topologically sorted valid roots of
        cluster definitions. For simplicity, the client can always call this method with single-element batches in
        topological order.

        @param new_cluster_subgraphs: a list of cluster subgraphs that can be batched together (no mutual dependencies)
        """
        flat_batch_np = make_flat_def_batch_np(new_cluster_subgraphs)
        flat_batch = np_to_tensor_def(flat_batch_np)
        self.set_def_fn(flat_batch)

    def predict_tactic_logits(self, state: tuple) -> np.ndarray:  # [tactics]
        """
        Return logits for all possible tactics.

        This are unnormalized.  To turn them into a probability distribution use softmax.

        The output has one dimension: the number of tactics.
        """
        batch = [(state, self.dummy_action, self.dummy_id)]  # dummy action and dummy_id isn't used to get tactic logits
        model_input = np_to_tensor(make_flat_batch_np(batch, len(self.dataset_consts.global_context),
                                                      self.dataset_consts.tactic_max_arg_num))
        tactic_logits, _, _ = self.pred_fn(model_input)  # [bs, tactics], _, _
        return tactic_logits[0].numpy()

    def predict_arg_logits(self, state: tuple, tactic_id: int) -> np.ndarray:  # [args, cxt]
        """
        Return logits for all possible elements of local context and all argument positions.

        This are unnormalized.  To turn them into a probability distribution use softmax.
        The None option is removed.

        The output matrix has two dimensions:
        - first dimension is the num args for that tactic
        - second dimension is the num of elements in the cxt
        """
        action = (tactic_id, self.dummy_action[1], self.dummy_action[2])
        batch = [(state, action, self.dummy_id)]  # only action tactic is used to get arg logits
        cxt_len = len(state[3])  # this is how many local context elements there are
        model_input = np_to_tensor(make_flat_batch_np(batch, len(self.dataset_consts.global_context),
                                                      self.dataset_consts.tactic_max_arg_num))
        _, arg_nums, arg_logits = self.pred_fn(model_input)  # [bs], [total_args, (local_global_ctx)] (ragged)
        arg_cnt = int(arg_nums[0])
        local_context_arg_cnt = arg_cnt * cxt_len  # how many logits there are for local context elements in all arguments
        return tf.reshape(arg_logits.values[:local_context_arg_cnt], [arg_cnt, cxt_len])

    def predict_tactic_arg_logits(self, state: tuple, tactic_expand_bound, allowed_model_tactics: list, available_global):
        """
        returns a matrix of tactic / arg logits expanding arguments call with top_tactics
        """
        tactic_logits = self.predict_tactic_logits(state)
        pre_top_tactic_ids = np.argsort(-tactic_logits)  # [:tactic_expand_bound]
        top_tactic_ids = pre_top_tactic_ids[np.isin(pre_top_tactic_ids, allowed_model_tactics)][:tactic_expand_bound]
        top_tactic_ids = top_tactic_ids[self.tactic_index_to_numargs[top_tactic_ids] < NUMPY_NDIM_LIMIT]
        batch = [(state, (tactic_id, self.dummy_action[1]), self.dummy_id) for tactic_id in top_tactic_ids]
        if not batch:
            return top_tactic_ids, tactic_logits[top_tactic_ids], [], []
        model_input = np_to_tensor(make_flat_batch_np(batch, len(self.dataset_consts.global_context),
                                                      self.dataset_consts.tactic_max_arg_num))
        _, arg_nums, arg_logits = self.pred_fn(model_input)
        cxt_len = len(state[3])  # this is how many local context elements there are
        arg_cnt = sum(arg_nums)
        local_context_arg_cnt = arg_cnt * cxt_len  # how many logits there are for local context elements in all arguments
        global_context_start = local_context_arg_cnt + arg_cnt
        local_arg_logits = tf.reshape(arg_logits.values[:local_context_arg_cnt], [arg_cnt, cxt_len])  # [all_args, cxt]
        gctx_len = len(self.dataset_consts.global_context)
        global_arg_logits = tf.reshape(arg_logits.values[global_context_start:], [arg_cnt, gctx_len])  # [all_args, gcxt]
        if available_global is not None:
            available_global = tf.constant(available_global, dtype=tf.int32)
            global_arg_logits = tf.transpose(tf.gather(tf.transpose(global_arg_logits), available_global))
        arg_logits = tf.concat([local_arg_logits, global_arg_logits], axis = 1)
        return top_tactic_ids, tactic_logits[top_tactic_ids], arg_nums, arg_logits


    def ranked_predictions(self, state: tuple, allowed_model_tactics: list, available_global=None, tactic_expand_bound=20, total_expand_bound=1000000):
        """
        available_global is np.array of indices into global_context
        """
        t0 = time.time()

        context_len = len(state[3])
        global_context = np.arange(len(self.dataset_consts.global_context), dtype = np.uint32)
        if available_global is not None:
            if len(available_global) > 0 and (available_global >= len(self.dataset_consts.global_context)).any():
                print("Warning: ignoring new definitions!")
                available_global = available_global[available_global < len(self.dataset_consts.global_context)]
            global_context = global_context[available_global]
        if (state[0] >= self.dataset_consts.node_label_num).any():
            #print("Warning: ignoring new node types!")
            nodes_c = state[0]
            mask = (nodes_c >= self.dataset_consts.node_label_num)
            nodes_c[mask] = self.dataset_consts.node_label_num-1
        top_tactic_ids, tactic_logits, arg_nums, arg_logits = self.predict_tactic_arg_logits(state, tactic_expand_bound, allowed_model_tactics, available_global)
        tactic_probs  = tf.nn.softmax(tactic_logits).numpy()
        result_idx = []
        result_value = []
        logit_idx_to_action = np.concatenate([
            np.stack( # local args
                [np.zeros(context_len, dtype = np.uint32),
                 np.arange(context_len, dtype = np.uint32)],
                axis = 1
            ),
            np.stack( # global args
                [np.ones_like(global_context, dtype = np.uint32),
                 np.array(global_context, dtype = np.uint32)],
                axis = 1
            )
        ])
        first_arg_id = 0  # the position of the first argument for each tactic
        for tactic_idx in range(len(top_tactic_ids)):
            num_args_for_tactic = arg_nums[tactic_idx]
            # the first dimension of arg logits goes:
            # tac0-arg0, ... tac0-argn0, tac1-arg0, ..., tac1-argn1, ..., tacm-arg0, ... tac1-argnm
            logits_at_tactic = arg_logits[first_arg_id: first_arg_id + num_args_for_tactic]  # [args, cxt]
            per_argument_probs = tf.nn.softmax(logits_at_tactic).numpy()

            number_of_args = arg_nums[tactic_idx].numpy().item()
            if  number_of_args > 0:
                expand_bound = int(total_expand_bound**(1/number_of_args))
            else:
                expand_bound = 0
            sorted_indices = np.argsort(-per_argument_probs)
            restricted_indices = sorted_indices[:,:expand_bound]
            if len(restricted_indices) == 0:
                result_idx.append(np.full((1,2), top_tactic_ids[tactic_idx], dtype=np.uint32))
                result_value.append(np.full((1,),tactic_probs[tactic_idx]))
            else:
                arg_combinations = cartesian_product(*restricted_indices)
                first_index = np.tile(np.arange(arg_nums[tactic_idx]), (arg_combinations.shape[0],1))
                temp_tactic_id = np.full((len(arg_combinations), 1, 2), top_tactic_ids[tactic_idx])
                result_idx.extend(np.hstack((temp_tactic_id, logit_idx_to_action[arg_combinations])))
                result_value.append(tactic_probs[tactic_idx]*np.prod(per_argument_probs[first_index, arg_combinations], axis=1))
            first_arg_id += num_args_for_tactic
        if result_value:
            result_value = np.concatenate(result_value)
        else:
            result_value = np.zeros(0)
        ranked_indices = np.argsort(-result_value)
        # ranked_actions = np.array(result_idx, dtype=object)[ranked_indices]
        ranked_actions = [np.array(result_idx[idx], dtype=np.uint32) for idx in ranked_indices]
        ranked_values = result_value[ranked_indices]

        t1 = time.time()
        self.t_predict += (t1 - t0)
        self.n_predict += 1
        # print(f'Predict.ranked_predictions this call {t1-t0:.6f} seconds')
        # if self.n_predict != 0 and self.n_predict % 10 == 0:
        #    print(f'Predict.ranked_predictions average  {self.t_predict/self.n_predict:.6f} second/call')


        return ranked_actions, ranked_values
