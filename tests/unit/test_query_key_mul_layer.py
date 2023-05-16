import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force tests to run on CPU

import tensorflow as tf

from graph2tac.tfgnn.tasks import QueryKeyMul

class TestQueryKeyMul:
    """Test that all methods of QueryKeyMul produces the correct values."""
    def build_queries_and_keys(self):
        # [batch, None(args), hdim]
        self.queries = tf.ragged.constant([
            [], 
            [[1.0, 2.0], [3.0, 4.0]], 
            [[2.0, 3.0],],
            [[4.0, 5.0], [6.0, 7.0]]
        ], ragged_rank=1, inner_shape=(2,))
        
        # [batch, None(context), hdim]
        self.keys = tf.ragged.constant([
            [[1.0, 2.0], [2.0, 3.0]],
            [],
            [[2.0, 3.0], [3.0, 4.0]],
            [[1.0, 2.0]]
        ], ragged_rank=1, inner_shape=(2,))

        # it is possible to have completely empty queries where there are no arguments in the whole batch
        # [batch, None(args), hdim]
        self.queries_empty = tf.ragged.constant([[], [], [], []], ragged_rank=1, inner_shape=(2,))  
        # it is possible to have completely empty keys where the local context is empty for the whole batch
        # [batch, None(context), hdim]
        self.keys_empty = tf.ragged.constant([[], [], [], []], ragged_rank=1, inner_shape=(2,)) 

        # during inference we often have singleton queries with a batch dimension of 1
        # [1, None(args), hdim]
        self.queries_singleton = tf.ragged.constant([[[4.0, 5.0], [6.0, 7.0]]], ragged_rank=1, inner_shape=(2,))  
        # it is possible to have completely empty keys where the local context is empty for the whole batch
        # [1, None(context), hdim]
        self.keys_singleton = tf.ragged.constant([[[1.0, 2.0]]], ragged_rank=1, inner_shape=(2,))  

    def query_key_logit_is_correct_value(self, query_key_mul_layer: QueryKeyMul):
        logits = query_key_mul_layer(queries=self.queries, keys=self.keys)
        # [batch, None(args), None(context)]
        correct = tf.ragged.constant([
            [],  # 0 args, 2 context
            [[], []],  # 2 args, 0 context
            [[13.0, 18.0]],  # 1 args, 2 context
            [[14.0], [20.0]],  # 2 args, 1 context
        ], ragged_rank=2, inner_shape=())
        assert logits.to_list() == correct.to_list()
    
    def query_empty_key_logit_is_correct_value(self, query_key_mul_layer: QueryKeyMul):
        logits = query_key_mul_layer(queries=self.queries_empty, keys=self.keys)
        correct = tf.ragged.constant([[], [], [], []], ragged_rank=2)
        assert logits.to_list() == correct.to_list()
    
    def empty_query_key_logit_is_correct_value(self, query_key_mul_layer: QueryKeyMul):
        logits = query_key_mul_layer(queries=self.queries, keys=self.keys_empty)
        correct = tf.ragged.constant([[], [[],[]], [[]], [[], []]], ragged_rank=2)
        assert logits.to_list() == correct.to_list()

    def empty_query_empty_key_logit_is_correct_value(self, query_key_mul_layer: QueryKeyMul):
        logits = query_key_mul_layer(queries=self.queries_empty, keys=self.keys_empty)
        correct = tf.ragged.constant([[], [], [], []], ragged_rank=2)
        assert logits.to_list() == correct.to_list()
    
    def singleton_query_singleton_key_logit_is_correct_value(self, query_key_mul_layer: QueryKeyMul):
        logits = query_key_mul_layer(queries=self.queries_singleton, keys=self.keys_singleton)
        correct = tf.ragged.constant([[[14.0], [20.0]]], ragged_rank=2)
        assert logits.to_list() == correct.to_list()

    def all_logits_are_correct_values(self, query_key_mul_layer: QueryKeyMul):
        self.build_queries_and_keys()
        self.query_key_logit_is_correct_value(query_key_mul_layer)
        self.empty_query_key_logit_is_correct_value(query_key_mul_layer)
        self.singleton_query_singleton_key_logit_is_correct_value(query_key_mul_layer)

    def test_default_method(self):
        self.all_logits_are_correct_values(QueryKeyMul())
    
    def test_map_fn(self):
        self.all_logits_are_correct_values(QueryKeyMul(method="map_fn"))
    
    def test_broadcast_ragged(self):
        self.all_logits_are_correct_values(QueryKeyMul(method="broadcast_ragged"))

    def test_ragged_to_tensor_to_ragged(self):
        self.all_logits_are_correct_values(QueryKeyMul(method="ragged_to_dense_to_ragged"))

        
        

