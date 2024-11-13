from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import tqdm
import tensorflow as tf

# for each datapoint get
  # index of tactic
  # index of embedding
# for each datapoint, build table of 1000 most recent examples (after sorting)
  # index of embedding (possible -1 if not enough)
  # index of tactic
# for each datapoint figure out if correct tactic is anywhere in table
  # collect data
  # report results
# figure out what packages I have
  # report
# for each datapoint, do max and softmax to see if tactic is correct
  # inner product
  # max
  # softmax
  # report accuracies
# for each datapoint do max and softmax to see if tactic is correct
  # 

def get_examples(name_i_to_indices: dict[int, list[int]], global_cxt: list[int], size: int):
    global_cxt = sorted(global_cxt)
    output = []
    for name_i in reversed(global_cxt):
        for i in reversed(name_i_to_indices[name_i]):
            if len(output) >= size:
                break
            output.append(i)
        if len(output) >= size:
            break
    if len(output) < size:
        output.extend([-1] * (size - len(output))) 
    assert len(output) == size, len(output)
    return output


def process_data(data_jsonl: Path, limit=None) -> list[dict[str, Any]]:
    data = []
    name_i_to_indices = defaultdict(list)
    bad_examples = set()
    with data_jsonl.open() as f:
        cnt = 0
        for i, l in enumerate(tqdm.tqdm(f)):
            if limit is not None and cnt >= limit:
                break
            cnt += 1

            datapoint = json.loads(l)
            global_cxt = datapoint["global_context"]
            
            last_thousand = get_examples(name_i_to_indices, global_cxt, 1000)
            data.append({
                "tactic_id": datapoint["tactic_id"],
                "name_id": datapoint["metadata_name_id"],
                "prev_indices": last_thousand
            })
            name_i_to_indices[datapoint["metadata_name_id"]].append(i)

            if datapoint["metadata_name_id"] != max(datapoint["global_context"], default=0) + 1:
                bad_examples.add(datapoint["metadata_name_id"])
    
    print("ID not at end of global context", len(bad_examples))
    return data

def process_names(name_jsonl: Path) -> list[str]:
    with name_jsonl.open() as f:
        names = [name for name in f]
    return names

def process_tactics(tactic_jsonl: Path) -> list[str]:
    with tactic_jsonl.open() as f:
        tactic = [tactic for tactic in f]
    return tactic

def check_if_tactic_is_in_history(data: list[dict[str, Any]]):
    checks = 0
    count = 0
    for d in tqdm.tqdm(data):
        tactic_id = d["tactic_id"]
        if any(i != -1 and data[i]["tactic_id"] == tactic_id for i in d["prev_indices"]):
            checks += 1
        count += 1
    print(checks, count)

def check_libraries(data: list[dict[str, Any]], names: list[str]):
    prefixes = Counter()
    double_prefixes = Counter()
    for d in data:
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        prefixes[prefix] += 1
        double_prefix = ".".join(names[name_id].split(".")[:2])
        double_prefixes[double_prefix] += 1
    print(prefixes)
    print(double_prefixes)

#def count_if_id_not_where_expected(data: list[dict[str, Any]], names: list[str]):
#    good = Counter()
#    bad = Counter()
#    cnt = 0
#    for d in data:
#        name_id = d["name_id"]
#        prefix = names[name_id].split(".")[0]
#        context = d["prev_indices"]
#        if name_id == max(context, default=0) + 1:
#            good[prefix] += 1
#        else:
#            bad[prefix] += 1
#            print(name_id, names[name_id], max(context, default=0))
#            cnt += 1
#            if cnt > 1000:
#                break
#    print(good, bad)

def count_proofstates_with_new_tactics(data: list[dict[str, Any]], names: list[str]):
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

def max_accuracy(data: list[dict[str, Any]], embeddings, dist: str = "cosign"):
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

    print(train_correct, train_count, train_correct/train_count , valid_correct, valid_count, valid_correct/valid_count, novel_correct, novel_count, novel_correct/novel_count)

def softmax_accuracy(data: list[dict[str, Any]], embeddings, names, temp, dist: str = "cosign"):
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
        max_tactic_id = max(combined_probs, key=lambda t: combined_probs[t], default=0)
        
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

    print(train_correct, train_count, train_correct/train_count , valid_correct, valid_count, valid_correct/valid_count, novel_correct, novel_count, novel_correct/novel_count)

def frequency_accuracy(data: list[dict[str, Any]], embeddings):
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
            tactic_id = data[d["prev_indices"][i]]["tactic_id"]
            combined_probs[tactic_id] += 1
        max_tactic_id = max(combined_probs, key=lambda t: combined_probs[t])
        
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

    print(train_correct, train_count, train_correct/train_count , valid_correct, valid_count, valid_correct/valid_count, novel_correct, novel_count, novel_correct/novel_count)

def train_valid_datasets(data: list[dict[str, Any]], names: list[str], embeddings):
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
        
def train_valid_datasets2(data: list[dict[str, Any]], names: list[str]):
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

def train_valid_datasets3(data: list[dict[str, Any]], names: list[str], num_tactics):
    train_query = []
    train_keys = []
    train_tactic_ids = []
    train_correct = []
    valid_query = []
    valid_keys = []
    valid_tactic_ids = []
    valid_correct = []
    for i, d in enumerate(tqdm.tqdm(data)):
        name_id = d["name_id"]
        prefix = names[name_id].split(".")[0]
        tactic_id = d["tactic_id"]
        tactic_ids = np.array([num_tactics if j == -1 else data[j]["tactic_id"] for j in d["prev_indices"]])
        if all(id != tactic_id for id in tactic_ids):
            continue
        if prefix == "Coq":
            train_query.append(i)
            train_keys.append(d["prev_indices"])
            train_tactic_ids.append(tactic_ids)
            train_correct.append(tactic_id)
        else:
            valid_query.append(i)
            valid_keys.append(d["prev_indices"])
            valid_tactic_ids.append(tactic_ids)
            valid_correct.append(tactic_id)
    return np.array(train_query), np.array(train_keys), np.array(train_tactic_ids), np.array(train_correct), np.array(valid_query), np.array(valid_keys), np.array(valid_tactic_ids), np.array(valid_correct)

class MyLayer(tf.keras.layers.Layer):

    def call(self, x, y, temp):
        #tf.print("tf_query", x)
        #tf.print("tf_keys", y)
        # normalize
        x = x / tf.norm(x, keepdims=True, axis=-1)
        y = y / tf.norm(y, keepdims=True, axis=-1)
        #tf.print("tf_query_norm", x)
        #tf.print("tf_keys_norm", y)
        # multiply
        logits = tf.einsum("ik,ijk->ij", x, y)
        # temp
        logits = logits / temp
        #tf.print("tf_logits", logits)
        # softmax
        #logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        #probs = tf.exp(logits)
        #logits = logits - tf.math.log(tf.reduce_sum(probs, axis=-1, keepdims=True))
        
        # output
        return logits

class SoftmaxCollapse(tf.keras.layers.Layer):
    def __init__(self, num_tactics):
        super().__init__()
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

def train_model(data, names, embeddings, num_tactics):
    embeddings = np.concatenate([embeddings, np.ones([1, embeddings.shape[1]])])

    dim = 128
    query_id = tf.keras.layers.Input(shape=tuple(), dtype=tf.int32)
    key_ids = tf.keras.layers.Input(shape=(1000,), dtype=tf.int32)
    tactic_ids = tf.keras.layers.Input(shape=(1000,), dtype=tf.int32)

    embeddings = tf.constant(embeddings, dtype=tf.float32)
    query = tf.gather(embeddings, query_id % len(embeddings))
    keys = tf.gather(embeddings, key_ids % len(embeddings))

    temp = tf.Variable(0.03)
    
    x = tf.keras.layers.Dropout(rate=0.1)(query)
    x = tf.keras.layers.Dense(dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.Dense(dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.Dense(dim)(x)
    x = x + query

    y = tf.keras.layers.Dropout(rate=0.1)(keys)
    y = tf.keras.layers.Dense(dim, activation="relu")(y)
    y = tf.keras.layers.Dropout(rate=0.1)(y)
    y = tf.keras.layers.Dense(dim, activation="relu")(y)
    y = tf.keras.layers.Dropout(rate=0.1)(y)
    y = tf.keras.layers.Dense(dim)(y)
    y = y + keys

    logits = MyLayer()(x, y, temp)
    prediction = SoftmaxCollapse(num_tactics)(logits, tactic_ids)
    #prediction = tf.keras.layers.Dense(num_tactics+1)(x)
    
    model = tf.keras.Model(inputs=[query_id, key_ids, tactic_ids], outputs=prediction)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0003, weight_decay=0.1, global_clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True), tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    train_query, train_keys, train_tactic_ids, train_correct, valid_query, valid_keys, valid_tactic_ids, valid_correct = train_valid_datasets3(data, names, num_tactics)

    model.fit(
        x=[train_query, train_keys, train_tactic_ids],
        y=train_correct,
        epochs=10,
        validation_data=([valid_query, valid_keys, valid_tactic_ids], valid_correct),
        batch_size=128
    )

    return model

def softmax_accuracy_from_model(data: list[dict[str, Any]], names, model, embeddings, num_tactics):
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

    print(train_correct, train_count, train_correct/train_count , valid_correct, valid_count, valid_correct/valid_count, novel_correct, novel_count, novel_correct/novel_count)
    return all_results

# TODO: train matrix
# TODO: breakdown on train/valid

if __name__ == "__main__":
    data_dir = Path(sys.argv[1])
    data = process_data(data_dir / "data.jsonl", limit=100000)
    check_if_tactic_is_in_history(data)
    names = process_names(data_dir / "context_names.jsonl")
    check_libraries(data, names)
    count_proofstates_with_new_tactics(data, names)
    embeddings = np.load(data_dir / "embeddings.npy")  
    print(embeddings.shape)
    tactics = process_tactics(data_dir / "tactic_names.jsonl")
    num_tactics = len(tactics)

    model = train_model(data, names, embeddings, num_tactics)
    softmax_accuracy_from_model(data, names, model, embeddings, num_tactics)
    
    max_accuracy(data, embeddings)
    frequency_accuracy(data, embeddings)
    ##for temp in [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9]:
    ##for temp in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    #for temp in [1e-1, 6e-2, 3e-2, 1e-2, 6e-3, 3e-3, 1e-3]:
    for temp in [3e-2]:
        print(temp)
        softmax_accuracy(data, embeddings, names, temp)
    
