import hashlib
from typing import List
from itertools import accumulate
from bisect import bisect_left

FAVORITE_PRIME = 10**9 + 7

def get_split_label(obj, prob: List[float], seed: int = 0) -> int:
    '''
    returns a split label attached to a python object obj that has repr defined

    obj:  string, int is the hash of the datapoint
    prob: List[float] is the list of non-negative numbers with non-zero sum describing the proportions of the split
    seed: int is the parameter to control the split


    returns integer index of the bin for a data point

    Example:
    split_label("data_point1", [80.0, 10.0, 10.0], 42)  returns 0
    split_label("data_point5", [80.0, 10.0, 10.0], 42)  returns 2
    '''
    obj_hash = repr(obj) + '#' + str(seed)
    md5obj_hash = hashlib.md5(obj_hash.encode())
    aprob = list(accumulate(prob))
    position = ((int(md5obj_hash.hexdigest(),16) % FAVORITE_PRIME) / FAVORITE_PRIME) * aprob[-1]
    index = bisect_left(aprob, position)
    return index

def test_split_label(cnt: int, prob: List[float], seed: int = 0):
    '''
    test split_label on integers in range(cnt)

    returns the sample probabilities
    '''
    counter  = [0] * len(prob)
    for i in range(cnt):
        index = get_split_label(i, prob, seed)
        counter[index] += 1

    s = sum(counter)
    return list(x / s for x in counter)
