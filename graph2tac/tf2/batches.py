"""
This is stream to stream converter.
It takes a stream of datapoints and output a stream of batches
"""

import random
from typing import Generic, Iterable, TypeVar

T = TypeVar('T')

class Batches(Generic[T]):
    """
    constructs iterator of batches from stream iterator of datapoints
    and batch size. Never returns empty batch but stops iterations
    when input is depleted
    """
    def __init__(self, stream: Iterable[T], batch_size: int):
        self.__stream = iter(stream)
        self.__batch_size = batch_size
        if hasattr(stream, '__len__'):
            self.__size = (len(stream) - 1) // batch_size + 1
        elif hasattr(stream, '__length_hint__'):
            self.__size = (stream.__length_hint__() - 1) // batch_size + 1
        else:
            self.__size = None

    def __iter__(self):
        return self

    def __next_batch(self) -> list[T]:
        batch = []
        while len(batch) < self.__batch_size:
            try:
                batch.append(next(self.__stream))
            except StopIteration:
                break
        return batch

    def __next__(self) -> list[T]:
        batch = self.__next_batch()
        if len(batch) == 0:
            raise StopIteration
        return batch

    def __len__(self):
        return (self.__size)


def Repeat(data: Iterable[T], shuffled=False) -> Iterable[T]:
    """
    repeates infinitely the original iterator, reshuffling before each pass   
    """
    data = list(data)
    while True:
        if shuffled:
            random.shuffle(data)
        yield from data

if __name__ == "__main__":
    def test_1():
        t_input = iter([1,2,3,4,5,6])
        output = [b for b in Batches(t_input, 2)]
        assert output == [[1,2],[3,4],[5,6]]

    def test_2():
        t_input = [1,2,3,4,5,6]
        output = [b for b in Batches(t_input, 2)]
        assert output == [[1,2],[3,4],[5,6]]

    def test_3():
        t_input = [1,2,3,4,5,6,7]
        output = [b for b in Batches(t_input, 3)]
        assert output == [[1,2,3],[4,5,6],[7]]

    def test_4():
        import numpy as np
        t_input = np.array([1,2,3,4,5,6,7])
        output = [b for b in Batches(t_input, 3)]
        assert output == [[1,2,3],[4,5,6],[7]]

    def test_5():
        t_input = set([1,2,3,4,5,6,7])
        output = [b for b in Batches(t_input, 3)]
        assert output == [[1,2,3],[4,5,6],[7]]


    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
