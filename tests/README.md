# Tests

## To run tests

This project uses `pytest`.  In general, to run tests, just use:
```bash
$ pytest
```
Optionally the `-v` or `-vv` options will display more information, e.g. names of all the tests being run.

To run a particular test, just provide a (partial) test file path, e.g.
```bash
$ pytest tests/unit
$ pytest tests/unit/test_example.py
```

### The training integration tests
The test `tests/integration/test_training_pipeline.py` runs the pipeline starting with various synthetic datasets
all the way through a small training run.
Various training history statistics, like loss and accuracy, are verified to come out the same as expected.

To run `tests/integration/test_training_pipeline.py`, either run all tests with `pytest` or run only this test with
```bash
$ pytest tests/integration/test_training_pipeline.py
```

To run a specific dataset (useful for debugging a particular problem), use the follow pattern
(which can be found in the test results).
```bash
$ pytest tests/integration/test_training_pipeline.py::test_training[propchain_small-normal]
```

Note: By default, `pytest` seperates the various output types, `stdout`, `stderr`, `warnings`, `logs`, etc.
For full training runs, it may be more natural to run with any/all
of the options `-v` or `-vv`, `--tb=native`, `--assert=plain` and `--capture=no`.

If you made changes to the neural network that change the expected outcomes, you can overwrite the expected outcomes with
```bash
$ pytest --overwrite tests/integration/test_training_pipeline.py
```
or overwrite the expected outcomes of a particular dataset with, e.g.
```bash
$ pytest --overwrite tests/integration/test_training_pipeline.py::test_training[propchain_small-normal]
```

The datasets are stored in `tests/data/*/dataset/` and the parameter files for each integration test are stored in `tests/data/*/params/*`.

### Upgrading to another version of the dataset
When upgrading the version of our dataset, the test `.bin` files in `tests/data` need to be regenerated.
This requires a rather complicated setup and is not fully documented.

## Adding unit tests
Adding small unit tests is encouraged and easy to do.
Just make a test file somewhere in the `tests/unit/` directory of the form `test_*.py`.
Any definition starting with `test_` will be run as a test.  For example:
```python
# file: tests/unit/test_example.py
import pytest

def test_1_plus_1_equals_2():
    assert 1 + 1 == 2

def test_dictionary_comparison():
    expected_result = {"foo": 1.0, "bar": 1.2}
    result = {"foo": 0.9999999, "bar": 1.20000001}
    # pytest.approx is useful when comparing floats
    assert result == pytest.approx(expected_result, rel=1e-3)

def test_dictionary_of_list_comparison():
    expected_result = {"foo": [1.0, 2.0], "bar": [1.2, 3.4]}
    result = {"foo": [0.9999999, 1.999999], "bar": [1.20000001, 3.4000002]}
    # pytest.approx only works one level deep
    assert result == {k:pytest.approx(v, rel=1e-3) for k,v in expected_result.items()}
```

See the existing tests and the [pytest documentation](https://docs.pytest.org/) for more examples.