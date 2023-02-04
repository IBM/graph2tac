#TODO(jrute): Remove file when we have real unit tests

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