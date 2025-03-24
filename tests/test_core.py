import pandas as pd
import pytest
import numpy as np

from groupby.numba import MIN_INT
from groupby.core import GroupBy


# Unit tests


class TestGroupBy:

    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max", "count"])
    @pytest.mark.parametrize("key_dtype", [int, str, float, "float32"])
    @pytest.mark.parametrize("key_type", [np.array, pd.Series])
    @pytest.mark.parametrize("value_dtype", [int, float, "float32", bool])
    @pytest.mark.parametrize("value_type", [np.array, pd.Series])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_basic(
        self, method, key_dtype, key_type, value_dtype, value_type, use_mask
    ):
        key = key_type([1, 1, 2, 1, 3, 3, 6, 1, 6], dtype=key_dtype)
        value = value_type([-1, 0, 4, 3, 8, 6, 3, 1, 12], dtype=value_dtype)

        if use_mask:
            mask = pd_mask = key.astype(int) != 1
        else:
            mask = None
            pd_mask = slice(None)
        result = getattr(GroupBy, method)(key, value, mask=mask)

        expected = getattr(pd.Series(value)[pd_mask].groupby(key[pd_mask]), method)()
        pd.testing.assert_series_equal(result, expected, check_dtype=method != "mean")

        gb = GroupBy(key)
        result = getattr(gb, method)(value, mask=mask)
        pd.testing.assert_series_equal(result, expected, check_dtype=method != "mean")

    @pytest.mark.parametrize("use_mask", [True, False])
    @pytest.mark.parametrize("dtype", [float, int])
    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max", "count"])
    def test_with_nulls(self, method, dtype, use_mask):
        null = MIN_INT if dtype == int else np.nan
        key = np.array([1, 1, 2, 1, 3, 3, 6, 1, 6])
        value = pd.Series([-1, 0, null, 3, 8, 6, null, 1, null], dtype=dtype)
        not_null = value != null if dtype == int else value.isnull()
        if use_mask:
            mask = value < 6
            pd_mask = mask & not_null
        else:
            mask = None
            pd_mask = not_null
        result = getattr(GroupBy, method)(key, value, mask=mask)

        expected = value[pd_mask].groupby(key[pd_mask]).agg(method)
        pd.testing.assert_series_equal(result, expected)
