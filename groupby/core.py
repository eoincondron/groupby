from typing import Callable
from functools import cached_property, wraps
from collections.abc import Sequence, Mapping

import numpy as np
import pandas as pd
import polars as pl

from .numba import group_sum, group_mean, group_min, group_max, group_count
from .util import (
    ArrayType1D,
    ArrayType2D,
    convert_array_inputs_to_dict,
    get_array_name,
    TempName,
)

ArrayCollection = (
    ArrayType1D | ArrayType2D | Sequence[ArrayType1D] | Mapping[ArrayType1D]
)


def array_to_series(arr: ArrayType1D):
    if isinstance(arr, pl.Series):
        return arr.to_pandas()
    else:
        return pd.Series(arr)


def val_to_numpy(val: ArrayType1D):
    try:
        return val.to_numpy()
    except:
        return np.asarray(val)


def _validate_input_indexes(indexes):
    lengths = set(map(len, indexes))
    if len(lengths) > 1:
        raise ValueError(f"found more than one unique length: {lengths}")
    non_trivial = [
        index
        for index in indexes
        if not (
            isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1
        )
    ]
    if len(non_trivial) == 0:
        return

    for left, right in zip(non_trivial, non_trivial[1:]):
        if not left.equals(right):
            raise ValueError

    return non_trivial[0]


def combomethod(method):
    @wraps(method)
    def wrapper(key, values, mask=None):
        if not isinstance(key, GroupBy):
            key = GroupBy(key)
        return method(key, values, mask)

    return wrapper


class GroupBy:

    def __init__(self, group_keys: ArrayCollection):
        group_key_dict = convert_array_inputs_to_dict(group_keys)
        group_key_dict = {
            key: array_to_series(val) for key, val in group_key_dict.items()
        }
        indexes = [s.index for s in group_key_dict.values()]
        common_index = _validate_input_indexes(indexes)
        if common_index is not None:
            group_key_dict = {
                key: s.set_axis(common_index, axis=0, copy=False)
                for key, s in group_key_dict.items()
            }

        self._group_df = pd.DataFrame(group_key_dict, copy=False)
        self.grouper = self._group_df.groupby(
            list(group_key_dict), observed=True
        )._grouper
        self.key_names = [
            None if isinstance(key, TempName) else key for key in group_key_dict
        ]

    @property
    def ngroups(self):
        return self.grouper.ngroups

    @cached_property
    def result_index(self):
        index = self.grouper.result_index
        index.names = self.key_names
        return index

    @cached_property
    def group_ikey(self):
        return self.grouper.group_info[0]

    # @combomethod
    def _apply_gb_func(
        self,
        func: Callable,
        values: ArrayCollection,
        mask: ArrayType1D = None,
    ):
        value_dict = convert_array_inputs_to_dict(values)
        np_values = list(map(val_to_numpy, value_dict.values()))
        results = map(
            lambda v: func(
                group_key=self.group_ikey, values=v, mask=mask, ngroups=self.ngroups
            ),
            np_values,
        )
        out_dict = dict(zip(values, results))
        out = pd.DataFrame(out_dict, index=self.result_index)

        return_1d = len(value_dict) == 1 and isinstance(values, ArrayType1D)
        if return_1d:
            out = out.iloc[:, 0]
            if get_array_name(values) is None:
                out.name = None

        if mask is not None:
            count = group_sum(
                group_key=self.group_ikey, values=mask, mask=None, ngroups=self.ngroups
            )
            out = out.loc[count > 0]

        return out

    @combomethod
    def count(self, values: ArrayCollection, mask: ArrayType1D = None):
        return self._apply_gb_func(group_count, values=values, mask=mask)

    @combomethod
    def sum(self, values: ArrayCollection, mask: ArrayType1D = None):
        return self._apply_gb_func(group_sum, values=values, mask=mask)

    @combomethod
    def mean(self, values: ArrayCollection, mask: ArrayType1D = None):
        return self._apply_gb_func(group_mean, values=values, mask=mask)

    @combomethod
    def min(self, values: ArrayCollection, mask: ArrayType1D = None):
        return self._apply_gb_func(group_min, values=values, mask=mask)

    @combomethod
    def max(self, values: ArrayCollection, mask: ArrayType1D = None):
        return self._apply_gb_func(group_max, values=values, mask=mask)
