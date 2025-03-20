from combomethod import combomethod
from typing import Callable
from functools import cached_property, partial
from collections.abc import Sequence, Mapping

import numpy as np
import pandas as pd
import polars as pl
import numba as nb


from .util import ArrayType1D, ArrayType2D, convert_array_inputs_to_dict, check_data_inputs_aligned, parallel_map
ArrayCollection = ArrayType1D | ArrayType2D | Sequence[ArrayType1D] | Mapping[ArrayType1D]

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
    non_trivial = [index for index in indexes
                   if not (isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1)]
    if len(non_trivial) == 0:
        return

    for left, right in zip(non_trivial, non_trivial[1:]):
        if not left.equals(right):
            raise ValueError

    return non_trivial[0]


@nb.njit(parallel=False, nogil=False)
def _group_sum(group_key, values, target, mask):
    for i in nb.prange(len(group_key)):
        if len(mask):
            if not mask[i]:
                continue
        key = group_key[i]
        val = values[i]
        target[key] = target[key] + val

    return target


@check_data_inputs_aligned('group_key', 'values', 'mask')
def group_sum(group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D=None):
    if values.dtype.kind == 'f':
        out_type = values.dtype
    else:
        out_type = int

    target = np.zeros(ngroups, dtype=out_type)
    if mask is None:
        mask = np.array([])

    return _group_sum(group_key, values, target, mask)


@nb.njit(parallel=False, nogil=False)
def group_count(group_key: np.ndarray, mask: np.ndarray, ngroups: int):
    count = np.zeros(ngroups, dtype=nb.int_)

    for i in nb.prange(len(group_key)):
        if len(mask):
            if not mask[i]:
                continue
        key = group_key[i]
        count[key] = count[key] + 1

    return count


class GroupBy:

    def __init__(self, group_keys: ArrayCollection):
        group_key_dict = convert_array_inputs_to_dict(group_keys)
        group_key_dict = {key: array_to_series(val) for key, val in group_key_dict.items()}
        indexes = [s.index for s in group_key_dict.values()]
        common_index = _validate_input_indexes(indexes)
        if common_index is not None:
            group_key_dict = {key: s.set_axis(common_index, axis=0, copy=False)
                              for key, s in group_key_dict.items()}

        self._group_df = pd.DataFrame(group_key_dict, copy=False)
        self.grouper = self._group_df.groupby(list(group_key_dict), observed=True)._grouper

    @property
    def ngroups(self):
        return self.grouper.ngroups

    @property
    def result_index(self):
        return self.grouper.result_index

    @cached_property
    def group_ikey(self):
        return self.grouper.group_info[0]

    @combomethod
    def _apply_gb_func(
            grouper,
            func: Callable,
            values: ArrayCollection,
            mask: ArrayType1D = None,
    ):
        if not isinstance(grouper, GroupBy):
            grouper = GroupBy(grouper)
        value_dict = convert_array_inputs_to_dict(values)
        np_values = list(map(val_to_numpy, value_dict.values()))
        results = map(
            lambda v: func(group_key=grouper.group_ikey, values=v, mask=mask, ngroups=grouper.ngroups),
            np_values,
        )
        out_dict = dict(zip(values, results))
        out = pd.DataFrame(out_dict, index=grouper.result_index)

        return_1d = len(value_dict) == 1 and isinstance(values, ArrayType1D)
        if return_1d:
            out = out.iloc[:, 0]

        if mask is not None:
            count = group_sum(group_key=grouper.group_ikey, values=mask, mask=None, ngroups=grouper.ngroups)
            out = out.loc[count > 0]

        return out

    @combomethod
    def sum(key, values: ArrayCollection, mask: ArrayType1D = None):
        if isinstance(key, GroupBy):
            grouper = key
        else:
            grouper = GroupBy(key)
        return grouper._apply_gb_func(group_sum, values=values, mask=mask)