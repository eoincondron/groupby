import inspect
from typing import Callable
from functools import cached_property, wraps
from collections.abc import Sequence, Mapping

import numpy as np
import pandas as pd
import polars as pl
import numba as nb

from .util import (
    ArrayType1D,
    ArrayType2D,
    convert_array_inputs_to_dict,
    check_data_inputs_aligned,
    get_array_name,
    TempName,
)

ArrayCollection = (
    ArrayType1D | ArrayType2D | Sequence[ArrayType1D] | Mapping[ArrayType1D]
)


MIN_INT = np.iinfo(np.int64).min


@nb.njit
def _isnull_float(x):
    return x != x


@nb.njit
def _isnull_int(x):
    return x == MIN_INT


@nb.njit(parallel=False, nogil=False)
def _group_count(
    group_key: np.ndarray,
    values: ArrayType1D,
    null_check: Callable,
    mask: np.ndarray,
    ngroups: int,
):
    count = np.zeros(ngroups, dtype=nb.int_)

    for i in nb.prange(len(group_key)):
        if len(mask) and not mask[i]:
            continue
        key = group_key[i]
        if not null_check(values[i]):
            count[key] = count[key] + 1

    return count


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


def _prepare_mask_for_numba(mask):
    if mask is None:
        mask = np.array([])
    else:
        mask = np.asarray(mask)
        if mask.dtype.kind != "b":
            raise TypeError(f"mask must of Boolean type. Got {mask.dtype}")
    return mask


@nb.njit
def _update_target(
    i: int,
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    reduce_func: Callable,
    null_check: Callable,
):
    key = group_key[i]
    val = values[i]
    if null_check(val) or (len(mask) and not mask[i]):
        return

    target[key] = reduce_func(target[key], val)


@nb.njit
def _nb_sum(x, y):
    return x + y


@nb.njit(parallel=False, nogil=False)
def _group_sum(
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    null_check: Callable,
):
    for i in nb.prange(len(group_key)):
        _update_target(
            i,
            group_key=group_key,
            values=values,
            target=target,
            mask=mask,
            reduce_func=_nb_sum,
            null_check=null_check,
        )

    return target


@check_data_inputs_aligned("group_key", "values", "mask")
def _group_func_wrap(
    nb_func: Callable,
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
):
    if values.dtype.kind == "f":
        out_type = values.dtype
        null_check = _isnull_float
    else:
        out_type = int
        null_check = _isnull_int

    target = np.zeros(ngroups, dtype=out_type)
    mask = _prepare_mask_for_numba(mask)
    values = np.asarray(values)

    kwargs = locals().copy()
    return nb_func(
        **{
            arg_name: kwargs[arg_name]
            for arg_name in inspect.signature(nb_func).parameters
        }
    )


@check_data_inputs_aligned("group_key", "values", "mask")
def group_sum(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    return _group_func_wrap(_group_sum, **locals())


# @nb.njit(parallel=False, nogil=False)
# def _group_count(
#     group_key: np.ndarray,
#     values: np.ndarray,
#     target: np.ndarray,
#     mask: np.ndarray,
#     null_check: Callable,
# ):
#     for i in nb.prange(len(group_key)):
#         _update_target(
#             i,
#             group_key=group_key,
#             value=1,
#             target=target,
#             mask=mask,
#             groupby_func=_nb_sum,
#             null_check=null_check,
#         )
#
#     return target


@nb.njit(parallel=False, nogil=False)
def group_size(group_key: np.ndarray, mask: np.ndarray, ngroups: int):
    count = np.zeros(ngroups, dtype=nb.int_)

    for i in nb.prange(len(group_key)):
        if len(mask) and not mask[i]:
            continue
        key = group_key[i]
        count[key] = count[key] + 1

    return count


@nb.njit(parallel=False, nogil=False)
def _group_count(
    group_key: np.ndarray,
    values: ArrayType1D,
    null_check: Callable,
    mask: np.ndarray,
    ngroups: int,
):
    count = np.zeros(ngroups, dtype=nb.int_)

    for i in nb.prange(len(group_key)):
        if len(mask) and not mask[i]:
            continue
        key = group_key[i]
        if not null_check(values[i]):
            count[key] = count[key] + 1

    return count


@check_data_inputs_aligned("group_key", "values", "mask")
def group_count(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    mask = _prepare_mask_for_numba(mask)
    kwargs = locals().copy()
    if values.dtype.kind == "f":
        null_check = _isnull_float
    else:
        null_check = _isnull_int

    return _group_count(**kwargs, null_check=null_check)


@check_data_inputs_aligned("group_key", "values", "mask")
def group_mean(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    kwargs = locals().copy()
    return group_sum(**kwargs) / group_count(**kwargs)


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
    def sum(self, values: ArrayCollection, mask: ArrayType1D = None):
        return self._apply_gb_func(group_sum, values=values, mask=mask)

    @combomethod
    def mean(self, values: ArrayCollection, mask: ArrayType1D = None):
        return self._apply_gb_func(group_mean, values=values, mask=mask)
