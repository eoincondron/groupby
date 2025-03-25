from typing import Callable

import numba as nb
from numba.extending import overload
import numpy as np
from pandas.api.types import is_float_dtype, is_integer_dtype, is_bool_dtype

from groupby.util import check_data_inputs_aligned, ArrayType1D


MIN_INT = np.iinfo(np.int64).min
MAX_INT = np.iinfo(np.int64).max


def is_null(x):
    return np.isnan(x)


@overload(is_null)
def jit_is_null(x):
    if isinstance(x, nb.types.Float) or isinstance(x, float):

        def is_null(x):
            return np.isnan(x)

        return is_null
    if isinstance(x, nb.types.Integer):

        def is_null(x):
            return x == MIN_INT

        return is_null
    elif isinstance(x, nb.types.Boolean):

        def is_null(x):
            return False

        return is_null


class NumbaReductionOps:

    @staticmethod
    @nb.njit
    def add_one(x, y):
        return x + 1

    @staticmethod
    @nb.njit
    def min(x, y):
        return x if x <= y else y

    @staticmethod
    @nb.njit
    def max(x, y):
        return x if x >= y else y

    @staticmethod
    @nb.njit
    def sum(x, y):
        return x + y

    @staticmethod
    @nb.njit
    def first(x, y):
        return x

    @staticmethod
    @nb.njit
    def first_skipna(x, y):
        return y if is_null(x) else x

    @staticmethod
    @nb.njit
    def last(x, y):
        return y

    @staticmethod
    @nb.njit
    def last_skipna(x, y, is_null: Callable):
        return x if is_null(y) else y


@nb.njit(parallel=False, nogil=False, fastmath=True)
def _group_by_iterator(
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    reduce_func: Callable,
):
    seen = np.zeros(len(target), dtype=nb.bool_)
    for i in nb.prange(len(group_key)):
        key = group_key[i]
        val = values[i]
        if is_null(val) or (len(mask) and not mask[i]):
            continue

        if seen[key]:
            target[key] = reduce_func(target[key], val)
        else:
            target[key] = val
            seen[key] = True

    return target, seen


def _prepare_mask_for_numba(mask):
    if mask is None:
        mask = np.array([])
    else:
        mask = np.asarray(mask)
        if mask.dtype.kind != "b":
            raise TypeError(f"mask must of Boolean type. Got {mask.dtype}")
    return mask


def get_initial_value_for_dtype(dtype):
    if is_float_dtype(dtype):
        return np.array(np.nan, dtype=dtype)
    elif is_integer_dtype(dtype):
        return np.iinfo(dtype).min
    elif is_bool_dtype(dtype):
        return True
    else:
        raise TypeError("Only floats, integer and Booleans are supported")


@check_data_inputs_aligned("group_key", "values", "mask")
def _group_func_wrap(
    reduce_func: Callable,
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    initial_value: int | float = None,
    mask: ArrayType1D = None,
):
    target = np.full(ngroups, initial_value)
    mask = _prepare_mask_for_numba(mask)
    values = np.asarray(values)

    return _group_by_iterator(
        group_key=group_key,
        values=values,
        target=target,
        mask=mask,
        reduce_func=reduce_func,
    )


@nb.njit(parallel=False, nogil=False)
def _group_count(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    mask: np.ndarray,
):
    out = np.zeros(ngroups)
    for i in nb.prange(len(group_key)):
        key = group_key[i]
        val = values[i]
        if is_null(val) or (len(mask) and not mask[i]):
            continue

        out[key] += 1

    return out


@check_data_inputs_aligned("group_key", "values", "mask")
def group_count(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    mask = _prepare_mask_for_numba(mask)
    values = np.asarray(values)
    return _group_count(**locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_sum(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    if is_float_dtype(values):
        initial_value = np.array(0.0, dtype=values.dtype)
    elif is_integer_dtype(values) or is_bool_dtype(values):
        initial_value = 0
    else:
        raise TypeError("Only floats, integer and Booleans are supported")
    return _group_func_wrap(NumbaReductionOps.sum, **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_mean(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    kwargs = locals().copy()
    sum, seen = group_sum(**kwargs)
    count = group_count(**kwargs)
    return sum / count, seen


@check_data_inputs_aligned("group_key", "values", "mask")
def group_min(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    initial_value = get_initial_value_for_dtype(values.dtype)
    return _group_func_wrap(NumbaReductionOps.min, **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_max(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    initial_value = get_initial_value_for_dtype(values.dtype)
    return _group_func_wrap(NumbaReductionOps.max, **locals())


class NumbaGroupByMethods:

    @staticmethod
    @check_data_inputs_aligned("group_key", "values", "mask")
    def first(
        group_key: ArrayType1D,
        values: ArrayType1D,
        ngroups: int,
        mask: ArrayType1D = None,
        skip_na=True,
    ):
        initial_value = get_initial_value_for_dtype(values.dtype)
        if skip_na:
            reduce_func = NumbaReductionOps.first_skipna
        else:
            reduce_func = NumbaReductionOps.first
        return _group_func_wrap(
            reduce_func=reduce_func,
            group_key=group_key,
            values=values,
            ngroups=ngroups,
            initial_value=initial_value,
            mask=mask,
        )

    @staticmethod
    @check_data_inputs_aligned("group_key", "values", "mask")
    def last(
        group_key: ArrayType1D,
        values: ArrayType1D,
        ngroups: int,
        mask: ArrayType1D = None,
    ):
        initial_value = get_initial_value_for_dtype(values.dtype)
        return _group_func_wrap(NumbaReductionOps.last, **locals())
