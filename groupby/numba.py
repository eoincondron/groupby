import operator
from typing import Callable
from inspect import signature
from functools import reduce
import concurrent.futures

import numba as nb
from numba.extending import overload
import numpy as np
from pandas.api.types import is_float_dtype, is_integer_dtype, is_bool_dtype

from groupby.util import check_data_inputs_aligned, ArrayType1D, parallel_reduce


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


def _scalar_func_decorator(func):
    return staticmethod(nb.njit(nogil=True)(func))


class NumbaReductionOps:

    @_scalar_func_decorator
    def count(x, y):
        return x + 1

    @_scalar_func_decorator
    def min(x, y):
        return x if x <= y else y

    @_scalar_func_decorator
    def max(x, y):
        return x if x >= y else y

    @staticmethod
    @nb.njit
    def sum(x, y):
        return x + y

    @_scalar_func_decorator
    def first(x, y):
        return x

    @_scalar_func_decorator
    def first_skipna(x, y):
        return y if is_null(x) else x

    @_scalar_func_decorator
    def last(x, y):
        return y

    @_scalar_func_decorator
    def last_skipna(x, y):
        return x if is_null(y) else y

    @_scalar_func_decorator
    def sum_square(x, y):
        return x + y**2


@nb.njit(nogil=True, fastmath=False)
def _group_by_iterator(
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    reduce_func: Callable,
    must_see: bool = True,
):
    seen = np.full(len(target), not must_see)
    for i in range(len(group_key)):
        key = group_key[i]
        val = values[i]
        if is_null(val) or (len(mask) and not mask[i]):
            continue

        if seen[key]:
            target[key] = reduce_func(target[key], val)
        else:
            target[key] = val
            seen[key] = True

    return target


def _prepare_mask_for_numba(mask):
    if mask is None:
        mask = np.array([], dtype=bool)
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
def _chunk_groupby_args(
    n_chunks: int,
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    reduce_func: Callable,
):
    mask = _prepare_mask_for_numba(mask)
    values = np.asarray(values)

    chunked_args = []
    for gk, v, m in map(
        lambda a: np.array_split(a, n_chunks), [group_key, values, mask]
    ):
        kwargs = dict(
            group_key=gk,
            values=v,
            target=target.copy(),
            mask=m,
            reduce_func=reduce_func,
        )
        bound_args = signature(_group_by_iterator).bind(**kwargs)
        chunked_args.append(bound_args)

    return chunked_args


@check_data_inputs_aligned("group_key", "values", "mask")
def _group_func_wrap(
    reduce_func_name: str,
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    initial_value: int | float = None,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    target = np.full(ngroups, initial_value)
    mask = _prepare_mask_for_numba(mask)
    values = np.asarray(values)

    kwargs = dict(
        group_key=group_key,
        values=values,
        target=target,
        mask=mask,
        reduce_func=getattr(NumbaReductionOps, reduce_func_name),
        must_see=reduce_func_name != "count",
    )

    if n_threads == 1:
        return _group_by_iterator(**kwargs)
    else:
        chunked_args = _chunk_groupby_args(**kwargs, n_chunks=n_threads)
        return parallel_reduce(
            _group_by_iterator, reduce_func_name, [args.args for args in chunked_args]
        )


@check_data_inputs_aligned("group_key", "values", "mask")
def group_count(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    initial_value = 0
    return _group_func_wrap("count", **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_sum(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    if is_float_dtype(values):
        initial_value = np.array(0.0, dtype=values.dtype)
    elif is_integer_dtype(values) or is_bool_dtype(values):
        initial_value = 0
    else:
        raise TypeError("Only floats, integer and Booleans are supported")
    return _group_func_wrap("sum", **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_mean(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    kwargs = locals().copy()
    sum = group_sum(**kwargs)
    count = group_count(**kwargs)
    return sum / count


@check_data_inputs_aligned("group_key", "values", "mask")
def group_min(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    initial_value = get_initial_value_for_dtype(values.dtype)
    return _group_func_wrap("min", **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_max(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    initial_value = get_initial_value_for_dtype(values.dtype)
    return _group_func_wrap("max", **locals())


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
            reduce_func = "first_skipna"
        else:
            reduce_func = "first"
        return _group_func_wrap(
            reduce_func_name=reduce_func,
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
        return _group_func_wrap("last", **locals())


def _group_by_max_diff(group_key, values, max_diff, n_groups: int):
    group_counter = 0
    last_seen = np.empty(n_groups)
    group_tracker = np.full(n_groups, -1)
    out = np.full(len(group_key), -1)
    for i in range(len(group_key)):
        key = group_key[i]
        current_group = group_tracker[key]
        current_value = values[i]
        if current_group == -1:
            make_new_group = True
        else:
            make_new_group = (current_value - last_seen[key]) > max_diff

        last_seen[key] = current_value
        if make_new_group:
            group_counter += 1
            group_tracker[key] = group_counter

        out[i] = group_tracker[key]

    return out
