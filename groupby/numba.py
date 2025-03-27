import operator
from typing import Callable
from inspect import signature
from functools import reduce
import concurrent.futures

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
    def last_skipna(x, y, is_null: Callable):
        return x if is_null(y) else y


@nb.njit(nogil=True, fastmath=False)
def _group_by_iterator(
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    reduce_func: Callable,
    must_see: bool = True
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
    chunk_len = (len(group_key) + 1) // n_chunks

    chunked_args = []
    for chunk_no in range(n_chunks):
        slice_ = slice(chunk_no * chunk_len, (chunk_no + 1) * chunk_len)
        kwargs = dict(
            group_key=group_key[slice_],
            values=values[slice_],
            target=target.copy(),
            mask=mask[slice_],
            reduce_func=reduce_func,
        )
        bound_args = signature(_group_by_iterator).bind(**kwargs)
        chunked_args.append(bound_args)

    return chunked_args


@check_data_inputs_aligned("group_key", "values", "mask")
def _group_func_wrap(
    reduce_func: str,
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
        reduce_func=getattr(NumbaReductionOps, reduce_func),
        must_see=reduce_func != 'count',
    )

    if n_threads == 1:
        return _group_by_iterator(**kwargs)
    else:
        try:
            reduce_func_vec = dict(
                count=operator.add,
                sum=operator.add,
                max=np.maximum,
                min=np.minimum,
            )[reduce_func]
        except:
            raise ValueError(f"Multi-threading not supported for {reduce_func}")
        chunked_args = _chunk_groupby_args(**kwargs, n_chunks=n_threads)
        results = [None] * n_threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_to_chunk = {
                executor.submit(_group_by_iterator, *args.args): i
                for i, args in enumerate(chunked_args)
            }
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                results[chunk] = future.result()

        return reduce(reduce_func_vec, results)


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
    return _group_func_wrap('sum', **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_mean(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None, n_threads: int = 1,
):
    kwargs = locals().copy()
    sum, seen = group_sum(**kwargs)
    count = group_count(**kwargs)
    return sum / count, seen


@check_data_inputs_aligned("group_key", "values", "mask")
def group_min(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None, n_threads: int = 1,
):
    initial_value = get_initial_value_for_dtype(values.dtype)
    return _group_func_wrap('min', **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_max(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None, n_threads: int = 1,
):
    initial_value = get_initial_value_for_dtype(values.dtype)
    return _group_func_wrap('max', **locals())


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
            reduce_func = 'first_skipna'
        else:
            reduce_func = 'first'
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
        return _group_func_wrap('last', **locals())


from pandas.core._numba.kernels import grouped_sum
