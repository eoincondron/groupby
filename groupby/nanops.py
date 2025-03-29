from functools import partial

import numba as nb
import numpy as np
import pandas as pd
from pandas.core import nanops
from .numba import NumbaReductionOps, is_null
from .util import parallel_reduce, parallel_map, n_threads_from_array_length


@nb.njit(nogil=True)
def _get_first_non_null(arr):
    for i, x in enumerate(arr):
        if not is_null(x):
            return i, x


@nb.njit(nogil=True)
def _nb_reduce(arr, reduce_func, skipna: bool = True, initial_value=None):
    if skipna:
        if initial_value is None:
            start, out = _get_first_non_null(arr)
        else:
            start, out = 0, initial_value
        for j in range(start + 1, len(arr)):
            x = arr[j]
            if not is_null(x):
                out = reduce_func(out, x)
    else:
        out = arr[0]
        for i in range(1, len(arr)):
            x = arr[i]
            out = reduce_func(out, x)
    return out


def reduce_1d(reduce_func_name: str, arr, skipna: bool = True, n_threads: int = 1):
    reduce_func = getattr(NumbaReductionOps, reduce_func_name)
    if n_threads is None:
        n_threads = n_threads_from_array_length(len(arr))
    if n_threads == 1:
        return _nb_reduce(
            arr,
            reduce_func=reduce_func,
            skipna=skipna,
            initial_value=0 if reduce_func_name == "count" else None,
        )
    else:
        return parallel_reduce(
            lambda a: _nb_reduce(a, reduce_func=reduce_func, skipna=skipna),
            reduce_func_name,
            list(zip(np.array_split(arr, n_threads))),
        )


def reduce_2d(
    reduce_func_name: str, arr, skipna: bool = True, n_threads: int = 1, axis=0
):
    if n_threads is None:
        n_threads = n_threads_from_array_length(arr.size)
    if axis == 0:
        arr = arr.T
    mapper = lambda x: reduce_1d(reduce_func_name, x, skipna)
    if n_threads == 1:
        results = list(map(mapper, arr))
    else:
        results = parallel_map(mapper, list(zip(arr)))
    return np.array(results)


def reduce(arr, reduce_func_name: str, skipna=True, min_count=0, axis=None):
    arr = np.asarray(arr)
    if min_count != 0:
        return getattr(nanops, f"nan{reduce_func_name}")(**locals())

    if arr.ndim == 1:
        return reduce_1d(reduce_func_name, arr, skipna=skipna)
    else:
        if axis is None:
            # warn
            axis = 0
        return reduce_2d(reduce_func_name, arr, axis=axis)


def nansum(arr, skipna: bool = True, min_count: int = 0, axis: int = None):
    return reduce(reduce_func_name="sum", **locals())


def nancount(arr, skipna: bool = True, min_count: int = 0, axis: int = None):
    return reduce(reduce_func_name="count", **locals())


def nanmean(arr, skipna: bool = True, min_count: int = 0, axis: int = None):
    return nansum(**locals()) / nancount(**locals())


def nanmax(arr, skipna: bool = True, min_count: int = 0, axis: int = None):
    return reduce(reduce_func_name="max", **locals())


def nanmin(arr, skipna: bool = True, min_count: int = 0, axis: int = None):
    return reduce(reduce_func_name="min", **locals())


def nanvar(
    arr, skipna: bool = True, min_count: int = 0, axis: int = None, ddof: int = 1
):
    kwargs = locals().copy()
    del kwargs["ddof"]
    count = nancount(**kwargs) - ddof
    mean_sq = reduce(reduce_func_name="sum_square", **kwargs) / count
    mean = reduce(reduce_func_name="sum", **kwargs) / count
    return mean_sq - mean**2


def nanstd(
    arr, skipna: bool = True, min_count: int = 0, axis: int = None, ddof: int = 1
):
    return nanvar(**locals()) ** 0.5
