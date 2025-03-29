
from functools import partial

import numba as nb
import numpy as np
import pandas as pd
from pandas.core import nanops
from .numba import NumbaReductionOps, is_null


@nb.njit(parallel=True)
def _reduce(arr, reduce_func, skipna: bool = True):
    out = 0
    if skipna:
        for i in nb.prange(len(arr)):
            x = arr[i]
            if not is_null(x):
                out += reduce_func(out, x)
    else:
        for i in nb.prange(len(arr)):
            x = arr[i]
            out += reduce_func(out, x)
    return out


class NumbaArrayReductions:

    @staticmethod
    @nb.njit
    def sum(arr: np.ndarray, skipna: bool) -> int | float:
         return _reduce(arr, reduce_func=NumbaReductionOps.sum, skipna=skipna)


@nb.njit
def sum(arr: np.ndarray, skipna: bool = True) -> int | float:
     return _reduce(arr, reduce_func=NumbaReductionOps.sum, skipna=skipna)


@nb.njit(parallel=True)
def apply_2d(array_reduce: "Callable", arr, axis=0, skipna: bool = True):
    if axis == 0:
        arr = arr.T
    out = np.zeros(len(arr))
    for col in nb.prange(len(arr)):
        out[col] = array_reduce(arr[col], skipna=True)
    return out


def _apply_op(arr, func_name: str, skipna=True, min_count = 0, axis = None):
    if min_count != 0:
        return getattr(nanops, f'nan{func_name}')(**locals())

    array_reduce = getattr(NumbaArrayReductions, func_name)
    if arr.ndim == 1:
        return array_reduce(arr, skipna=skipna)
    else:
        if axis is None:
            # warn
            axis = 0
        return apply_2d(array_reduce, arr, axis=axis)
