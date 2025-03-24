
from typing import Callable

import numba as nb
import numpy as np

from groupby.util import check_data_inputs_aligned, ArrayType1D

MIN_INT = np.iinfo(np.int64).min


@nb.njit
def isnull_float(x):
    return x != x

@nb.njit
def isnull_int(x):
    return x == MIN_INT

@nb.njit
def add_one(x, y):
    return x + 1

@nb.njit
def nb_min(x, y):
    return x if x <= y else y

@nb.njit
def nb_max(x, y):
    return x if x >= y else y

@nb.njit
def nb_sum(x, y):
    return x + y


@nb.njit(parallel=False, nogil=False)
def _group_by_iterator(
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    reduce_func: Callable,
    is_null: Callable,
):
    for i in nb.prange(len(group_key)):
        key = group_key[i]
        val = values[i]
        if is_null(val) or (len(mask) and not mask[i]):
            continue

        target[key] = reduce_func(target[key], val)

    return target


def _prepare_mask_for_numba(mask):
    if mask is None:
        mask = np.array([])
    else:
        mask = np.asarray(mask)
        if mask.dtype.kind != "b":
            raise TypeError(f"mask must of Boolean type. Got {mask.dtype}")
    return mask


@check_data_inputs_aligned("group_key", "values", "mask")
def _group_func_wrap(
    reduce_func: Callable,
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    initial_value: int | float,
    mask: ArrayType1D = None,
):
    if values.dtype.kind == "f":
        out_type = values.dtype
        null_check = isnull_float
    else:
        out_type = int
        null_check = isnull_int
        if abs(initial_value) == np.inf:
            initial_value = abs(MIN_INT) * int(np.sign(initial_value))

    target = np.full(ngroups, initial_value, dtype=out_type)
    mask = _prepare_mask_for_numba(mask)
    values = np.asarray(values)

    return _group_by_iterator(
        group_key=group_key,
        values=values,
        target=target,
        mask=mask,
        reduce_func=reduce_func,
        is_null=null_check,
    )


@check_data_inputs_aligned("group_key", "values", "mask")
def group_count(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    return _group_func_wrap(add_one, initial_value=0, **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_sum(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    return _group_func_wrap(nb_sum, initial_value=0, **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_mean(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    kwargs = locals().copy()
    return group_sum(**kwargs) / group_count(**kwargs)


@check_data_inputs_aligned("group_key", "values", "mask")
def group_min(
        group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    return _group_func_wrap(nb_min, initial_value=np.inf, **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_max(
    group_key: ArrayType1D, values: ArrayType1D, ngroups: int, mask: ArrayType1D = None
):
    return _group_func_wrap(nb_max, initial_value=-np.inf, **locals())


