from typing import Callable, List
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
    except AttributeError:
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


def groupby_method(method):

    @wraps(method)
    def wrapper(
        group_key: ArrayCollection, values: ArrayCollection, mask: ArrayType1D = None
    ):
        if not isinstance(group_key, GroupBy):
            group_key = GroupBy(group_key)
        return method(group_key, values, mask)

    __doc__ = f"""
    Calculate the group-wise {method.__name__} of the given values over the groups defined by `key`
    
    Parameters
    ----------
    key: An array/Series or a container of same, such as dict, list or DataFrame
        Defines the groups. May be a single dimension like an array or Series, 
        or multi-dimensional like a list/dict of 1-D arrays or 2-D array/DataFrame. 
    values: An array/Series or a container of same, such as dict, list or DataFrame
        The values to be aggregated. May be a single dimension like an array or Series, 
        or multi-dimensional like a list/dict of 1-D arrays or 2-D array/DataFrame. 
    mask: array/Series
        Optional Boolean array which filters elements out of the calculations
        
    Returns
    -------
    pd.Series / pd.DataFrame
    
    The result of the group-by calculation. 
    A Series is returned when `values` is a single array/Series, otherwise a DataFrame. 
    The index of the result has one level per array/column in the group key. 
        
    """
    wrapper.__doc__ = __doc__

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

    def _apply_gb_func(
        self,
        func: Callable,
        values: ArrayCollection,
        mask: ArrayType1D = None,
        transform: bool = False,
    ):
        value_dict = convert_array_inputs_to_dict(values)
        np_values = list(map(val_to_numpy, value_dict.values()))
        results = map(
            lambda v: func(
                group_key=self.group_ikey, values=v, mask=mask, ngroups=self.ngroups
            ),
            np_values,
        )
        out_dict = {}
        for key, (result, seen) in zip(value_dict, results):
            if transform:
                result = out_dict[key] = pd.Series(
                    result[self.group_ikey], self._group_df.index
                )
            else:
                result = out_dict[key] = pd.Series(
                    result[seen], self.result_index[seen]
                )

            return_1d = len(value_dict) == 1 and isinstance(values, ArrayType1D)
            if return_1d:
                out = result
                if get_array_name(values) is None:
                    out.name = None
            else:
                out = pd.DataFrame(out_dict)

        if not transform and mask is not None:
            count = self.sum(values=mask)
            out = out.loc[count > 0]

        return out

    @groupby_method
    def count(
        self, values: ArrayCollection, mask: ArrayType1D = None, transform: bool = False
    ):
        return self._apply_gb_func(
            group_count, values=values, mask=mask, transform=transform
        )

    @groupby_method
    def sum(
        self, values: ArrayCollection, mask: ArrayType1D = None, transform: bool = False
    ):
        return self._apply_gb_func(
            group_sum, values=values, mask=mask, transform=transform
        )

    @groupby_method
    def mean(
        self, values: ArrayCollection, mask: ArrayType1D = None, transform: bool = False
    ):
        return self._apply_gb_func(
            group_mean, values=values, mask=mask, transform=transform
        )

    @groupby_method
    def min(
        self, values: ArrayCollection, mask: ArrayType1D = None, transform: bool = False
    ):
        return self._apply_gb_func(
            group_min, values=values, mask=mask, transform=transform
        )

    @groupby_method
    def max(
        self, values: ArrayCollection, mask: ArrayType1D = None, transform: bool = False
    ):
        return self._apply_gb_func(
            group_max, values=values, mask=mask, transform=transform
        )

    @groupby_method
    def agg(
        self,
        values: ArrayCollection,
        agg_func: str | List[str],
        mask: ArrayType1D = None,
        transform: bool = False,
    ):
        if np.ndim(agg_func) == 0:
            if isinstance(agg_func, Callable):
                agg_func = agg_func.__name__
            func = getattr(self, agg_func)
            return func(values, mask=mask, transform=transform)
        elif np.ndim(agg_func) == 1:
            values = convert_array_inputs_to_dict(values)
            if len(agg_func) != len(values):
                raise ValueError(
                    f"Mismatch between number of agg funcs ({len(agg_func)}) "
                    f"and number of values ({len(values)})"
                )
            return pd.concat(
                [
                    self.agg(v, agg_func=f, mask=mask, transform=transform)
                    for f, v in zip(agg_func, values)
                ],
                axis=1,
            )
        else:
            raise ValueError

    @groupby_method
    def ratio(
        self,
        values1: ArrayCollection,
        values2: ArrayCollection,
        mask: ArrayType1D = None,
        agg_func="sum",
    ):
        # check for nullity
        self.agg(values1, agg_func, mask) / self.agg(values2, agg_func, mask)
