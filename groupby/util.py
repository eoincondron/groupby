import unittest
import numpy as np
import pandas as pd
import polars as pl
import polars.testing
from typing import Union, Mapping

ArrayType1D = Union[np.ndarray, pl.Series, pd.Series, pd.Index, pd.Categorical]
ArrayType2D = Union[np.ndarray, pl.DataFrame, pd.DataFrame, pd.MultiIndex]


def get_array_name(array: Union[np.ndarray, pd.Series, pl.Series]):
    name = getattr(array, 'name', None)
    if name is None or name == '':
        return None
    return name


def convert_array_inputs_to_dict(arrays) -> dict:
    if isinstance(arrays, Mapping):
        return dict(arrays)
    elif isinstance(arrays, (tuple, list)):
        names = map(get_array_name, arrays)
        keys = [name or f'arr_{i}' for i, name in enumerate(names)]
        return dict(zip(keys, arrays))  # Fixed: arrays instead of array
    elif isinstance(arrays, np.ndarray) and arrays.ndim == 2:
        return convert_array_inputs_to_dict(list(arrays.T))
    elif isinstance(arrays, (pd.Series, pl.Series, np.ndarray, pd.Index, pd.Categorical)):
        name = get_array_name(arrays)
        return {name if name is not None else 'arr_0': arrays}
    elif isinstance(arrays, (pl.DataFrame, pd.DataFrame)):
        return {key: arrays[key] for key in arrays.columns}
    else:
        raise TypeError(f"Input type {type(arrays)} not supported")


import functools
import pandas as pd
from typing import Callable, Any, TypeVar, cast

F = TypeVar('F', bound=Callable[..., Any])


def check_data_inputs_aligned(*args_to_check, check_index: bool = True) -> Callable[[F], F]:
    """
    Factory function that returns a decorator which ensures all arguments passed to the
    decorated function have equal length and, if pandas objects and check_index is True,
    share a common index.

    Args:
        check_index: If True, also checks that pandas objects share the same index

    Returns:
        A decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not args:
                return func(*args, **kwargs)

            # Extract args that have a length
            if args_to_check is None:
                sized_args = [arg for arg in args if hasattr(arg, '__len__')]
            else:
                sized_args = args_to_check
            if not sized_args:
                return func(*args, **kwargs)

            # Check all args have the same length
            first_arg = sized_args[0]
            first_len = len(first_arg)

            for arg in sized_args[1:]:
                if arg is not None and len(arg) != first_len:
                    raise ValueError(f"All arguments must have equal length. "
                                     f"Expected length {first_len}, got {len(arg)}")

            # Check pandas objects share the same index
            if check_index:
                pandas_args = [arg for arg in args if isinstance(arg, (pd.Series, pd.DataFrame))]
                if pandas_args:
                    first_index = pandas_args[0].index
                    for arg in pandas_args[1:]:
                        if not first_index.equals(arg.index):
                            raise ValueError("All pandas objects must share the same index")

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


import concurrent.futures
from typing import TypeVar, Callable, List, Any, Optional

T = TypeVar('T')
R = TypeVar('R')


def parallel_map(func: Callable[[T], R], items: List[T], max_workers: Optional[int] = None) -> List[R]:
    """
    Apply a function to each item in a list in parallel using concurrent.futures.

    Args:
        func: The function to apply to each item
        items: List of items to process
        max_workers: Maximum number of worker threads or processes (None = auto)

    Returns:
        List of results in the same order as the input items

    Example:
        >>> def square(x):
        ...     return x * x
        >>> parallel_map(square, [1, 2, 3, 4, 5])
        [1, 4, 9, 16, 25]
    """
    # Choose between ProcessPoolExecutor and ThreadPoolExecutor based on your needs
    # ProcessPoolExecutor is better for CPU-bound tasks
    # ThreadPoolExecutor is better for I/O-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and store the future objects
        future_to_index = {executor.submit(func, item): i for i, item in enumerate(items)}

        # Collect results in the original order
        results = [None] * len(items)

        # Process completed futures as they finish
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                print(f"Item at index {index} generated an exception: {exc}")
                raise

    return results
