import unittest
import pytest
import numpy as np
import pandas as pd
import polars as pl
import polars.testing

from groupby.util import convert_array_inputs_to_dict, get_array_name




class TestArrayFunctions(unittest.TestCase):
    def test_get_array_name_with_numpy(self):
        # NumPy arrays don't have names
        arr = np.array([1, 2, 3])
        self.assertIsNone(get_array_name(arr))

    def test_get_array_name_with_pandas(self):
        # Pandas Series with name
        named_series = pd.Series([1, 2, 3], name="test_series")
        self.assertEqual(get_array_name(named_series), "test_series")

        named_series = pd.Series([1, 2, 3], name=0)
        self.assertEqual(get_array_name(named_series), 0)

        # Pandas Series without name
        unnamed_series = pd.Series([1, 2, 3])
        self.assertIsNone(get_array_name(unnamed_series))

        # Pandas Series with empty name
        empty_name_series = pd.Series([1, 2, 3], name="")
        self.assertIsNone(get_array_name(empty_name_series))

    def test_get_array_name_with_polars(self):
        # Polars Series with name
        named_series = pl.Series("test_series", [1, 2, 3])
        self.assertEqual(get_array_name(named_series), "test_series")

        # Polars Series with empty name
        empty_name_series = pl.Series("", [1, 2, 3])
        self.assertIsNone(get_array_name(empty_name_series))

    def test_convert_mapping_to_dict(self):
        # Test with dictionary
        input_dict = {"a": np.array([1, 2]), "b": np.array([3, 4])}
        result = convert_array_inputs_to_dict(input_dict)
        self.assertEqual(result, input_dict)

        # Test with other mapping types
        from collections import OrderedDict

        ordered_dict = OrderedDict([("x", np.array([1, 2])), ("y", np.array([3, 4]))])
        result = convert_array_inputs_to_dict(ordered_dict)
        self.assertEqual(result, dict(ordered_dict))

    def test_convert_list_to_dict(self):
        # Test with list of named arrays
        arrays = [
            pd.Series([1, 2, 3], name="first"),
            pd.Series([4, 5, 6], name="second"),
        ]
        expected = {"first": arrays[0], "second": arrays[1]}
        result = convert_array_inputs_to_dict(arrays)
        self.assertEqual(result.keys(), expected.keys())
        for k in expected:
            pd.testing.assert_series_equal(result[k], expected[k])

        # Test with list of unnamed arrays
        arrays = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        ]
        expected = {"arr_0": arrays[0], "arr_1": arrays[1]}
        result = convert_array_inputs_to_dict(arrays)
        self.assertEqual(result.keys(), expected.keys())
        for k in expected:
            np.testing.assert_array_equal(result[k], expected[k])

        # Test with mixed named and unnamed arrays
        arrays = [
            pd.Series([1, 2, 3], name="named"),
            np.array([4, 5, 6]),
            pd.Series([7, 8, 9]),
        ]
        expected = {"named": arrays[0], "arr_1": arrays[1], "arr_2": arrays[2]}
        result = convert_array_inputs_to_dict(arrays)
        self.assertEqual(result.keys(), expected.keys())

    def test_convert_numpy_array_to_dict(self):
        # Test with 1D numpy array
        arr = np.array([1, 2, 3])
        result = convert_array_inputs_to_dict(arr)
        self.assertEqual(len(result), 1)
        self.assertTrue("arr_0" in result)
        np.testing.assert_array_equal(result["arr_0"], arr)

        # Test with 2D numpy array (should return empty dict as per function logic)
        arr_2d = np.array([[1, 2], [3, 4]])
        result = convert_array_inputs_to_dict(arr_2d)
        self.assertEqual(list(result), ["arr_0", "arr_1"])

        np.testing.assert_array_equal(result["arr_0"], arr_2d[:, 0])

    def test_convert_series_to_dict(self):
        # Test with named pandas Series
        series = pd.Series([1, 2, 3], name="test_series")
        result = convert_array_inputs_to_dict(series)
        self.assertEqual(len(result), 1)
        self.assertTrue("test_series" in result)
        pd.testing.assert_series_equal(result["test_series"], series)

        # Test with unnamed pandas Series
        series = pd.Series([1, 2, 3])
        result = convert_array_inputs_to_dict(series)
        self.assertEqual(len(result), 1)
        self.assertTrue("arr_0" in result)
        pd.testing.assert_series_equal(result["arr_0"], series)

        # Test with polars Series
        series = pl.Series("polars_series", [1, 2, 3])
        result = convert_array_inputs_to_dict(series)
        self.assertEqual(len(result), 1)
        self.assertTrue("polars_series" in result)
        pl.testing.assert_series_equal(result["polars_series"], series)

    def test_convert_dataframe_to_dict(self):
        # Test with pandas DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = convert_array_inputs_to_dict(df)
        self.assertEqual(len(result), 2)
        self.assertTrue("a" in result and "b" in result)
        pd.testing.assert_series_equal(result["a"], df["a"])
        pd.testing.assert_series_equal(result["b"], df["b"])

        # Test with polars DataFrame
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = convert_array_inputs_to_dict(df)
        self.assertEqual(len(result), 2)
        self.assertTrue("x" in result and "y" in result)
        pl.testing.assert_series_equal(result["x"], df["x"])
        pl.testing.assert_series_equal(result["y"], df["y"])

    def test_unsupported_type(self):
        # Test with unsupported type
        with pytest.raises(TypeError, match="Input type <class 'int'> not supported"):
            convert_array_inputs_to_dict(123)


if __name__ == "__main__":
    unittest.main()
