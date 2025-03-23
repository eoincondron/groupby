from functools import cached_property
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn import linear_model as lm

from groupby.util import ArrayType1D


@dataclass
class GroupScatter:
    x: ArrayType1D
    y: ArrayType1D
    n_groups: int = 25
    filter: ArrayType1D = slice(None)
    sample_weight: ArrayType1D = None
    deg: int = 1
    fit_intercept: bool = True

    def __post_init__(self):
        self._x = np.asarray(self.x[self.filter])
        self._y = np.asarray(self.y[self.filter])
        null_filter = np.isnan(self._x) | np.isnan(self._y)
        if null_filter.any():
            self._x = self._x[null_filter]
            self._y = self._y[null_filter]

    @cached_property
    def _X(self):
        X = self._x[:, None]
        if self.deg > 1:
            X = X ** np.arange(1, self.deg + 1)
        return X

    def _calculate_bins(self):
        self.bins = pd.qcut(self._x, q=self.n_groups)
        self.y_means = pd.Series(self._y).groupby(self.bins, observed=True).mean()

    def _calculate_regression(self):
        # self.regression_coefs = np.polyfit(self._x, self._y, self.deg)
        # self.regression_poly = np.polynomial.Polynomial(self.regression_coefs[::-1])
        # self.regression_curve = Series(self.regression_poly(self._x).values, self._x.values)
        self.fit = fit = lm.LinearRegression(fit_intercept=self.fit_intercept).fit(
            X=self._X,
            y=self._y,
            sample_weight=self.sample_weight,
        )
        self.r_squared = fit.score(self._X, self._y)
        self.regression_curve = pd.Series(fit.predict(self._X), self._x)
        self.regression_coefs = [*fit.coef_, fit.intercept_]

    def plot(self, **plot_kwargs):
        bin_means = (self.bins.categories.right + self.bins.categories.left) / 2
        ax = pd.Series(self.y_means.values, bin_means).plot(**plot_kwargs, style="o")
        self.regression_curve.plot(style="-", ax=ax)

        return ax
