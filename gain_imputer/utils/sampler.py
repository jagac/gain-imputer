from typing import Optional

import numpy as np


class Sampler:
    """
    Initializes the Sampler class with hint rate and data.

    :param hint_rate: Probability threshold for binary sampling.
    :param data: 2D array used to define the shape for uniform sampling.
    """

    def __init__(self, hint_rate: float, data: np.ndarray) -> None:
        if not 0.0 <= hint_rate <= 1.0:
            raise ValueError("Hint rate must be between 0.0 and 1.0.")
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        self.hint_rate = hint_rate
        self.n_rows, self.dim = data.shape

    def binary_sampler(self, rows: int, cols: int) -> np.ndarray:
        """
        Samples binary random variables based on the hint rate.

        :param rows: Number of rows for the output binary matrix.
        :param cols: Number of columns for the output binary matrix.
        :return: Binary numpy array of shape (rows, cols).
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("Number of rows and columns must be positive integers.")

        unif_random_matrix = np.random.uniform(0.0, 1.0, size=(rows, cols))
        binary_random_matrix = (unif_random_matrix < self.hint_rate).astype(int)
        return binary_random_matrix

    def uniform_sampler(
        self,
        low: float,
        high: float,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
    ) -> np.ndarray:
        """
        Samples uniform random variables between specified bounds.

        :param low: Lower bound for uniform sampling.
        :param high: Upper bound for uniform sampling.
        :param rows: Number of rows for the output (default is based on input data shape).
        :param cols: Number of columns for the output (default is based on input data shape).
        :return: Uniform numpy array of shape (rows, cols).
        """
        rows = rows if rows is not None else self.n_rows
        cols = cols if cols is not None else self.dim

        if low >= high:
            raise ValueError("Lower bound must be less than upper bound.")
        if rows <= 0 or cols <= 0:
            raise ValueError("Number of rows and columns must be positive integers.")

        return np.random.uniform(low, high, size=(rows, cols))
