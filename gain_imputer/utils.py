from typing import List, Optional, Tuple

import numpy as np
import torch


class GainUtilities:
    """
    The class encapsulates the training and imputation process, providing a convenient interface for handling missing data in a given dataset
    """

    @staticmethod
    def binary_sampler(p: float, rows: int, cols: int) -> np.ndarray:
        """
        Sample binary random variables
        :param p: hint rate
        :param rows: number of rows
        :param cols: number of columns
        :return: numpy array
        """
        unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])
        binary_random_matrix = 1 * (unif_random_matrix < p)
        return binary_random_matrix

    @staticmethod
    def uniform_sampler(low: float, high: float, rows: int, cols: int) -> np.ndarray:
        """
        Sample uniform random variables
        :param low: lower bound
        :param high: upper bound
        :param rows: number of rows
        :param cols: number of columns
        :return: numpy array
        """
        return np.random.uniform(low, high, size=[rows, cols])

    @staticmethod
    def normalizer(
        data: np.ndarray, parameters: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Normalize the data [0, 1]
        :param data: original data
        :param parameters: min and max for each feature
        :return: normalized data and params
        """
        _, dim = data.shape
        norm_data = data.copy()

        if parameters is None:
            min_val = np.zeros(dim)
            max_val = np.zeros(dim)

            for i in range(dim):
                min_val[i] = np.nanmin(norm_data[:, i])
                norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
                max_val[i] = np.nanmax(norm_data[:, i])
                norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

            norm_parameters = {"min_val": min_val, "max_val": max_val}
        else:
            min_val = parameters["min_val"]
            max_val = parameters["max_val"]

            for i in range(dim):
                norm_data[:, i] = norm_data[:, i] - min_val[i]
                norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

            norm_parameters = parameters

        return norm_data, norm_parameters

    @staticmethod
    def renormalizer(norm_data: np.ndarray, norm_parameters: dict) -> np.ndarray:
        """
        Renormalize data from [0, 1] range to the original range
        :param norm_data: normalized data
        :param norm_parameters: min and max for each feature gotten from normalizer method
        :return: renormalized numpy array
        """
        min_val = norm_parameters["min_val"]
        max_val = norm_parameters["max_val"]

        _, dim = norm_data.shape
        renorm_data = norm_data.copy()

        for i in range(dim):
            renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
            renorm_data[:, i] = renorm_data[:, i] + min_val[i]

        return renorm_data

    @staticmethod
    def categorical_col_rounder(
        imputed_data: np.ndarray,
        data_x: torch.Tensor,
        cat_columns: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Rounds categorical columns
        :param imputed_data: imputed data numpy array
        :param data_x: normalized data numpy array
        :param cat_columns: indices of categorical columns
        :return: rounded data numpy array
        """
        _, dim = data_x.shape
        rounded_data = imputed_data.copy()

        for i in range(dim):
            if cat_columns is not None and i in cat_columns:
                rounded_data[:, i] = np.round(rounded_data[:, i])

        return rounded_data
