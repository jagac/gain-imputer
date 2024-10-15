from typing import List, Optional, Tuple

import numpy as np


class Normalizer:
    def __init__(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        _, self.dim = data.shape
        self.parameters = self._calculate_parameters(data)

    def _calculate_parameters(self, data: np.ndarray) -> dict:
        """Calculates min and max values for each dimension in the data."""
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return {"min_val": min_val, "max_val": max_val}

    def normalize(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Normalizes the data using the stored parameters."""
        if data.shape[1] != self.dim:
            raise ValueError("Input data dimensions do not match the initialized data.")

        norm_data = (data - self.parameters["min_val"]) / (
            self.parameters["max_val"] + 1e-6
        )
        return norm_data, self.parameters

    def denormalize(self, norm_data: np.ndarray) -> np.ndarray:
        """Denormalizes the normalized data back to the original scale."""
        if norm_data.shape[1] != self.dim:
            raise ValueError("Input normalized data dimensions do not match.")

        renorm_data = (
            norm_data * (self.parameters["max_val"] + 1e-6)
        ) + self.parameters["min_val"]
        return renorm_data

    @staticmethod
    def categorical_col_rounder(
        imputed_data: np.ndarray,
        data_x: np.ndarray,
        cat_columns: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Rounds categorical columns in the imputed data."""
        if imputed_data.shape != data_x.shape:
            raise ValueError("Shape of imputed_data and data_x must be the same.")

        rounded_data = imputed_data.copy()
        if cat_columns is not None:
            rounded_data[:, cat_columns] = np.round(rounded_data[:, cat_columns])
        return rounded_data
