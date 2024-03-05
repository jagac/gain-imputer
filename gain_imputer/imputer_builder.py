from typing import List

from .imputer import GainImputer


class GainImputerBuilder:
    """
    Builder for the gain imputer class
    :param dim: The total number of features or variables in your dataset. It represents the dimensionality of the data.
    :param h_dim: The dimensionality of the hidden layer in the GAIN model. It determines the capacity of the model.
    :param cat_columns: A list of indices representing the categorical columns in your dataset.
    :param batch_size: The size of mini-batches used during training. It controls how many samples are processed in each iteration of the optimization process.
    :param hint_rate: The probability of providing hints during training. Hints are used to guide the imputation process. It should be a value between 0 and 1.
    :param alpha:  A hyperparameter that balances the generator loss and mean squared error loss during training.
    :param iterations: The number of training iterations or epochs.
    """

    def __init__(self):
        self.dim = None
        self.h_dim = None
        self.cat_columns = None
        self.batch_size = 64
        self.hint_rate = 0.9
        self.alpha = 10
        self.iterations = 10000

    def with_dim(self, dim: int) -> "GainImputerBuilder":
        self.dim = dim
        return self

    def with_h_dim(self, h_dim: int) -> "GainImputerBuilder":
        self.h_dim = h_dim
        return self

    def with_cat_columns(self, cat_columns: List[int]) -> "GainImputerBuilder":
        self.cat_columns = cat_columns
        return self

    def with_batch_size(self, batch_size: int) -> "GainImputerBuilder":
        self.batch_size = batch_size
        return self

    def with_hint_rate(self, hint_rate: float) -> "GainImputerBuilder":
        self.hint_rate = hint_rate
        return self

    def with_alpha(self, alpha: int) -> "GainImputerBuilder":
        self.alpha = alpha
        return self

    def with_iterations(self, iterations: int) -> "GainImputerBuilder":
        self.iterations = iterations
        return self

    def build(self) -> "GainImputer":
        if self.dim is None or self.h_dim is None:
            raise ValueError(
                "Dimensionality (dim) and hidden dimension (h_dim) must be provided."
            )
        return GainImputer(
            self.dim,
            self.h_dim,
            self.cat_columns,
            self.batch_size,
            self.hint_rate,
            self.alpha,
            self.iterations,
        )
