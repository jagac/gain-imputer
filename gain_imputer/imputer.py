import logging
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from gain_imputer.gain_model import Gain
from gain_imputer.utils import GainUtilities

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class GainImputer:
    """
    Main class designed for imputing missing values in a dataset using the GAIN model
    :param dim: The total number of features or variables in your dataset. It represents the dimensionality of the data.
    :param h_dim: The dimensionality of the hidden layer in the GAIN model. It determines the capacity of the model.
    :param cat_columns: A list of indices representing the categorical columns in your dataset.
    :param batch_size: The size of mini-batches used during training. It controls how many samples are processed in each iteration of the optimization process.
    :param hint_rate: The probability of providing hints during training. Hints are used to guide the imputation process. It should be a value between 0 and 1.
    :param alpha:  A hyperparameter that balances the generator loss and mean squared error loss during training.
    :param iterations: The number of training iterations or epochs.
    """

    def __init__(
        self,
        dim: int,
        h_dim: int,
        cat_columns: Optional[List[int]] = None,
        batch_size: Optional[int] = 64,
        hint_rate: Optional[float] = 0.9,
        alpha: Optional[int] = 10,
        iterations: Optional[int] = 10000,
        verbose: Optional[bool] = True,
        show_progress: Optional[bool] = True,
    ) -> None:

        self.model = Gain(dim, h_dim)
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.norm_parameters: Optional[dict] = None
        self.cat_columns = cat_columns
        self.utilities = GainUtilities()
        self.verbose = verbose
        self.show_progress = show_progress

    def __str__(self):
        info = (
            f"batch_size: {self.batch_size}",
            f"hint_rate: {self.hint_rate}",
            f"alpha: {self.alpha}",
            f"iterations: {self.iterations}",
            f"cat_columns: {self.cat_columns}",
        )
        return "\n".join(info)

    def fit(self, data: np.ndarray) -> "GainImputer":
        """
        Fit the imputer
        :param data: data to be used for training
        :return: self
        """
        if(self.verbose):
            logger.info(f"Fitting with parameters:\n{self.__str__()}")
        data_m = 1 - np.isnan(data)
        no, dim = data.shape

        norm_data, norm_parameters = self.utilities.normalizer(data)
        norm_data_x = np.nan_to_num(norm_data, 0)

        data_x = torch.from_numpy(norm_data_x).float()
        data_m = torch.from_numpy(data_m).float()
        data_h = self.utilities.binary_sampler(self.hint_rate, no, dim)
        data_h = torch.from_numpy(data_m.numpy() * data_h).float()

        dataset = TensorDataset(data_x, data_m, data_h)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer_g = torch.optim.Adam(self.model.generator.parameters())
        optimizer_d = torch.optim.Adam(self.model.discriminator.parameters())

        with trange(self.iterations, disable=not self.show_progress) as t:
            for epoch in t:
                for batch_x, batch_m, batch_h in dataloader:
                    optimizer_d.zero_grad()
                    hat_x = batch_x * batch_m + self.model.generator(
                        batch_x, batch_m
                    ) * (1 - batch_m)
                    d_prob = self.model.discriminator(hat_x, batch_h)
                    d_loss_temp = -torch.mean(
                        batch_m * torch.log(d_prob + 1e-8)
                        + (1 - batch_m) * torch.log(1.0 - d_prob + 1e-8)
                    )
                    d_loss_temp.backward()
                    optimizer_d.step()

                    optimizer_g.zero_grad()
                    g_sample = self.model.generator(batch_x, batch_m)
                    hat_x = batch_x * batch_m + g_sample * (1 - batch_m)
                    d_prob = self.model.discriminator(hat_x, batch_h)
                    g_loss_temp = -torch.mean((1 - batch_m) * torch.log(d_prob + 1e-8))
                    mse_loss = torch.mean(
                        (batch_m * batch_x - batch_m * g_sample) ** 2
                    ) / torch.mean(batch_m)
                    g_loss = g_loss_temp + self.alpha * mse_loss
                    g_loss.backward()
                    optimizer_g.step()
                
                t.set_postfix({"d_loss": d_loss_temp.item(), "g_loss": g_loss.item()})

        self.norm_parameters = norm_parameters

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Imputes data based on the training done in the fit method
        :param data: data to be imputed
        :return: numpy array with imputed data
        """
        data_m = 1 - np.isnan(data)
        no, dim = data.shape
        if self.verbose:
            logger.info(f"Transforming data shapped {no, dim}")

        norm_data = self.utilities.renormalizer(data, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)

        data_x = torch.from_numpy(norm_data_x).float()
        data_m = torch.from_numpy(data_m).float()

        z_mb = self.utilities.uniform_sampler(0, 0.01, no, dim)
        m_mb = data_m
        x_mb = data_x
        x_mb = m_mb * x_mb + (1 - m_mb) * z_mb
        imputed_data = self.model.generator(x_mb.float(), m_mb.float()).detach().numpy()
        imputed_data = (
            data_m.numpy() * norm_data_x + (1 - data_m.numpy()) * imputed_data
        )

        imputed_data = self.utilities.categorical_col_rounder(
            imputed_data, data_x, self.cat_columns
        )

        return imputed_data.astype(np.float32)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Combination of both training and imputation in 1 method
        :param data: data to be transformed
        :return: numpy array of transformed data
        """
        self.fit(data)
        return self.transform(data)
