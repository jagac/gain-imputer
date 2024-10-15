import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gain_imputer.model import Gain
from gain_imputer.utils import Normalizer, Sampler


class GainImputer:
    def __init__(
        self,
        dim: int,
        h_dim: int,
        alpha: float = 100,
        batch_size: int = 128,
        iterations: int = 10000,
        hint_rate: float = 0.9,
        cat_cols: list = [],
    ):
        self.dim = dim
        self.h_dim = h_dim
        self.alpha = alpha
        self.batch_size = batch_size
        self.iterations = iterations
        self.hint_rate = hint_rate
        self.cat_cols = cat_cols
        self.model = Gain(dim, h_dim)
        self.normalizer = None
        self.sampler = None

    def fit(self, data: np.ndarray) -> "GainImputer":
        self.normalizer = Normalizer(data)
        self.sampler = Sampler(self.hint_rate, data)

        # Create the mask for missing values (1 if present, 0 if missing)
        data_m = 1 - np.isnan(data)
        norm_data, _ = self.normalizer.normalize(data)
        norm_data_x = np.nan_to_num(norm_data, 0)

        # Convert numpy arrays to torch tensors
        data_x = torch.from_numpy(norm_data_x).float()  # Normalized input data
        data_m = torch.from_numpy(data_m).float()  # Mask indicating observed entries
        data_h = self.sampler.binary_sampler(*data.shape)  # Hint vector
        data_h = torch.from_numpy(data_m.numpy() * data_h).float()

        # Create TensorDataset and DataLoader
        dataset = TensorDataset(data_x, data_m, data_h)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers for generator and discriminator
        optimizer_g = torch.optim.Adam(self.model.generator.parameters())
        optimizer_d = torch.optim.Adam(self.model.discriminator.parameters())

        # Training loop
        for epoch in range(self.iterations):
            for batch_x, batch_m, batch_h in dataloader:
                # Sample noise `z` from a uniform distribution
                z = torch.rand_like(batch_x)  # Random noise same size as batch_x

                # Generator produces imputed data using x, z, and m
                g_sample = self.model.generator(batch_x, z, batch_m)

                # Discriminator step
                optimizer_d.zero_grad()

                # Discriminator receives real data (batch_x), mask (batch_m), generated data (g_sample), and hint (batch_h)
                d_prob = self.model.discriminator(batch_x, batch_m, g_sample, batch_h)

                # Discriminator loss
                d_loss_temp = -torch.mean(
                    batch_m * torch.log(d_prob + 1e-8)
                    + (1 - batch_m) * torch.log(1.0 - d_prob + 1e-8)
                )

                # Backpropagation for discriminator
                d_loss_temp.backward()
                optimizer_d.step()

                # Generator step
                optimizer_g.zero_grad()

                # Generator re-samples and creates imputed data with both x, z, and m
                g_sample = self.model.generator(batch_x, z, batch_m)

                # Discriminator prediction for the imputed data
                d_prob = self.model.discriminator(batch_x, batch_m, g_sample, batch_h)

                # Generator loss encouraging it to fool the discriminator
                g_loss_temp = -torch.mean((1 - batch_m) * torch.log(d_prob + 1e-8))

                # Mean squared error loss for imputation accuracy
                mse_loss = torch.mean(
                    (batch_m * batch_x - batch_m * g_sample) ** 2
                ) / torch.mean(batch_m)

                # Total generator loss: adversarial loss + imputation accuracy (MSE)
                g_loss = g_loss_temp + self.alpha * mse_loss

                # Backpropagation for generator
                g_loss.backward()
                optimizer_g.step()

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Imputes data based on the training done in the fit method.
        :param data: data to be imputed
        :return: numpy array with imputed data
        """
        # Create a missing data mask
        data_m = 1 - np.isnan(data)
        no, dim = data.shape

        # Normalize the data
        norm_data, _ = self.normalizer.normalize(data)

        # Fill missing values with 0 for the imputation task
        norm_data_x = np.nan_to_num(norm_data, 0)

        # Convert to torch tensor
        data_x = torch.from_numpy(norm_data_x).float()
        data_m = torch.from_numpy(data_m).float()

        # Sample noise `z` for the missing values
        z_mb = self.sampler.uniform_sampler(0, 0.01, no, dim)
        z_mb = torch.from_numpy(z_mb).float()

        # Input to the generator: x, z, and m
        g_sample = self.model.generator(data_x, z_mb, data_m).detach().numpy()

        # Combine original data and imputed data
        imputed_data = data_m.numpy() * norm_data_x + (1 - data_m.numpy()) * g_sample

        # Denormalize the imputed data
        imputed_data = self.normalizer.denormalize(imputed_data)

        # Round the categorical columns, if any
        imputed_data = self.normalizer.categorical_col_rounder(
            imputed_data, data_x.numpy(), self.cat_cols
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
