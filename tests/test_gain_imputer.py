import unittest
import numpy as np
from sklearn.datasets import load_iris
from gain_imputer import GainImputer

class TestGainImputer(unittest.TestCase):
    def setUp(self):
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.nan_percentage = 0.5
        self.num_nans = int(np.prod(self.X.shape) * self.nan_percentage)
        self.nan_indices = np.random.choice(np.arange(np.prod(self.X.shape)), size=self.num_nans, replace=False)
        self.X_with_nans = self.X.flatten()
        self.X_with_nans[self.nan_indices] = np.nan
        self.X_with_nans = self.X_with_nans.reshape(self.X.shape)
        self.cat_columns = [0, 1]

    def test_fit_transform(self):
        gain_imputer = GainImputer(dim=self.X.shape[1], h_dim=128, cat_columns=self.cat_columns, batch_size=1024)
        imputed_X = gain_imputer.fit_transform(self.X_with_nans)
        self.assertEqual(imputed_X.shape, self.X.shape)

    def test_transform(self):
        gain_imputer = GainImputer(dim=self.X.shape[1], h_dim=128, cat_columns=self.cat_columns, batch_size=1024)
        gain_imputer.fit(self.X_with_nans)
        imputed_X = gain_imputer.transform(self.X_with_nans)
        self.assertEqual(imputed_X.shape, self.X.shape)

if __name__ == '__main__':
    unittest.main()
