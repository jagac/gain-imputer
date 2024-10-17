
# GAIN Imputer (Unstable)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A python library for imputing missing values in a dataset using GANs. It provides a convenient implementation to handle missing data, allowing users to train the imputer and apply it to their data.

This is an adaptation of the work done: https://github.com/jsyoon0823/GAIN implemented in torch and provides a familiar workflow to sklearn. 


## Installation

```bash
$ pip install git+https://github.com/jagac/gain-imputer
```
## Minimal Example
Gain requires indices of categorical columns to be provided as a list in order to round them properly. Check out for more demos: https://github.com/Jagac/gain-imputer/tree/main/docs

```python
from gain_imputer import GainImputer
cat_columns = [0,1,2]
gain_imputer = GainImputer(
    dim=data_with_nans.shape[1], h_dim=128, cat_columns=cat_columns
)
imputed_data = gain_imputer.fit_transform(data_with_nans)
```

## Full example (CPU)

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from gain_imputer import GainImputer
from sklearn.metrics import accuracy_score, f1_score

# sample data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# introducing missing data
nan_percentage = 0.5
num_nans = int(np.prod(X.shape) * nan_percentage)
nan_indices = np.random.choice(
    np.arange(np.prod(X.shape)), size=num_nans, replace=False
)
X_flat = X.flatten()
X_flat[nan_indices] = np.nan

X_with_nans = X_flat.reshape(X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X_with_nans, y, test_size=0.2, random_state=42
)

# gain needs to know the indices of categorical columns in order to round them
cat_columns = [0, 1]
gain_imputer = GainImputer(
    dim=X_with_nans.shape[1], h_dim=128, cat_columns=cat_columns, batch_size=2028
)

# simple fit_transform
imputed_gain_train = gain_imputer.fit_transform(X_train) 
# also can use
# gain_imputer.fit(X_train) 
# gain_imputer.transform(X_train)
imputed_gain_test = gain_imputer.transform(X_test) 

# model training
regressor = RandomForestClassifier(n_jobs=-1, random_state=42)
regressor.fit(imputed_gain_train, y_train)
pred = regressor.predict(imputed_gain_test)
print(accuracy_score(pred, y_test))
print(f1_score(pred, y_test))

```
## Saving for future use
```python
import joblib

# Train the GainImputer
gain_imputer = GainImputer(dim=X_with_nans.shape[1], h_dim=128, cat_columns=cat_columns)
gain_imputer.fit(X_train)

# Save the trained GainImputer
joblib.dump(gain_imputer, 'trained_gain_imputer.pkl')

# Later, to load the trained GainImputer
loaded_gain_imputer = joblib.load('trained_gain_imputer.pkl')

# Use the loaded GainImputer for transformation
imputed_gain_test = loaded_gain_imputer.transform(X_test)
```

## Using a builder
```python
from gain_imputer import GainImputerBuilder

cat_columns = [0, 1]
builder = (
    GainImputerBuilder()
    .with_dim(X_with_nans.shape[1])
    .with_h_dim(128)
    .with_cat_columns(cat_columns)
    .with_batch_size(2028)
    .build()
)
print(builder)
```

## Reference 

dim: The total number of features or variables in your dataset.

h_dim: The dimensionality of the hidden layer in the GAIN model.

cat_columns: A list of indices representing the categorical columns in your dataset.

batch_size: The size of mini-batches used during training.

hint_rate: The probability of providing hints during training.

alpha: A hyperparameter that balances the generator loss and mean squared error loss during training.

iterations: The number of training iterations or epochs.

