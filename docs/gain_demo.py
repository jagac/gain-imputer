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


