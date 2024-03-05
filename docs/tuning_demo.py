# Some hyperparameters is hard to nail down so this demo shows you how to use optuna and gain to
# achieve the best results

import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split

from gain_imputer import GainImputer

# Load sample data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Introduce missing data
nan_percentage = 0.5
num_nans = int(np.prod(X.shape) * nan_percentage)
nan_indices = np.random.choice(
    np.arange(np.prod(X.shape)), size=num_nans, replace=False
)
X_flat = X.flatten()
X_flat[nan_indices] = np.nan
X_with_nans = X_flat.reshape(X.shape)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_with_nans, y, test_size=0.2, random_state=42
)


def objective(trial):
    # Define parameters for GainImputer and RandomForestClassifier
    h_dim = trial.suggest_int("h_dim", 32, 256)
    cat_columns = [0, 1]  # Assuming categorical columns
    batch_size = 2028
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 3, 10)

    # Train GainImputer
    gain_imputer = GainImputer(
        dim=X_train.shape[1],
        h_dim=h_dim,
        cat_columns=cat_columns,
        batch_size=batch_size,
    )
    gain_imputer.fit(X_train)
    imputed_X_train = gain_imputer.transform(X_train)

    # Train RandomForestClassifier
    regressor = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in kf.split(imputed_X_train):
        X_train_fold, X_val_fold = imputed_X_train[train_idx], imputed_X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        regressor.fit(X_train_fold, y_train_fold)
        pred = regressor.predict(X_val_fold)
        f1_scores.append(f1_score(pred, y_val_fold))

    return np.mean(f1_scores)


# Create a study object and optimize hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Get best hyperparameters
best_h_dim = study.best_params["h_dim"]
best_n_estimators = study.best_params["n_estimators"]
best_max_depth = study.best_params["max_depth"]

print("Best hyperparameters:")
print("h_dim:", best_h_dim)
print("n_estimators:", best_n_estimators)
print("max_depth:", best_max_depth)
