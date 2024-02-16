# Training can take a while so you can also save the trained model for future use

import joblib

# Train the GainImputer
gain_imputer = GainImputer(dim=X_with_nans.shape[1], h_dim=128, cat_columns=cat_columns)
gain_imputer.fit(X_train)

# Save the trained GainImputer
joblib.dump(gain_imputer, "trained_gain_imputer.pkl")

# Later, to load the trained GainImputer
loaded_gain_imputer = joblib.load("trained_gain_imputer.pkl")

# Use the loaded GainImputer for transformation
imputed_gain_test = loaded_gain_imputer.transform(X_test)
