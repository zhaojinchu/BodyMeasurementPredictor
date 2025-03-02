import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load Datasets
def load_and_merge_data(folder):
    hwg_metadata_path = f"{folder}/hwg_metadata.csv"
    measurements_path = f"{folder}/measurements.csv"
    hwg_metadata = pd.read_csv(hwg_metadata_path)
    measurements = pd.read_csv(measurements_path)
    return pd.merge(measurements, hwg_metadata, on="subject_id")

# Load and combine datasets from train, testA, and testB
df_train = load_and_merge_data("train")
df_testA = load_and_merge_data("testA")
df_testB = load_and_merge_data("testB")

# Combine all datasets into one
df = pd.concat([df_train, df_testA, df_testB], ignore_index=True)

# Remove female data
df = df[df['gender'] == 'male']
df = df.drop(columns=['gender', 'subject_id'])

# Add polynomial weight difference feature
df["weight_difference"] = df["weight_kg"] - df["weight_kg"].mean()
df["weight_diff_squared"] = df["weight_difference"] ** 2

# Define features and target variables
features = ["height_cm", "weight_kg", "weight_difference", "weight_diff_squared", "bicep", "calf", "chest", "hip", "thigh", "waist"]
target = ["bicep", "calf", "chest", "hip", "thigh", "waist"]

# Drop missing values
df = df.dropna()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train separate RandomForest models for weight-insensitive features (replacing XGBoost)
rf_models = {}
for measurement in ["bicep", "calf"]:
    rf_models[measurement] = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_models[measurement].fit(X_train[["height_cm", "weight_kg", "weight_difference", "weight_diff_squared"]], y_train[measurement])

# Train separate Ridge Regression models for weight-sensitive features
ridge_models = {}
for measurement in ["chest", "hip", "thigh", "waist", "bicep", "calf"]:
    ridge_models[measurement] = Ridge(alpha=0.5)
    ridge_models[measurement].fit(X_train[["weight_difference", "weight_diff_squared"]], y_train[measurement])

# Save trained models
joblib.dump(rf_models, "rf_models.pkl")
joblib.dump(ridge_models, "ridge_models.pkl")
joblib.dump(features, "features.pkl")
joblib.dump(target, "target.pkl")

# Evaluate models
mae_scores = []
r2_scores = []
for measurement in target:
    if measurement in ["chest", "hip", "thigh", "waist"]:
        y_pred = ridge_models[measurement].predict(X_test[["weight_difference", "weight_diff_squared"]])
    else:
        y_pred = rf_models[measurement].predict(X_test[["height_cm", "weight_kg", "weight_difference", "weight_diff_squared"]])
    mae = mean_absolute_error(y_test[measurement], y_pred)
    r2 = r2_score(y_test[measurement], y_pred)
    mae_scores.append(mae)
    r2_scores.append(r2)

# Print average model performance
print(f"Model Performance:\nAverage MAE: {np.mean(mae_scores):.2f} cm\nAverage R² Score: {np.mean(r2_scores):.2f}")
