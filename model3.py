import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Load Dataset
hwg_metadata_path = "train/hwg_metadata.csv"
measurements_path = "train/measurements.csv"

hwg_metadata = pd.read_csv(hwg_metadata_path)
measurements = pd.read_csv(measurements_path)

# Merge datasets on subject_id
df = pd.merge(measurements, hwg_metadata, on="subject_id")

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

# Train separate XGBoost models for weight-insensitive features
xgb_models = {}
for measurement in ["bicep", "calf"]:
    xgb_models[measurement] = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    xgb_models[measurement].fit(X_train[["height_cm", "weight_kg", "weight_difference", "weight_diff_squared"]], y_train[measurement])

# Train separate Ridge Regression models for weight-sensitive features
ridge_models = {}
for measurement in ["chest", "hip", "thigh", "waist", "bicep", "calf"]:
    ridge_models[measurement] = Ridge(alpha=0.5)
    ridge_models[measurement].fit(X_train[["weight_difference", "weight_diff_squared"]], y_train[measurement])

# Evaluate models
mae_scores = []
r2_scores = []
for measurement in target:
    if measurement in ["chest", "hip", "thigh", "waist", "bicep", "calf"]:
        y_pred = ridge_models[measurement].predict(X_test[["weight_difference", "weight_diff_squared"]])
    else:
        y_pred = xgb_models[measurement].predict(X_test[["height_cm", "weight_kg", "weight_difference", "weight_diff_squared"]])
    mae = mean_absolute_error(y_test[measurement], y_pred)
    r2 = r2_score(y_test[measurement], y_pred)
    mae_scores.append(mae)
    r2_scores.append(r2)

# Print average model performance
print(f"Model Performance:\nAverage MAE: {np.mean(mae_scores):.2f} cm\nAverage R² Score: {np.mean(r2_scores):.2f}")

# Prediction Function
def predict_new_measurements(user_input, target_weight):
    """
    Predicts new body measurements based on user input and target weight.
    Uses Ridge Regression for all weight-sensitive features.
    """
    input_data = pd.DataFrame([user_input])
    input_data["weight_difference"] = target_weight - input_data["weight_kg"]
    input_data["weight_diff_squared"] = input_data["weight_difference"] ** 2
    
    # Ensure the input features match training features in order
    input_data = input_data[features]
    
    predicted_measurements = {}
    
    for measurement in target:
        predicted_measurements[measurement] = round(
            ridge_models[measurement].predict(input_data[["weight_difference", "weight_diff_squared"]])[0], 2
        )
    
    return predicted_measurements

# Example User Input
user_input = {
    "height_cm": 175,
    "weight_kg": 80,
    "bicep": 33,
    "calf": 38,
    "chest": 102,
    "hip": 108,
    "thigh": 57,
    "waist": 92
}

# Example: Predict measurements at user-input weight
target_weight = float(input("Enter the new body weight (kg): "))
new_measurements = predict_new_measurements(user_input, target_weight)
print("Predicted new body measurements:", new_measurements)

