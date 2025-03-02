import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

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

# Compute derived features
df["weight_difference"] = df["weight_kg"] - df["weight_kg"].mean()
df["weight_diff_squared"] = df["weight_difference"] ** 2
df["height_weight_interaction"] = df["height_cm"] * df["weight_kg"]

# Apply feature scaling
scaler = StandardScaler()
df["scaled_weight_difference"], df["scaled_weight_diff_squared"], df["scaled_height_weight_interaction"] = scaler.fit_transform(
    df[["weight_difference", "weight_diff_squared", "height_weight_interaction"]]
).T

# Define features and target variables
xgb_features = ["height_cm", "weight_kg", "scaled_weight_difference", "scaled_weight_diff_squared", "scaled_height_weight_interaction"]
ridge_features = ["scaled_weight_difference", "scaled_weight_diff_squared", "scaled_height_weight_interaction"]
target = ["bicep", "calf", "chest", "hip", "thigh", "waist"]

# Drop missing values
df = df.dropna()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[xgb_features + target], df[target], test_size=0.2, random_state=42)

# Train XGBoost models
xgb_models = {}
for measurement in target:
    xgb_models[measurement] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_models[measurement].fit(X_train[xgb_features], y_train[measurement])

# Train Ridge Regression models
ridge_models = {}
for measurement in target:
    ridge_models[measurement] = Ridge(alpha=10.0)  # Increased regularization to prevent overfitting
    ridge_models[measurement].fit(X_train[ridge_features], y_train[measurement])

# Evaluate models
mae_scores = []
r2_scores = []
for measurement in target:
    y_pred = xgb_models[measurement].predict(X_test[xgb_features])
    mae = mean_absolute_error(y_test[measurement], y_pred)
    r2 = r2_score(y_test[measurement], y_pred)
    mae_scores.append(mae)
    r2_scores.append(r2)



# Save trained models
joblib.dump(xgb_models, "xgb_models.pkl")
joblib.dump(ridge_models, "ridge_models.pkl")

# Save feature list
joblib.dump(features, "features.pkl")
joblib.dump(target, "target.pkl")

# Evaluate models
mae_scores = []
r2_scores = []
for measurement in target:
    if measurement in ["chest", "hip", "thigh", "waist"]:
        y_pred = ridge_models[measurement].predict(X_test[["weight_difference", "weight_diff_squared"]])
    else:
        y_pred = xgb_models[measurement].predict(X_test[features])
    mae = mean_absolute_error(y_test[measurement], y_pred)
    r2 = r2_score(y_test[measurement], y_pred)
    mae_scores.append(mae)
    r2_scores.append(r2)

# Print average model performance
print(f"Model Performance:\nAverage MAE: {np.mean(mae_scores):.2f} cm\nAverage R² Score: {np.mean(r2_scores):.2f}")










# Print average model performance
print(f"Model Performance:\nAverage MAE: {np.mean(mae_scores):.2f} cm\nAverage R² Score: {np.mean(r2_scores):.2f}")

# Prediction Function
def predict_new_measurements(user_input, target_weight):
    """
    Predicts new body measurements based on user input and target weight.
    Uses XGBoost for final prediction.
    """
    input_data = pd.DataFrame([user_input])
    input_data["weight_difference"] = target_weight - input_data["weight_kg"]
    input_data["weight_diff_squared"] = input_data["weight_difference"] ** 2
    input_data["height_weight_interaction"] = input_data["height_cm"] * target_weight
    
    # Apply the pre-fitted scaler
    input_data["scaled_weight_difference"], input_data["scaled_weight_diff_squared"], input_data["scaled_height_weight_interaction"] = scaler.transform(
        input_data[["weight_difference", "weight_diff_squared", "height_weight_interaction"]]
    ).T
    
    # Ensure input features match training set
    input_data = input_data[xgb_features]
    
    predicted_measurements = {}
    
    for measurement in target:
        predicted_measurements[measurement] = round(xgb_models[measurement].predict(input_data)[0], 2)
    
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
