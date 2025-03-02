import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

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

# Add weight difference feature
df["weight_difference"] = df["weight_kg"] - df["weight_kg"].mean()

# Define features and target variables
features = ["height_cm", "weight_kg", "weight_difference", "bicep", "calf", "chest", 
            "hip", "thigh", "waist"]  # Removed shoulder-breadth, shoulder-to-crotch

target = ["bicep", "calf", "chest", "hip", "thigh", "waist"]

# Drop missing values
df = df.dropna()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train separate Gradient Boosting models for weight-insensitive features
gb_models = {}
for measurement in ["bicep", "calf"]:
    gb_models[measurement] = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        subsample=0.7,
        random_state=42
    )
    gb_models[measurement].fit(X_train[["height_cm", "weight_kg", "weight_difference"]], y_train[measurement])

# Train separate Linear Regression models for weight-sensitive features
linear_models = {}
for measurement in ["chest", "hip", "thigh", "waist", "bicep", "calf"]:
    linear_models[measurement] = LinearRegression()
    linear_models[measurement].fit(X_train[["weight_difference"]], y_train[measurement])

# Evaluate models
mae_scores = []
r2_scores = []
for measurement in target:
    if measurement in ["chest", "hip", "thigh", "waist", "bicep", "calf"]:
        y_pred = linear_models[measurement].predict(X_test[["weight_difference"]])
    else:
        y_pred = gb_models[measurement].predict(X_test[["height_cm", "weight_kg", "weight_difference"]])
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
    Uses Linear Regression for all weight-sensitive features.
    """
    input_data = pd.DataFrame([user_input])
    input_data["weight_difference"] = target_weight - input_data["weight_kg"]
    
    # Ensure the input features match training features in order
    input_data = input_data[features]
    
    predicted_measurements = {}
    
    for measurement in target:
        predicted_measurements[measurement] = round(linear_models[measurement].predict([[input_data["weight_difference"].values[0]]])[0], 2)
    
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