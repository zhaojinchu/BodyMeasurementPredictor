import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

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

# Define features and target variables
features = ["height_cm", "weight_kg", "ankle", "arm-length", "bicep", "calf", 
            "chest", "forearm", "hip", "leg-length", "shoulder-breadth", 
            "shoulder-to-crotch", "thigh", "waist", "wrist"]

target = ["ankle", "arm-length", "bicep", "calf", "chest", "forearm", "hip", 
          "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", "waist", "wrist"]

# Drop missing values
df = df.dropna()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Model
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nMean Absolute Error: {mae:.2f} cm\nR² Score: {r2:.2f}")

# Prediction Function
def predict_new_measurements(user_input, target_weight):
    """
    Predicts new body measurements based on user input and target weight.
    user_input: dict with current measurements and current weight.
    target_weight: float (new desired weight).
    """
    input_data = pd.DataFrame([user_input])
    input_data["weight_kg"] = target_weight  # Replace with target weight
    predicted_measurements = rf_model.predict(input_data)
    result = {target[i]: round(predicted_measurements[0][i], 2) for i in range(len(target))}
    return result

# Example User Input
user_input = {
    "height_cm": 175,
    "weight_kg": 80,
    "ankle": 24,
    "arm-length": 52,
    "bicep": 33,
    "calf": 38,
    "chest": 102,
    "forearm": 27,
    "hip": 108,
    "leg-length": 81,
    "shoulder-breadth": 36,
    "shoulder-to-crotch": 67,
    "thigh": 57,
    "waist": 92,
    "wrist": 16
}

# Example: Predict measurements at 75kg

target_weight = input("Enter the new body weight (kg): ")
new_measurements = predict_new_measurements(user_input, target_weight)
print("Predicted new body measurements:", new_measurements)
