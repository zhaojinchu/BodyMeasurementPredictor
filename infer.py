import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib  # Load the scaler

# Load the correct scalers
scaler_input = joblib.load("scaler_input.pkl")
scaler_output = joblib.load("scaler_output.pkl")  # Load the correct output scaler

# Define input/output features
input_features = [
    'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm', 'height', 'hip',
    'leg-length', 'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist',
    'height_cm', 'weight_kg', 'new_weight'  # Ensure new_weight is included
]

output_features = [
    'ankle', 'bicep', 'calf', 'chest', 'forearm', 'hip',
    'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
]

# Define model class
class BodyMeasurementNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(BodyMeasurementNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256, track_running_stats=True)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128, track_running_stats=True)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load model
input_size = len(input_features)
output_size = len(output_features)
model_nn = BodyMeasurementNN(input_size, output_size)
model_nn.load_state_dict(torch.load("best_model.pth"))
model_nn.eval()  # Set model to evaluation mode

# Example test dictionary
test_input = {
    'ankle': 23.5, 'arm-length': 52.0, 'bicep': 32.0, 'calf': 37.0, 'chest': 105.0,
    'forearm': 27.0, 'height': 180.0, 'hip': 100.0, 'leg-length': 82.0,
    'shoulder-breadth': 36.0, 'shoulder-to-crotch': 68.0, 'thigh': 56.0,
    'waist': 90.0, 'wrist': 16.5, 'height_cm': 180.0, 'weight_kg': 80.0,
    'new_weight': 85.0
}

# Convert input dictionary to DataFrame
input_df = pd.DataFrame([test_input])

# Standardize the input using the same scaler from training
input_scaled = scaler_input.transform(input_df[input_features])

# Convert to tensor
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# Define which measurements should be constrained
skeletal_features = ['ankle', 'wrist', 'shoulder-to-crotch']
soft_tissue_features = ['bicep', 'calf', 'chest', 'forearm', 'hip', 'shoulder-breadth', 'thigh', 'waist']

# Define maximum percentage change for skeletal features
MAX_SKELETAL_CHANGE = 0.02  # 2% max change for skeletal structure

# Compute weight change percentage
weight_change_factor = (test_input['new_weight'] / test_input['weight_kg']) - 1  # e.g., if 80 → 85 kg, this is 0.0625 (6.25% weight gain)

# Define scaling factor for skeletal growth (e.g., 10% of weight change)
SKELETAL_GROWTH_SCALING = 0.1  # Skeletal growth is 10% of the weight change

# Predict % change
with torch.no_grad():
    delta_percentage_scaled = model_nn(input_tensor).numpy()

# Reverse transformation
delta_percentage_original = scaler_output.inverse_transform(delta_percentage_scaled)

# Apply % change to original body measurements, dynamically scaling skeletal growth
predicted_measurements = {}
for i, feature in enumerate(output_features):
    if feature in skeletal_features:
        # Limit skeletal growth based on weight change
        max_growth = SKELETAL_GROWTH_SCALING * weight_change_factor * test_input[feature]
        actual_growth = test_input[feature] * delta_percentage_original[0][i]
        constrained_growth = min(actual_growth, max_growth)  # Apply dynamic constraint
        predicted_measurements[feature] = test_input[feature] + constrained_growth
    else:
        # Soft tissue measurements can grow normally
        predicted_measurements[feature] = test_input[feature] * (1 + delta_percentage_original[0][i])

print("Predicted Measurements:", predicted_measurements)