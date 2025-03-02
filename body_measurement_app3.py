import tkinter as tk
from tkinter import messagebox
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
import sys
import numpy as np

def resource_path(relative_path):
    """ Get the absolute path to a resource, works for both development and PyInstaller EXE """
    if getattr(sys, 'frozen', False):  # If running as a PyInstaller EXE
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

# Load neural network model and scalers
scaler_input = joblib.load(resource_path("scaler_input.pkl"))
scaler_output = joblib.load(resource_path("scaler_output.pkl"))

# Define input/output features (consistent with training)
input_features = [
    'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm', 'height', 'hip',
    'leg-length', 'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist',
    'height_cm', 'weight_kg', 'new_weight'
]

output_features = [
    'ankle', 'bicep', 'calf', 'chest', 'forearm', 'hip',
    'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
]

# Define the neural network structure (must match training)
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

# Load trained model
input_size = len(input_features)
output_size = len(output_features)
model_nn = BodyMeasurementNN(input_size, output_size)
model_nn.load_state_dict(torch.load(resource_path("best_model.pth"), map_location=torch.device('cpu')))
model_nn.eval()  # Set model to evaluation mode

# Define skeletal features that should grow minimally
skeletal_features = ['ankle', 'wrist', 'shoulder-to-crotch']
SKELETAL_GROWTH_SCALING = 0.1  # 10% of weight change

# Prediction function
def predict_measurements():
    try:
        # Get user input
        user_input = {
            "ankle": float(entries["Ankle (cm)"].get()),
            "arm-length": float(entries["Arm Length (cm)"].get()),
            "bicep": float(entries["Bicep (cm)"].get()),
            "calf": float(entries["Calf (cm)"].get()),
            "chest": float(entries["Chest (cm)"].get()),
            "forearm": float(entries["Forearm (cm)"].get()),
            "height": float(entries["Height (cm)"].get()),
            "hip": float(entries["Hip (cm)"].get()),
            "leg-length": float(entries["Leg Length (cm)"].get()),
            "shoulder-breadth": float(entries["Shoulder Breadth (cm)"].get()),
            "shoulder-to-crotch": float(entries["Shoulder-to-Crotch (cm)"].get()),
            "thigh": float(entries["Thigh (cm)"].get()),
            "waist": float(entries["Waist (cm)"].get()),
            "wrist": float(entries["Wrist (cm)"].get()),
            "height_cm": float(entries["Height (cm)"].get()),  # Duplicate for consistency
            "weight_kg": float(entries["Current Weight (kg)"].get())
        }

        target_weight = float(entries["Target Weight (kg)"].get())

        # Compute weight difference features
        user_input["new_weight"] = target_weight

        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])

        # Compute weight change percentage
        weight_change_factor = (target_weight / user_input["weight_kg"]) - 1  # Example: 80 → 85 kg → 0.0625 (6.25% gain)

        # Standardize input
        input_scaled = scaler_input.transform(input_df[input_features])

        # Convert to tensor
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

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
                max_growth = SKELETAL_GROWTH_SCALING * weight_change_factor * user_input[feature]
                actual_growth = user_input[feature] * delta_percentage_original[0][i]
                constrained_growth = min(actual_growth, max_growth)  # Apply dynamic constraint
                predicted_measurements[feature] = user_input[feature] + constrained_growth
            else:
                # Soft tissue measurements can grow normally
                predicted_measurements[feature] = user_input[feature] * (1 + delta_percentage_original[0][i])

        # Display results
        result_text.set("Predicted Measurements:\n" + "\n".join([f"{k}: {v:.1f} cm" for k, v in predicted_measurements.items()]))
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

import tkinter as tk
from tkinter import ttk

# Create Tkinter GUI
root = tk.Tk()
root.title("Body Measurement Predictor")

# Set a fixed window size (Width x Height)
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 680

# Get screen width and height to center the window
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate position to center the window
x_position = (screen_width - WINDOW_WIDTH) // 2
y_position = (screen_height - WINDOW_HEIGHT) // 2

# Set fixed window size and position it in the center
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x_position}+{y_position}")
root.resizable(False, False)  # Prevent resizing

# Configure background color
root.configure(bg="#f4f4f4")

# Main frame (Left Side) - User Inputs
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky="NSEW")

# Result frame (Right Side) - Predictions
result_frame = ttk.Frame(root, padding="10", relief="ridge")
result_frame.grid(row=0, column=1, sticky="NS", padx=20, pady=10)

# Style configurations
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=5)
style.configure("TLabel", font=("Arial", 10), background="#f4f4f4")
style.configure("TEntry", font=("Arial", 10), padding=5)

# Input labels and fields
labels = [
    "Height (cm)", "Current Weight (kg)", "Target Weight (kg)", "Ankle (cm)", "Arm Length (cm)",
    "Bicep (cm)", "Calf (cm)", "Chest (cm)", "Forearm (cm)", "Hip (cm)", "Leg Length (cm)",
    "Shoulder Breadth (cm)", "Shoulder-to-Crotch (cm)", "Thigh (cm)", "Waist (cm)", "Wrist (cm)"
]

entries = {}

# Place labels and entry fields in a grid (Left Side)
for i, label in enumerate(labels):
    ttk.Label(main_frame, text=label + ":").grid(row=i, column=0, sticky="W", padx=5, pady=3)
    entry = ttk.Entry(main_frame, width=10)
    entry.grid(row=i, column=1, padx=5, pady=3)
    entries[label] = entry  # Store entries in dictionary

# Predict button
predict_button = ttk.Button(main_frame, text="Predict Measurements", command=predict_measurements)
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

# Result display (Right Side)
result_text = tk.StringVar()
result_label = ttk.Label(result_frame, textvariable=result_text, font=("Arial", 12), background="#ffffff",
                         relief="solid", padding=10, justify="left", anchor="w", width=30)
result_label.pack(expand=True, fill="both")

# Run the GUI
root.mainloop()