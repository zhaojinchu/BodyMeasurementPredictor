import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

import os
import sys
import joblib

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

# Load the models
xgb_models = joblib.load(resource_path("xgb_models.pkl"))
ridge_models = joblib.load(resource_path("ridge_models.pkl"))
features = joblib.load(resource_path("features.pkl"))
target = joblib.load(resource_path("target.pkl"))

# Load trained models and feature list
#xgb_models = joblib.load("xgb_models.pkl")
#ridge_models = joblib.load("ridge_models.pkl")
#features = joblib.load("features.pkl")
#target = joblib.load("target.pkl")

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

# Prediction function
def predict_measurements():
    try:
        # Get user input
        user_input = {
            "height_cm": float(entry_height.get()),
            "weight_kg": float(entry_weight.get()),
            "bicep": float(entry_bicep.get()),
            "calf": float(entry_calf.get()),
            "chest": float(entry_chest.get()),
            "hip": float(entry_hip.get()),
            "thigh": float(entry_thigh.get()),
            "waist": float(entry_waist.get())
        }
        target_weight = float(entry_target_weight.get())

        user_input["weight_difference"] = target_weight - user_input["weight_kg"]
        user_input["weight_diff_squared"] = user_input["weight_difference"] ** 2

        # Convert to DataFrame
        input_data = pd.DataFrame([user_input])[features]

        # Generate predictions
        predictions = {}
        for measurement in target:
            if measurement in ridge_models:
                predictions[measurement] = round(ridge_models[measurement].predict(input_data[["weight_difference", "weight_diff_squared"]])[0], 2)
            else:
                predictions[measurement] = round(xgb_models[measurement].predict(input_data)[0], 2)

        # Display results
        result_text.set("Predicted Measurements:\n" + "\n".join([f"{k}: {v:.1f} cm" for k, v in predictions.items()]))
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Create Tkinter GUI
root = tk.Tk()
root.title("Body Measurement Predictor")
root.geometry("400x600")

# Labels and input fields
tk.Label(root, text="Height (cm):").pack()
entry_height = tk.Entry(root)
entry_height.pack()

tk.Label(root, text="Current Weight (kg):").pack()
entry_weight = tk.Entry(root)
entry_weight.pack()

tk.Label(root, text="Target Weight (kg):").pack()
entry_target_weight = tk.Entry(root)
entry_target_weight.pack()

tk.Label(root, text="Bicep (cm):").pack()
entry_bicep = tk.Entry(root)
entry_bicep.pack()

tk.Label(root, text="Calf (cm):").pack()
entry_calf = tk.Entry(root)
entry_calf.pack()

tk.Label(root, text="Chest (cm):").pack()
entry_chest = tk.Entry(root)
entry_chest.pack()

tk.Label(root, text="Hip (cm):").pack()
entry_hip = tk.Entry(root)
entry_hip.pack()

tk.Label(root, text="Thigh (cm):").pack()
entry_thigh = tk.Entry(root)
entry_thigh.pack()

tk.Label(root, text="Waist (cm):").pack()
entry_waist = tk.Entry(root)
entry_waist.pack()

# Predict button
tk.Button(root, text="Predict Measurements", command=predict_measurements).pack()

# Result display
result_text = tk.StringVar()
tk.Label(root, textvariable=result_text, wraplength=380, justify="left").pack()

# Run the GUI
root.mainloop()
