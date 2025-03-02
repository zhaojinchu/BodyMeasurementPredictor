import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
file_path = "merged_dataset.csv"  # Ensure this file exists
df = pd.read_csv(file_path)

# Filter for males only
df_male = df[df['gender'] == 'male'].copy()

# Define input and output features
input_features = [
    'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm', 'height', 'hip',
    'leg-length', 'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist',
    'height_cm', 'weight_kg'
]

output_features = [
    'ankle', 'bicep', 'calf', 'chest', 'forearm', 'hip',
    'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
]

# Ensure 'new_weight' column (simulating weight change)
df_male['new_weight'] = df_male['weight_kg'] * np.random.uniform(0.9, 1.1, len(df_male))

# Compute percentage change for each body measurement
for feature in output_features:
    df_male[f'percent_change_{feature}'] = (df_male[feature] * (df_male['new_weight'] / df_male['weight_kg']) - df_male[feature]) / df_male[feature]

# Data augmentation: Generate synthetic weight variations (10%, 15%, 20% change)
augmented_rows = []
for _, row in df_male.iterrows():
    for factor in [1.1, 1.15, 1.2]:  # Simulate weight gains of 10%, 15%, 20%
        new_row = row.copy()
        new_row['new_weight'] = row['weight_kg'] * factor
        for feature in output_features:
            new_row[f'percent_change_{feature}'] = (row[feature] * (new_row['new_weight'] / row['weight_kg']) - row[feature]) / row[feature]
        augmented_rows.append(new_row)

# Add augmented data
df_male = pd.concat([df_male, pd.DataFrame(augmented_rows)], ignore_index=True)

# Scale input and output separately
scaler_input = StandardScaler()
scaler_output = StandardScaler()

X_scaled = scaler_input.fit_transform(df_male[input_features + ['new_weight']])  # Scale inputs
y_scaled = scaler_output.fit_transform(df_male[[f'percent_change_{feature}' for feature in output_features]])  # Scale percentage changes

# Save scalers
joblib.dump(scaler_input, "scaler_input.pkl")
joblib.dump(scaler_output, "scaler_output.pkl")

# Save preprocessed data
df_male.to_csv("preprocessed_dataset.csv", index=False)

print("Preprocessing complete. Scalers and dataset saved.")
