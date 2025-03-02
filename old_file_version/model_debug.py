import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import os
import matplotlib.pyplot as plt

# Load dataset
file_path = "merged_dataset.csv"  # Adjust if needed
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
    'ankle', 'bicep', 'calf', 'chest', 'forearm', 'height', 'hip', 
    'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
]

# Ensure necessary columns exist
missing_columns = set(input_features + output_features + ['gender']) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# Create a 'new_weight' column (simulating weight change)
df_male['new_weight'] = df_male['weight_kg'] * np.random.uniform(0.9, 1.1, len(df_male))

def augment_data(df, noise_level=0.02):
    df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns
    noise = np.random.normal(0, noise_level, df_numeric.shape)
    df_augmented = df_numeric + noise  # Add noise only to numeric columns
    return pd.concat([df_numeric, df_augmented])  # Return augmented numeric data only
# Apply augmentation
df_male = augment_data(df_male)

# Prepare input (X) and output (y)
X = df_male[input_features + ['new_weight']]
y = df_male[output_features]

# Standardize input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

# Split into train and test sets
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
X_train_tensor, X_test_tensor = random_split(X_tensor, [train_size, test_size])
y_train_tensor, y_test_tensor = random_split(y_tensor, [train_size, test_size])

# Extract tensors from Subset objects
X_train_tensor = torch.stack([X_tensor[i] for i in X_train_tensor.indices])
X_test_tensor = torch.stack([X_tensor[i] for i in X_test_tensor.indices])
y_train_tensor = torch.stack([y_tensor[i] for i in y_train_tensor.indices])
y_test_tensor = torch.stack([y_tensor[i] for i in y_test_tensor.indices])

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=8, shuffle=True, drop_last = True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=8, shuffle=False, drop_last = True)

print(f"Train set size: {len(X_train_tensor)}, Test set size: {len(X_test_tensor)}")

class BodyMeasurementNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(BodyMeasurementNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256, track_running_stats=False)  # Batch Norm after first layer
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128, track_running_stats=False)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.3)  

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # Batch Norm applied
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define function to save model weights
def save_model(model, filename="model_weights.pth"):
    torch.save(model.state_dict(), filename)
    #print(f"Model weights saved to {filename}")

# Define function to load model weights
def load_model(model, filename="model_weights.pth"):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
        print(f"Loaded model weights from {filename}")
    else:
        print("No saved model weights found. Training from scratch.")

# Initialize model
input_size = X.shape[1]  
output_size = y.shape[1]  
model_nn = BodyMeasurementNN(input_size, output_size)

# Load existing weights if available
load_model(model_nn)

# Define optimizer and loss function
optimizer = optim.Adam(model_nn.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # Reduce LR every 100 epochs
criterion = nn.SmoothL1Loss()  # Huber Loss

# Lists to store training and validation loss
train_loss_history = []
val_loss_history = []

# Train the Neural Network
num_epochs = 500
for epoch in range(num_epochs):
    model_nn.train()
    total_train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model_nn(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    scheduler.step()
    
    # Compute average training loss
    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # Compute validation loss
    model_nn.eval()
    total_val_loss = 0
    with torch.no_grad():
        for val_inputs, val_targets in test_loader:
            val_outputs = model_nn(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(test_loader)
    val_loss_history.append(avg_val_loss)

    # Save model weights
    save_model(model_nn)

    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

# Plot Training and Validation Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_loss_history, label="Training Loss", color='blue')
plt.plot(range(num_epochs), val_loss_history, label="Validation Loss", color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()


# Plot Training Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), val_loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

# Evaluate the model
model_nn.eval()
predictions, actuals = [], []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model_nn(inputs)
        predictions.append(outputs.numpy())
        actuals.append(targets.numpy())

# Convert predictions to NumPy arrays
predictions = np.vstack(predictions)
actuals = np.vstack(actuals)

# Compute MAE and R² for validation
mae_nn = mean_absolute_error(actuals, predictions)
r2_nn = r2_score(actuals, predictions)

print(f"Final MAE: {mae_nn}, R² Score: {r2_nn}")

# Scatter plot: Predictions vs Actual
plt.figure(figsize=(10, 5))
plt.scatter(actuals.flatten(), predictions.flatten(), alpha=0.5)
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r', linewidth=2)
plt.xlabel("Actual Measurements")
plt.ylabel("Predicted Measurements")
plt.title("Predicted vs Actual Measurements")
plt.show()

# Residual Plot
residuals = actuals - predictions
plt.figure(figsize=(10, 5))
plt.scatter(predictions.flatten(), residuals.flatten(), alpha=0.5)
plt.axhline(y=0, color='r', linestyle='dashed')
plt.xlabel("Predicted Measurements")
plt.ylabel("Residual (Error)")
plt.title("Residual Plot")
plt.show()


print(f"MAE: {mae_nn}, R² Score: {r2_nn}")





# Function to make predictions
def predict_new_measurements(input_dict):
    input_df = pd.DataFrame([input_dict])
    
    # Ensure required features are present
    required_features = input_features + ['new_weight']
    missing_features = set(required_features) - set(input_df.columns)
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")

    # Standardize input
    input_scaled = scaler.transform(input_df[required_features])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Make prediction
    model_nn.eval()
    with torch.no_grad():
        prediction = model_nn(input_tensor).numpy()

    return dict(zip(output_features, prediction[0]))

# Example test dictionary
test_input = {
    'ankle': 23.5,
    'arm-length': 52.0,
    'bicep': 32.0,
    'calf': 37.0,
    'chest': 105.0,
    'forearm': 27.0,
    'height': 180.0,
    'hip': 100.0,
    'leg-length': 82.0,
    'shoulder-breadth': 36.0,
    'shoulder-to-crotch': 68.0,
    'thigh': 56.0,
    'waist': 90.0,
    'wrist': 16.5,
    'height_cm': 180.0,
    'weight_kg': 80.0,
    'new_weight': 85.0  # Example of a weight change
}

# Run prediction and print result
predicted_output = predict_new_measurements(test_input)
print("Predicted New Measurements:", predicted_output)
