import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Save function
def save_model(model, filename="best_model.pth"):
    torch.save(model.state_dict(), filename)
    #print(f"Model saved as {filename}")
    
# Load preprocessed dataset
df = pd.read_csv("preprocessed_dataset.csv")

# Define input and output features
input_features = [
    'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm', 'height', 'hip',
    'leg-length', 'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist',
    'height_cm', 'weight_kg', 'new_weight'
]

output_features = [
    'ankle', 'bicep', 'calf', 'chest', 'forearm', 'hip',
    'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
]

# Load scalers
scaler_input = joblib.load("scaler_input.pkl")
scaler_output = joblib.load("scaler_output.pkl")

# Scale inputs and outputs
X_scaled = scaler_input.transform(df[input_features])  # Inputs
y_scaled = scaler_output.transform(df[[f'percent_change_{feature}' for feature in output_features]])  # Percentage change

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Split into train and test sets
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size

train_indices, test_indices = random_split(range(len(X_tensor)), [train_size, test_size])

X_train_tensor = X_tensor[train_indices]
X_test_tensor = X_tensor[test_indices]
y_train_tensor = y_tensor[train_indices]
y_test_tensor = y_tensor[test_indices]


# Define DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=8, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=8, shuffle=False, drop_last=True)

# Define Model
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

# Initialize model
input_size = len(input_features)
output_size = len(output_features)
model_nn = BodyMeasurementNN(input_size, output_size)

# Define optimizer and loss function
optimizer = optim.Adam(model_nn.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
criterion = nn.SmoothL1Loss()

# Track best validation loss
best_val_loss = float('inf')

# Train the Neural Network
num_epochs = 500
train_loss_history, val_loss_history = [], []

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

    # Save model only if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_model(model_nn, "best_model.pth")

    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Best Val Loss: {best_val_loss}")

# Final message
print(f"Training complete. Best model saved with validation loss: {best_val_loss}.")