import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# Joint names (for reference)
left_joints = ["left_hip","left_knee","left_ankle","left_heel","left_foot_index"]
right_joints = ["right_hip","right_knee","right_ankle","right_heel","right_foot_index"]

X_cols = [f"{j}_{axis}" for j in left_joints for axis in ["x","y","z"]]
y_cols = [f"{j}_{axis}" for j in right_joints for axis in ["x","y","z"]]


# Function to create sequences from an array
def create_sequences_from_array(X_arr, y_arr, seq_len):
    X_seq, y_seq = [], []
    n = len(X_arr)
    for i in range(n - seq_len + 1):
        X_seq.append(X_arr[i:i+seq_len])
        y_seq.append(y_arr[i+seq_len-1])
    if len(X_seq) == 0:
        return np.zeros((0, seq_len, X_arr.shape[1])), np.zeros((0, y_arr.shape[1]))
    return np.array(X_seq), np.array(y_seq)

# LSTM Model
class LegMotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=15, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

# Function to convert predicted vector back to coordinates
def predict_to_coords(y_pred_scaled, scaler_y, right_joints=right_joints):
    y_inv = scaler_y.inverse_transform(y_pred_scaled.reshape(1, -1))[0]
    pts = []
    for j in range(len(right_joints)):
        x = int(round(y_inv[j*3]))
        y = int(round(y_inv[j*3+1]))
        z = float(y_inv[j*3+2])
        pts.append((x,y,z))
    return pts
