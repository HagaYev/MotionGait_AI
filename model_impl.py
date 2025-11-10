import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import cv2

from model import LegMotionLSTM, create_sequences_from_array, predict_to_coords, X_cols, y_cols, right_joints, left_joints
from sklearn.preprocessing import StandardScaler

# Config
folder_path = "" # choose your file
sequence_length = 30
batch_size = 64
epochs = 100
lr = 1e-3
train_val_split = 0.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_video_path = r'Robotic_Leg\video\focused frame\leg_prediction.mp4'

# Read CSVs and prepare sequences
csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv") and not f.startswith('example1')])
if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in {folder_path}")

all_X_frames, all_y_frames = [], []

for file in csv_files:
    df = pd.read_csv(os.path.join(folder_path, file))
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=X_cols + y_cols, how='all').reset_index(drop=True)

    missing = [c for c in X_cols + y_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file}: {missing}")

    all_X_frames.append(df[X_cols].values)
    all_y_frames.append(df[y_cols].values)

# Fit scalers
all_X_concat = np.vstack([a for a in all_X_frames if len(a) > 0])
all_y_concat = np.vstack([a for a in all_y_frames if len(a) > 0])

scaler_X = StandardScaler().fit(all_X_concat)
scaler_y = StandardScaler().fit(all_y_concat)

# Generate sequences
X_sequences_list, y_sequences_list = [], []

for X_arr_raw, y_arr_raw in zip(all_X_frames, all_y_frames):
    if len(X_arr_raw) < sequence_length:
        continue
    X_scaled = scaler_X.transform(X_arr_raw)
    y_scaled = scaler_y.transform(y_arr_raw)
    X_seq, y_seq = create_sequences_from_array(X_scaled, y_scaled, sequence_length)
    if X_seq.shape[0] > 0:
        X_sequences_list.append(X_seq)
        y_sequences_list.append(y_seq)

X_all = np.vstack(X_sequences_list)
y_all = np.vstack(y_sequences_list)

X_tensor = torch.tensor(X_all, dtype=torch.float32)
y_tensor = torch.tensor(y_all, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)

n_train = int(len(dataset) * train_val_split)
n_val = len(dataset) - n_train
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# Initialize model
input_size = X_all.shape[2]
output_size = y_all.shape[1]

model = LegMotionLSTM(input_size=input_size, hidden_size=128, num_layers=2, output_size=output_size).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training
train_losses, val_losses = [], []

for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            running_val += loss.item() * X_batch.size(0)
    epoch_val_loss = running_val / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch}/{epochs} — Train Loss: {epoch_train_loss:.6f} — Val Loss: {epoch_val_loss:.6f}")

# Plot Loss Curve
plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

# Video Generation
first_csv = os.path.join(folder_path, csv_files[0])
df = pd.read_csv(first_csv)
df[X_cols + y_cols] = df[X_cols + y_cols].fillna(method='ffill').fillna(0)

height, width = 480, 640
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

X_scaled_df = scaler_X.transform(df[X_cols].values)
model.eval()

with torch.no_grad():
    for i in range(len(df)):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw left leg
        left_points = {}
        for joint in left_joints:
            x, y = int(round(df[f"{joint}_x"].iloc[i])), int(round(df[f"{joint}_y"].iloc[i]))
            x, y = np.clip(x, 0, width-1), np.clip(y, 0, height-1)
            left_points[joint] = (x,y)
            cv2.circle(canvas, (x,y), 6, (255,0,0), -1)

        # Draw right leg real
        right_real_points = {}
        for joint in right_joints:
            x, y = int(round(df[f"{joint}_x"].iloc[i])), int(round(df[f"{joint}_y"].iloc[i]))
            x, y = np.clip(x, 0, width-1), np.clip(y, 0, height-1)
            right_real_points[joint] = (x,y)
            cv2.circle(canvas, (x,y), 6, (0,255,0), -1)

        # Prediction
        if i < sequence_length:
            seq_input = np.zeros((sequence_length, X_scaled_df.shape[1]), dtype=np.float32)
            seq_input[-(i+1):] = X_scaled_df[:i+1]
        else:
            seq_input = X_scaled_df[i-sequence_length+1:i+1]
        seq_tensor = torch.tensor(seq_input, dtype=torch.float32).unsqueeze(0).to(device)
        y_pred_scaled = model(seq_tensor).cpu().numpy()[0]
        pred_pts = predict_to_coords(y_pred_scaled, scaler_y)

        right_pred_points = {}
        for j, joint in enumerate(right_joints):
            x, y, _ = pred_pts[j]
            x, y = np.clip(x, 0, width-1), np.clip(y, 0, height-1)
            right_pred_points[joint] = (x,y)
            cv2.circle(canvas, (x,y), 6, (0,0,255), -1)

        # Connect joints
        connections = [("hip","knee"), ("knee","ankle"), ("ankle","heel"), ("heel","foot_index")]

        for from_key, to_key in connections:
            # Left
            left_from, left_to = f"left_{from_key}", f"left_{to_key}"
            if left_from in left_points and left_to in left_points:
                cv2.line(canvas, left_points[left_from], left_points[left_to], (255,0,0), 2)
            # Right real
            r_from, r_to = f"right_{from_key}", f"right_{to_key}"
            if r_from in right_real_points and r_to in right_real_points:
                cv2.line(canvas, right_real_points[r_from], right_real_points[r_to], (0,255,0), 2)
            # Right predicted
            if r_from in right_pred_points and r_to in right_pred_points:
                cv2.line(canvas, right_pred_points[r_from], right_pred_points[r_to], (0,0,255), 2)

        out.write(canvas)

out.release()
print(f" Video generated: {output_video_path}")
