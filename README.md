# MotionGait AI

AI-based system for predicting and tracking joint motion during human walking using computer vision and deep learning.

## Project Overview

Computer vision and deep learning have always fascinated me. Each is impressive on its own, but when combined, they open possibilities I never imagined. Deep learning doesn't just improve computer vision—it makes it smarter, able to perceive and understand in ways that feel almost human.

In the past, when I watched robots, whether on screen or in the lab, I was drawn to their algorithms—the rules and logic that controlled them. Today, I see things differently.

For me, an algorithm isn't just a set of functions or logic gates. The real magic happens where people, data, and intuition meet, enhanced by deep learning. That interplay is what turns a simple program into something that can learn, adapt, and even surprise us.

### Visualization Legend

- **Red leg**: Predicted leg (model output)
- **Green leg**: Real leg (ground truth)
- **Blue leg**: Left leg (input)

![ezgif-3faaabcb69329b13](https://github.com/user-attachments/assets/09deab10-0689-4a4e-a0e6-47df406214d9)

## Features

- **Person Tracking**: Uses YOLO and DeepSort to track people in videos
- **Pose Estimation**: Extracts leg joint positions using MediaPipe
- **LSTM Prediction**: Predicts right leg joint positions based on left leg positions
- **Visualization**: Creates videos with predicted and actual joint positions overlaid

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MotionGait_AI.git
cd MotionGait_AI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLO model weights (optional, will be downloaded automatically):
```bash
# The yolov8n.pt file will be downloaded automatically on first use
# Or download manually from: https://github.com/ultralytics/assets/releases
```

## Usage

### 1. Track People in Video

Extract cropped videos of individual people walking from a video file:

```bash
python videos2frame_impl.py path/to/video.mp4 --output_dir output/focused_frames
```

**Options:**
- `--output_dir`: Directory to save output videos (default: focused_frames in video folder)
- `--yolo_model`: Path to YOLO model weights (default: yolov8n.pt) # Or another one
- `--target_size`: Target size for output videos (default: 400)
- `--fps`: Frames per second for output videos (default: 30)
- `--min_seconds`: Minimum seconds of walking required (default: 4)
- `--no-preview`: Disable video preview during processing

### 2. Convert Videos to Labels

Extract joint positions from videos and save them as CSV files:

```bash
python video2labels_impl.py path/to/video.avi --output_folder csv_data
```

**Options:**
- `--output_folder`: Folder to save CSV files (default: csv_data in video folder)
- `--no-preview`: Disable video preview during processing


### 3. Train the Model

Train the LSTM model to predict leg joint positions:

```bash
python model_impl.py --data_folder csv_data
```

**Options:**
- `--data_folder`: Path to folder containing CSV files (required)
- `--sequence_length`: Length of sequences (default: 30)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--train_val_split`: Fraction of data for training (default: 0.8)
- `--output_video`: Path to save prediction video (optional)
- `--test_csv`: Path to CSV file for video generation (optional)
- `--model_save`: Path to save trained model (optional)

**Example:**
```bash
python train_model.py \
    --data_folder csv_data \
    --epochs 100 \
    --batch_size 64 \
    --model_save model.pth \
    --test_csv csv_data/test.csv \
    --output_video output/prediction.mp4
```

## Project Workflow

1. **Load videos** of people walking
2. **Track people** in videos and extract cropped sequences (`track_people.py`)
3. **Extract joint positions** from videos and save as CSV files (`video2labels.py`)
4. **Prepare data** for training (remove irrelevant files, handle empty rows or NaN values)
5. **Train LSTM model** to predict leg joint positions (`train_model.py`)
6. **Monitor training** progress with training graphs
7. **Test predictions** on new video files
8. **Visualize results** with predicted and actual joint positions


## Data Structure

The CSV files contain the following columns:
- `frame`: Frame number
- `left_hip_x`, `left_hip_y`, `left_hip_z`: Left hip coordinates
- `left_knee_x`, `left_knee_y`, `left_knee_z`: Left knee coordinates
- `left_ankle_x`, `left_ankle_y`, `left_ankle_z`: Left ankle coordinates
- `left_heel_x`, `left_heel_y`, `left_heel_z`: Left heel coordinates
- `left_foot_index_x`, `left_foot_index_y`, `left_foot_index_z`: Left foot index coordinates
- `right_hip_x`, `right_hip_y`, `right_hip_z`: Right hip coordinates
- `right_knee_x`, `right_knee_y`, `right_knee_z`: Right knee coordinates
- `right_ankle_x`, `right_ankle_y`, `right_ankle_z`: Right ankle coordinates
- `right_heel_x`, `right_heel_y`, `right_heel_z`: Right heel coordinates
- `right_foot_index_x`, `right_foot_index_y`, `right_foot_index_z`: Right foot index coordinates

## Model Architecture

The LSTM model consists of:
- **Input**: Sequences of left leg joint positions (15 features: 5 joints × 3 coordinates)
- **LSTM Layers**: 2 layers with 128 hidden units
- **Output**: Right leg joint positions (15 features: 5 joints × 3 coordinates)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

### Performance Tips

- Use GPU for faster training and inference
- Reduce video resolution for faster processing
- Use batch processing for multiple videos
- Adjust sequence length based on your data

- [YouTube: Human Pose Estimation with YOLOv8](https://www.youtube.com/watch?v=MmWPYKF2_A0) – Tutorial on using YOLOv8 for tracking people and extracting keypoints.
- [YouTube: Pose Estimation and Joint Prediction](https://www.youtube.com/watch?v=ag3DLKsl2vk&t=296s) – Guide on creating datasets and processing joint coordinates for learning.
- [Prosthetic Classification and Prediction](https://github.com/benjyb1/Prosthetic-Classifcation-and-Prediction) – Reference implementation.

## Acknowledgments

- MediaPipe for pose estimation
- Ultralytics for YOLO implementation
- DeepSort for person tracking
- PyTorch for deep learning framework LSTM model
