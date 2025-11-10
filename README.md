# GaitJointPredictor

AI-based system for predicting and tracking joint motion during human walking.

---

## Project Overview
Computer vision and deep learning are two transformative technologies, each powerful on its own, yet even more impactful when combined. Deep learning amplifies the capabilities of computer vision, creating a system capable of sophisticated perception and understanding.

In the past, when I observed robots whether in reality or on screen I focused on the algorithm that governed their behavior. Today, my perspective has evolved.

This project embraces the idea that an algorithm is not merely a set of mathematical functions or logical gates. The true “algorithm” emerges from the interplay between people, data, and human insight augmented and strengthened by deep learning.

---

**Key Features:**
- Detects and tracks people in video sequences.
- Predicts joint positions during gait (walking motion).
- Visualizes tracked joints and motion trajectories on video output.
- Lightweight and fast inference using YOLOv8n.
- Fully adaptable to different video sources.


---

## Folder Structure

Robotic_Leg/
├── video/                     # Input videos and output results (ignored by git)
├── csv_data/                   # CSV files generated from videos (ignored by git)
├── Model.py                    # Main script: YOLO detection + tracking, GPU recommended
├── model_impl.py               # Implementation of joint prediction model
├── video2labels.py             # Helper script: convert tracked videos into CSV labels
├── video2frames.py             # Helper script: extract frames from videos
├── yolov8n.pt                  # YOLOv8n model weights (ignored by git)
└── README.md                   # Project documentation




