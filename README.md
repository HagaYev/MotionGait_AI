# GaitJointPredictor

AI-based system for predicting and tracking joint motion during human walking.

---

## Project Overview
Computer vision and deep learning are two transformative technologies, each powerful on its own, yet even more impactful when combined. Deep learning amplifies the capabilities of computer vision, creating a system capable of sophisticated perception and understanding.

In the past, when I observed robots whether in reality or on screen I focused on the algorithm that governed their behavior. Today, my perspective has evolved.

This project embraces the idea that an algorithm is not merely a set of mathematical functions or logical gates. The true “algorithm” emerges from the interplay between people, data, and human insight augmented and strengthened by deep learning.

The finel result is amazing:

https://github.com/user-attachments/assets/533df8d6-acdc-45ca-8170-bae5bc1701d8



---

## Project Steps

1. Load videos of people walking.
2. Split each video into individual frames for each pedestrian.
3. Generate CSV files containing labels of the relative positions of pedestrian joints in the XYZ axes.
4. Prepare the data for learning (remove irrelevant files, handle empty rows or NaN values).
5. Feed the data into the LSTM model.
6. Monitor learning progress with training graphs.
7. Test predictions on a new video file.

<img width="1326" height="722" alt="image" src="https://github.com/user-attachments/assets/3a9069cd-fc54-48f9-aa6f-6c618c45da70" />
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


## Sources / References

- [YouTube: Human Pose Estimation with YOLOv8](https://www.youtube.com/watch?v=MmWPYKF2_A0) – Tutorial on using YOLOv8 for tracking people and extracting keypoints.
- [YouTube: Pose Estimation and Joint Prediction](https://www.youtube.com/watch?v=ag3DLKsl2vk&t=296s) – Guide on creating datasets and processing joint coordinates for learning.





