# GaitJointPredictor

AI-based system for predicting and tracking joint motion during human walking.

---

## Project Overview
Computer vision and deep learning have always fascinated me. Each is impressive on its own, but when combined, they open possibilities I never imagined. Deep learning doesn’t just improve computer vision it makes it smarter, able to perceive and understand in ways that feel almost human.

In the past, when I watched robots, whether on screen or in the lab, I was drawn to their algorithms the rules and logic that controlled them. Today, I see things differently.

For me, an algorithm isn’t just a set of functions or logic gates. The real magic happens where people, data, and intuition meet, enhanced by deep learning. That interplay is what turns a simple program into something that can learn, adapt, and even surprise us.

The final result is pretty cool:

Red leg - predicted leg

green leg- real leg

blue leg- seconed leg

![ezgif-3faaabcb69329b13](https://github.com/user-attachments/assets/09deab10-0689-4a4e-a0e6-47df406214d9)


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
- https://github.com/benjyb1/Prosthetic-Classifcation-and-Prediction





