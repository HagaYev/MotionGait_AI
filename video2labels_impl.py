import os
from video2labels import video_to_labels

FOLDER_PATH = r"Robotic_Leg\video\focused frame"

for file_name in os.listdir(FOLDER_PATH):
    if file_name.lower().endswith(".avi"):
        video_path = os.path.join(FOLDER_PATH, file_name)
        print(f"Processing: {file_name}")
        video_to_labels(video_path)
