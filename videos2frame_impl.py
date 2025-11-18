from video2frames import process_video
import os

if __name__ == "__main__":

    folder = r"Robotic_Leg/raw_videos_file"
    for video_name in os.listdir(folder):
        if video_name.endswith((".mp4", ".avi", ".mov")):
            print(f"Processing {video_name}...")
            process_video(folder_path=folder, video_name)