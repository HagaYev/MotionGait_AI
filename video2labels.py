import cv2
import mediapipe as mp
import time
import csv
import os


class PoseDetector:
    """
    A class for detecting human body poses using MediaPipe Pose.
    It supports extracting only the leg landmarks (hips, knees, ankles, etc.).
    """
    def __init__(self, mode=False, smooth=True, detection_con=0.5, track_con=0.5):
        """
        Initialize the pose detector with optional parameters.
        :param mode: Whether to treat input as static images (default: False)
        :param smooth: Whether to smooth landmark positions across frames
        :param detection_con: Minimum detection confidence
        :param track_con: Minimum tracking confidence
        """
        self.mode = mode
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        # Initialize MediaPipe Pose model
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )

    def find_pose(self, img, draw=True):
        """
        Detect and optionally draw the human pose on the input image.
        :param img: Input image (BGR)
        :param draw: Whether to draw the landmarks and connections
        :return: Image with or without drawn landmarks
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        return img

    def get_leg_landmarks(self, img):
        """
        Extract leg-related pose landmarks (hips, knees, ankles, heels, foot indexes).
        :param img: Input image
        :return: Dictionary of leg landmarks with (x, y, z) coordinates
        """
        leg_ids = {
            23: "left_hip", 24: "right_hip",
            25: "left_knee", 26: "right_knee",
            27: "left_ankle", 28: "right_ankle",
            29: "left_heel", 30: "right_heel",
            31: "left_foot_index", 32: "right_foot_index"
        }
        h, w, c = img.shape
        leg_landmarks = {}

        # Retrieve landmark coordinates if pose was detected
        if self.results.pose_landmarks:
            for id in leg_ids:
                lm = self.results.pose_landmarks.landmark[id]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cz = lm.z
                leg_landmarks[leg_ids[id]] = (cx, cy, cz)
        return leg_landmarks


def video_to_labels(video_name, folder_path=None, output_folder=None):
    """
    Process a walking video, extract leg landmarks for each frame,
    and save them into a CSV file.

    :param video_name: Name of the input video file
    :param folder_path: Folder containing the video (optional)
    :param output_folder: Folder where CSV files will be saved (optional)
    """
    if folder_path is None:
        folder_path = r'Robotic_Leg\video\focused frame'

    # Always save CSV files in this path if not specified
    if output_folder is None:
        output_folder = r'Robotic_Leg\csv_data'

    video_path = os.path.join(folder_path, video_name)
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    prev_time = 0

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{video_name}.csv")

    # Define CSV header
    header = ["frame"]
    for joint in [
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]:
        header += [f"{joint}_x", f"{joint}_y", f"{joint}_z"]

    # Overwrite any existing file with the same name
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        frame_count = 0
        while True:
            success, img = cap.read()
            if not success:
                print(" Video processing finished.")
                break

            frame_count += 1

            # Detect pose and extract leg landmarks
            img = detector.find_pose(img)
            leg_coords = detector.get_leg_landmarks(img)

            # Build row for the CSV
            row = [frame_count]
            for key in [
                "left_hip", "right_hip", "left_knee", "right_knee",
                "left_ankle", "right_ankle", "left_heel", "right_heel",
                "left_foot_index", "right_foot_index"
            ]:
                if key in leg_coords:
                    row.extend(leg_coords[key])
                else:
                    row.extend(["", "", ""])  # Empty values if landmark not detected
            writer.writerow(row)

            # Display FPS on video
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(img, f'FPS: {int(fps)}', (70, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

            # Show the processed frame
            cv2.imshow('Pose Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n Data successfully saved to: {output_file}")
