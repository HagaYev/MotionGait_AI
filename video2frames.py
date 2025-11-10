import cv2
import numpy as np
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
from collections import deque

# File and folder configuration
FOLDER_PATH = r"Robotic_Leg\video"
VIDEO_NAME = "" # choose your video
VIDEO_PATH = os.path.join(FOLDER_PATH, VIDEO_NAME)
OUTPUT_DIR = r'Robotic_Leg\video\focused frame'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# General parameters
TARGET_SIZE = 400
fps = 30
MIN_SECONDS_WALK = 4
MIN_FRAMES_WALK = int(fps * MIN_SECONDS_WALK)
padding_factor = 0.4
CONF_THRESHOLD = 0.6
MIN_LANDMARKS = 20
MAX_NO_DETECT_FRAMES = 8
SMOOTHING_ALPHA = 0.2
MAX_MOVE = 100  # Maximum allowed pixel movement between frames
HISTORY_LEN = 5  # Number of frames to keep in center history

# Load models
yolo_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
tracker = DeepSort(max_age=MAX_NO_DETECT_FRAMES, n_init=30)

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Data storage for each person
people_data = {}  # track_id -> dict with all person data

# Frame processing loop
for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    canvas = None

    # YOLO detection stage
    results = yolo_model(frame)[0]
    dets = []
    for det in results.boxes:
        if int(det.cls[0]) == 0:  # Only people
            conf = float(det.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            dets.append(([x1, y1, x2, y2], conf, "person"))

    # DeepSort tracking stage
    tracks = tracker.update_tracks(dets, frame=frame)
    current_ids = set()

    for t in tracks:
        if not t.is_confirmed():
            continue
        track_id = t.track_id
        current_ids.add(track_id)

        try:
            x1, y1, x2, y2 = map(int, t.to_ltrb())
        except:
            continue

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Mediapipe Pose detection
        rgb_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pose_res = pose_model.process(rgb_crop)
        if not pose_res.pose_landmarks or len(pose_res.pose_landmarks.landmark) < MIN_LANDMARKS:
            continue

        h, w, _ = cropped.shape
        xs = [lm.x * w for lm in pose_res.pose_landmarks.landmark]
        ys = [lm.y * h for lm in pose_res.pose_landmarks.landmark]
        center_x = int(np.mean(xs)) + x1
        center_y = int(np.mean(ys)) + y1
        curr_width = max(xs) - min(xs)
        curr_height = max(ys) - min(ys)

        # Create a new record for a new person
        if track_id not in people_data:
            people_data[track_id] = {
                "writer": None,
                "frames_queue": deque(),
                "detected_frames": 0,
                "last_center": (center_x, center_y),
                "max_width": int(curr_width),
                "max_height": int(curr_height),
                "center_history": deque(maxlen=HISTORY_LEN)
            }

        data = people_data[track_id]

        # Check for sudden jumps (distance check)
        last_cx, last_cy = data["last_center"]
        dx = center_x - last_cx
        dy = center_y - last_cy
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist > MAX_MOVE:
            center_x, center_y = last_cx, last_cy  # Ignore jump

        # Update center history and use averaged position
        data["center_history"].append((center_x, center_y))
        smooth_cx = int(np.mean([c[0] for c in data["center_history"]]))
        smooth_cy = int(np.mean([c[1] for c in data["center_history"]]))
        data["last_center"] = (smooth_cx, smooth_cy)

        # Update width/height and frame count
        data["max_width"] = max(data["max_width"], int(curr_width))
        data["max_height"] = max(data["max_height"], int(curr_height))
        data["detected_frames"] += 1

        # Crop around center with padding
        half_w = int((data["max_width"] // 2) * (1 + padding_factor))
        half_h = int((data["max_height"] // 2) * (1 + padding_factor))
        x_min = max(0, smooth_cx - half_w)
        x_max = min(frame_width, smooth_cx + half_w)
        y_min = max(0, smooth_cy - half_h)
        y_max = min(frame_height, smooth_cy + half_h)

        cropped_center = frame[y_min:y_max, x_min:x_max]
        ch, cw, _ = cropped_center.shape
        if cw == 0 or ch == 0:
            continue

        scale = min(TARGET_SIZE / cw, TARGET_SIZE / ch)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        resized = cv2.resize(cropped_center, (new_w, new_h))

        canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        x_offset = (TARGET_SIZE - new_w) // 2
        y_offset = (TARGET_SIZE - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        data["frames_queue"].append(canvas)

        # Initialize VideoWriter after minimum frames threshold
        if data["detected_frames"] >= MIN_FRAMES_WALK and data["writer"] is None:
            out_path = os.path.join(
                OUTPUT_DIR,
                f"{os.path.splitext(VIDEO_NAME)[0]}_person{track_id}.avi"
            )
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            data["writer"] = cv2.VideoWriter(out_path, fourcc, fps, (TARGET_SIZE, TARGET_SIZE))
            for q_frame in data["frames_queue"]:
                data["writer"].write(q_frame)
            data["frames_queue"].clear()

        if data["writer"] is not None:
            data["writer"].write(canvas)

    # Close writers for people not detected in the current frame
    missing_ids = set(people_data.keys()) - current_ids
    for pid in missing_ids:
        data = people_data[pid]
        if data.get("writer"):
            data["writer"].release()
            data["writer"] = None
        people_data.pop(pid)

    if canvas is not None:
        cv2.imshow("Tracked People Cropped", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Finish
cap.release()
for pid, data in people_data.items():
    if data.get("writer"):
        data["writer"].release()
cv2.destroyAllWindows()

print(" Finished saving all people who walked for at least 4 seconds")
