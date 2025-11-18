import os
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp


def process_video(
    folder_path=r"Robotic_Leg/raw_videos_file",
    video_name,
    output_dir=r"Robotic_Leg/video/focused_frame",
    target_size=400,
    fps=30,
    min_seconds_walk=4,
    padding_factor=0.4,
    conf_threshold=0.6,
    min_landmarks=20,
    max_no_detect_frames=8,
    smoothing_alpha=0.2,
    max_move=100,
    history_len=5
):
    # Derived parameters
    min_frames_walk = int(fps * min_seconds_walk)

    # Prepare paths
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(folder_path, video_name)

    # Load models
    yolo_model = YOLO("yolov8n.pt")
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    tracker = DeepSort(max_age=max_no_detect_frames, n_init=30)

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    people_data = {}

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        canvas = None

        # YOLO detection
        results = yolo_model(frame)[0]
        dets = []
        for det in results.boxes:
            if int(det.cls[0]) == 0:  # Only people
                conf = float(det.conf[0])
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                dets.append(([x1, y1, x2, y2], conf, "person"))

        # Tracking
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

            # Pose detection
            rgb_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pose_res = pose_model.process(rgb_crop)
            if not pose_res.pose_landmarks or len(pose_res.pose_landmarks.landmark) < min_landmarks:
                continue

            h, w, _ = cropped.shape
            xs = [lm.x * w for lm in pose_res.pose_landmarks.landmark]
            ys = [lm.y * h for lm in pose_res.pose_landmarks.landmark]
            center_x = int(np.mean(xs)) + x1
            center_y = int(np.mean(ys)) + y1
            curr_width = max(xs) - min(xs)
            curr_height = max(ys) - min(ys)

            # New person init
            if track_id not in people_data:
                people_data[track_id] = {
                    "writer": None,
                    "frames_queue": deque(),
                    "detected_frames": 0,
                    "last_center": (center_x, center_y),
                    "max_width": int(curr_width),
                    "max_height": int(curr_height),
                    "center_history": deque(maxlen=history_len)
                }

            data = people_data[track_id]

            # Smoothing center
            last_cx, last_cy = data["last_center"]
            dx, dy = center_x - last_cx, center_y - last_cy
            dist = np.sqrt(dx**2 + dy**2)
            if dist > max_move:
                center_x, center_y = last_cx, last_cy

            data["center_history"].append((center_x, center_y))
            smooth_cx = int(np.mean([c[0] for c in data["center_history"]]))
            smooth_cy = int(np.mean([c[1] for c in data["center_history"]]))
            data["last_center"] = (smooth_cx, smooth_cy)

            # Update size and frame count
            data["max_width"] = max(data["max_width"], int(curr_width))
            data["max_height"] = max(data["max_height"], int(curr_height))
            data["detected_frames"] += 1

            # Crop around center
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

            scale = min(target_size / cw, target_size / ch)
            new_w, new_h = int(cw * scale), int(ch * scale)
            resized = cv2.resize(cropped_center, (new_w, new_h))

            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            x_offset = (target_size - new_w) // 2
            y_offset = (target_size - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            data["frames_queue"].append(canvas)

            # Save videos
            if data["detected_frames"] >= min_frames_walk and data["writer"] is None:
                out_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(video_name)[0]}_person{track_id}.avi"
                )
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                data["writer"] = cv2.VideoWriter(out_path, fourcc, fps, (target_size, target_size))
                for q_frame in data["frames_queue"]:
                    data["writer"].write(q_frame)
                data["frames_queue"].clear()

            if data["writer"] is not None:
                data["writer"].write(canvas)

        # Release writers for lost tracks
        missing_ids = set(people_data.keys()) - current_ids
        for pid in missing_ids:
            data = people_data[pid]
            if data.get("writer"):
                data["writer"].release()
            people_data.pop(pid)

        if canvas is not None:
            cv2.imshow("Tracked People Cropped", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    for pid, data in people_data.items():
        if data.get("writer"):
            data["writer"].release()
    cv2.destroyAllWindows()

    print("Finished saving all people who walked for at least 4 seconds.")
