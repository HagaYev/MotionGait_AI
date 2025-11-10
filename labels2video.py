import cv2
import pandas as pd
import numpy as np

# Path to your CSV file
CSV_PATH = r"Robotic_Leg\csv_data" # choose your files

# Joint connections for drawing the leg skeleton
connections = [
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),

    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),

    ("left_hip", "right_hip")  # Connection between hips
]

# Load the CSV data
df = pd.read_csv(CSV_PATH)
print(f" Loaded {len(df)} frames from {CSV_PATH}")

# Compute the Z range for normalization
z_cols = [col for col in df.columns if col.endswith("_z")]
z_min, z_max = df[z_cols].min().min(), df[z_cols].max().max()

def get_color_from_z(z):
    """Convert Z value to a color — red for closer, blue for farther"""
    if np.isnan(z):
        return (255, 255, 255)
    z_norm = (z - z_min) / (z_max - z_min + 1e-6)
    # Red = close (low Z), Blue = far (high Z)
    return (int(255 * z_norm), 0, int(255 * (1 - z_norm)))

# Visualization window size
width, height = 400, 400

# Iterate through each frame
for i, row in df.iterrows():
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Collect all available joint points in the frame
    points = {}
    for name in ["left_hip", "right_hip", "left_knee", "right_knee",
                 "left_ankle", "right_ankle", "left_heel", "right_heel",
                 "left_foot_index", "right_foot_index"]:
        x = row.get(f"{name}_x")
        y = row.get(f"{name}_y")
        z = row.get(f"{name}_z")
        if not np.isnan(x) and not np.isnan(y):
            points[name] = (int(x), int(y), z)

    # Draw points based on depth (Z value)
    for name, (x, y, z) in points.items():
        color = get_color_from_z(z)
        cv2.circle(img, (x, y), 6, color, cv2.FILLED)
        cv2.putText(img, name.replace("_", " "), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    # Draw skeleton lines with color based on average depth between joints
    for p1, p2 in connections:
        if p1 in points and p2 in points:
            x1, y1, z1 = points[p1]
            x2, y2, z2 = points[p2]
            avg_z = (z1 + z2) / 2
            color = get_color_from_z(avg_z)
            cv2.line(img, (x1, y1), (x2, y2), color, 2)

    # Display the frame
    cv2.imshow("Leg Motion Visualization", img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(" Visualization finished.")
