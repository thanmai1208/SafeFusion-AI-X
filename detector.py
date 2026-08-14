import cv2
import os
import json
import time
import numpy as np
from ultralytics import YOLO

print("=== DETECTOR STARTED ===")

# -------------------------------------------------
# SafeFusion AI X Pro
# Fast YOLO + ByteTrack + Lane Detection + HUD
# -------------------------------------------------

# Load YOLO model
model = YOLO("yolo11n.pt")

# Paths
input_video = "videos/input_video.mp4"
output_video = "output/detected_video.mp4"

os.makedirs("output", exist_ok=True)

# Open video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Keep same resolution and FPS
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), True)

# COCO classes
target_classes = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Analytics
unique_persons = set()
unique_vehicles = set()
warned_objects = set()
previous_heights = {}
trajectory_history = {}

# Vehicle type counters
car_count = 0
truck_count = 0
bus_count = 0
motorcycle_count = 0
bicycle_count = 0

highest_risk = "LOW"
collision_warnings = 0
frame_count = 0
current_ttc = None


confidence_sum = 0.0
confidence_count = 0

# Faster processing
frame_skip = 2
frame_index = 0

def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=80,
        maxLineGap=30,
    )

    lane_mask = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            # Works for all OpenCV versions
            x1, y1, x2, y2 = line.reshape(4)

            cv2.line(
                lane_mask,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

    return lane_mask
# -------------------------------------------------
# Video Processing Loop
# -------------------------------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_index += 1

    # Skip alternate frames for speed
    if frame_index % frame_skip != 0:
        out.write(frame)
        continue

    frame_count += 1

    # Lane overlay
    lane_overlay = detect_lane(frame)
    frame = cv2.addWeighted(frame, 1.0, lane_overlay, 0.6, 0)

    # YOLO + ByteTrack
    results = model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    imgsz=416,      # smaller image = faster processing
    conf=0.35,      # fewer weak detections
    verbose=False,
    classes=[0, 1, 2, 3, 5, 7],  # only detect relevant objects
)

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Confidence statistics
            confidence_sum += conf
            confidence_count += 1

            if cls not in target_classes or conf < 0.30:
                continue

            if box.id is None:
                continue

            track_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = target_classes[cls]

            # Unique counting by vehicle type
            if cls == 0:
                unique_persons.add(track_id)
            elif cls == 1:
                bicycle_count += 1
                unique_vehicles.add(track_id)
            elif cls == 2:
                car_count += 1
                unique_vehicles.add(track_id)
            elif cls == 3:
                motorcycle_count += 1
                unique_vehicles.add(track_id)
            elif cls == 5:
                bus_count += 1
                unique_vehicles.add(track_id)
            elif cls == 7:
                truck_count += 1
                unique_vehicles.add(track_id)

            # Distance estimation
            box_height = y2 - y1

            if box_height > 300:
                distance = 3
            elif box_height > 180:
                distance = 6
            else:
                distance = 10

            risk = "LOW"
            action = "SAFE"
            color = (0, 255, 0)

            current_height = box_height

            if track_id in previous_heights:

                previous_height = previous_heights[track_id]

                relative_speed = abs(current_height - previous_height) * fps / 100.0

                if relative_speed > 0.1:

                    ttc = distance / relative_speed
                    current_ttc = ttc

                    if ttc < 2.0:
                        risk = "HIGH"
                        action = "BRAKE IMMEDIATELY"
                        color = (0, 0, 255)
                        highest_risk = "HIGH"

                        if track_id not in warned_objects:
                            collision_warnings += 1
                            warned_objects.add(track_id)

                    elif ttc < 4.0:
                        risk = "MEDIUM"
                        action = "SLOW DOWN"
                        color = (0, 165, 255)

                        if highest_risk != "HIGH":
                            highest_risk = "MEDIUM"

                    else:
                        risk = "LOW"
                        action = "SAFE"
                        color = (0, 255, 0)

                else:
                    ttc = None

            else:
                ttc = None

            previous_heights[track_id] = current_height
                        # -------------------------------------------------
            # Trajectory Prediction
            # -------------------------------------------------
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if track_id not in trajectory_history:
                trajectory_history[track_id] = []

            trajectory_history[track_id].append((center_x, center_y))

            if len(trajectory_history[track_id]) > 20:
                trajectory_history[track_id].pop(0)

            # Draw past trajectory
            for i in range(1, len(trajectory_history[track_id])):
                cv2.line(
                    frame,
                    trajectory_history[track_id][i - 1],
                    trajectory_history[track_id][i],
                    (0, 255, 255),
                    2,
                )

            # Predict future trajectory
            if len(trajectory_history[track_id]) >= 2:
                p1 = trajectory_history[track_id][-2]
                p2 = trajectory_history[track_id][-1]

                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]

                future_x = p2[0] + dx * 8
                future_y = p2[1] + dy * 8

                cv2.arrowedLine(
                    frame,
                    p2,
                    (future_x, future_y),
                    (255, 0, 255),
                    3,
                    tipLength=0.3,
                )

                cv2.circle(
                    frame,
                    (future_x, future_y),
                    12,
                    (0, 0, 255),
                    -1,
                )

            # -------------------------------------------------
            # Draw Bounding Box
            # -------------------------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{label} ID:{track_id}",
                (x1, y1 - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            cv2.putText(
                frame,
                f"Distance: {distance}m",
                (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            cv2.putText(
                frame,
                f"Risk: {risk}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            if ttc is not None:
                cv2.putText(
                    frame,
                    f"TTC: {ttc:.2f}s",
                    (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.putText(
                frame,
                action,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3,
            )

    # -------------------------------------------------
    # Premium Tesla HUD
    # -------------------------------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 170), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(
        frame,
        "SafeFusion AI X Pro",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    status_color = (0, 255, 0)
    if highest_risk == "MEDIUM":
        status_color = (0, 165, 255)
    elif highest_risk == "HIGH":
        status_color = (0, 0, 255)

    cv2.putText(
        frame,
        f"Pedestrians : {len(unique_persons)}",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Vehicles    : {len(unique_vehicles)}",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Risk Level  : {highest_risk}",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )

    if current_ttc is not None:
        cv2.putText(
            frame,
            f"TTC         : {current_ttc:.2f}s",
            (25, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
        )

    if highest_risk == "HIGH":
        cv2.rectangle(frame, (width - 240, 15), (width - 20, 55), (0, 0, 255), -1)
        cv2.putText(
            frame,
            "COLLISION ALERT",
            (width - 225, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    elif highest_risk == "MEDIUM":
        cv2.rectangle(frame, (width - 220, 15), (width - 20, 55), (0, 165, 255), -1)
        cv2.putText(
            frame,
            "SLOW DOWN",
            (width - 185, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # Write processed frame
    out.write(frame)

# -------------------------------------------------
# Finish
# -------------------------------------------------
cap.release()
out.release()
avg_conf = (
    round((confidence_sum / confidence_count) * 100, 1)
    if confidence_count > 0
    else 0.0
)

stats = {
    "unique_pedestrians": int(len(unique_persons)),
    "unique_vehicles": int(len(unique_vehicles)),

    "cars": int(car_count),
    "trucks": int(truck_count),
    "buses": int(bus_count),
    "motorcycles": int(motorcycle_count),
    "bicycles": int(bicycle_count),

    "highest_risk": highest_risk,
    "collision_warnings": int(collision_warnings),
    "video_frames": int(frame_count),
    "estimated_ttc": round(current_ttc, 2) if current_ttc is not None else None,
    "fps": round(fps, 2),
    "average_confidence": avg_conf,
    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "input_video": input_video
}

with open("output/stats.json", "w") as f:
    json.dump(stats, f, indent=4)

print("=== STATS GENERATED ===")
print(stats)

print("SafeFusion AI processing completed successfully!")
print("Output saved to:", output_video)