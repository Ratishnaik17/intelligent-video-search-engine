# src/detect.py

import os
import json
import cv2
from ultralytics import YOLO
from tqdm import tqdm


# -----------------------------------
# Load YOLOv8 Model
# -----------------------------------
MODEL = YOLO("yolov8n.pt")


# -----------------------------------
# Detect + Save Bounding Boxes
# -----------------------------------
def detect_objects(
    frame_folder="data/frames",
    save_file="results/metadata/frame_objects.json",
    annotated_folder="results/annotated_frames",
    conf=0.35
):

    os.makedirs("results/metadata", exist_ok=True)
    os.makedirs(annotated_folder, exist_ok=True)

    # Load frame files
    files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not files:
        print("No frames found.")
        return

    all_data = []

    print(f"Running YOLOv8 on {len(files)} frames...\n")

    for file in tqdm(files):

        path = os.path.join(frame_folder, file)

        # -----------------------------------
        # Run YOLO Detection
        # -----------------------------------
        results = MODEL.predict(
            source=path,
            conf=conf,
            save=False,
            verbose=False
        )

        result = results[0]
        names = result.names

        # -----------------------------------
        # Save Annotated Image with Boxes
        # -----------------------------------
        annotated_img = result.plot()

        # Convert RGB -> BGR for OpenCV save
        annotated_img = cv2.cvtColor(
            annotated_img,
            cv2.COLOR_RGB2BGR
        )

        save_annotated_path = os.path.join(
            annotated_folder,
            file
        )

        cv2.imwrite(save_annotated_path, annotated_img)

        # -----------------------------------
        # If No Detections
        # -----------------------------------
        if result.boxes is None or len(result.boxes) == 0:

            all_data.append({
                "frame": path,
                "annotated_frame": save_annotated_path,
                "objects": [],
                "counts": {},
                "total_objects": 0
            })

            continue

        # -----------------------------------
        # Extract Classes
        # -----------------------------------
        class_ids = result.boxes.cls.cpu().numpy().tolist()

        objects = [names[int(i)] for i in class_ids]

        # Count Objects
        counts = {}

        for obj in objects:
            counts[obj] = counts.get(obj, 0) + 1

        all_data.append({
            "frame": path,
            "annotated_frame": save_annotated_path,
            "objects": sorted(list(set(objects))),
            "counts": counts,
            "total_objects": len(objects)
        })

    # -----------------------------------
    # Save Metadata
    # -----------------------------------
    with open(save_file, "w") as f:
        json.dump(all_data, f, indent=4)

    print("\nDetection metadata + bounding boxes saved.")
    print(f"Saved annotated images in: {annotated_folder}")


# -----------------------------------
# Run Directly
# -----------------------------------
if __name__ == "__main__":
    detect_objects()