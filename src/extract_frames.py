# src/extract_frames.py

import cv2
import os


def extract_frames(video_path, output_folder="data/frames", fps=1):
    """
    Extract frames from a single video.

    Args:
        video_path (str): Path to input video
        output_folder (str): Folder to save frames
        fps (int): Frames per second to extract
    """

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps == 0:
        print(f"Invalid FPS for video: {video_path}")
        return

    frame_interval = int(video_fps / fps)

    count = 0
    saved = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"\nProcessing: {video_name}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % frame_interval == 0:
            timestamp = int(count / video_fps)

            filename = f"{video_name}_frame_{saved:05d}_{timestamp}.jpg"
            save_path = os.path.join(output_folder, filename)

            cv2.imwrite(save_path, frame)
            saved += 1

        count += 1

    cap.release()

    print(f"Saved {saved} frames from {video_name}")


def process_all_videos(video_folder="data/videos", output_folder="data/frames", fps=1):
    """
    Process all videos inside data/videos folder.
    """

    if not os.path.exists(video_folder):
        print("Video folder not found.")
        return

    video_files = [
        file for file in os.listdir(video_folder)
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        print("No video files found.")
        return

    for file in video_files:
        path = os.path.join(video_folder, file)
        extract_frames(path, output_folder, fps)


# Run directly
if __name__ == "__main__":
    process_all_videos(fps=1)