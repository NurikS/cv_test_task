import cv2
from pathlib import Path

video_dir = Path("../clips")
output_dir = video_dir / "frames"
output_dir.mkdir(exist_ok=True)

frames_per_second = 2
video_extensions = [".mov", ".mp4", ".avi"]

for video_file in video_dir.iterdir():
    if video_file.suffix.lower() not in video_extensions:
        continue

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Could not open {video_file.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps // frames_per_second))

    frame_count = 0
    saved_count = 0
    print(f"Processing {video_file.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_filename = f"{video_file.stem}_frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(output_dir / output_filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()

