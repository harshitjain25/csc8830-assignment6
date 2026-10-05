"""Track features between consecutive frames: python tracking_validation.py videos/video1.mp4."""
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def validate_tracking(path):
    """Save a frame pair, Lucas-Kanade coordinates, and a tracking illustration."""
    if not path.is_file():
        raise ValueError(f"Video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has invalid FPS.")
        if not math.isfinite(count) or count < 2:
            raise ValueError("Video must report at least two frames.")
        duration = count / fps
        # Sample one consecutive pair per second, within the first 30 seconds.
        # Decode sequentially so every candidate really uses adjacent frames.
        limit = min(int(count), math.ceil(30 * fps))
        sample_interval = max(1, round(fps))
        success, previous_frame = capture.read()
        if not success:
            raise ValueError("Cannot read the first frame.")
        best_score = -1.0
        candidates = 0
        for index_b in range(1, limit):
            success, current_frame = capture.read()
            if not success:
                break
            if (index_b - 1) % sample_interval == 0:
                if previous_frame.shape != current_frame.shape:
                    raise ValueError("Candidate frames have different dimensions.")
                gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                # Larger mean absolute difference indicates more image change.
                # This is a simple motion proxy; lighting changes can affect it.
                score = float(np.mean(cv2.absdiff(gray_previous, gray_current)))
                candidates += 1
                if score > best_score:
                    best_score = score
                    index_a = index_b - 1
                    frame_a = previous_frame.copy()
                    frame_b = current_frame.copy()
            previous_frame = current_frame
        if candidates == 0:
            raise ValueError("Cannot read a consecutive frame pair.")
    finally:
        capture.release()
    if frame_a.shape != frame_b.shape:
        raise ValueError("The selected frames have different dimensions.")

    project = Path(__file__).resolve().parent
    frames_dir = project / "frames"
    results_dir = project / "results"
    frames_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    for suffix, frame in (("a", frame_a), ("b", frame_b)):
        output = frames_dir / f"{path.stem}_frame_{suffix}.png"
        if not cv2.imwrite(str(output), frame):
            raise ValueError(f"Cannot save frame: {output}")
    print(f"Selected frames {index_a} and {index_a + 1} (zero-based), "
          f"at {index_a / fps:.3f}s and {(index_a + 1) / fps:.3f}s, "
          f"because they had the highest motion score among {candidates} sampled pairs "
          f"(mean grayscale difference: {best_score:.3f}/255).")
    if best_score < 1.0:
        print("Note: even the strongest sampled pair has little image change; motion may be weak.")
    if duration < 30:
        print("Note: this video is shorter than the required 30 seconds.")

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    points_a = cv2.goodFeaturesToTrack(
        gray_a, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)
    if points_a is None:
        raise ValueError("No good feature points found in frame A; try a textured video.")
    # A pyramid supports larger displacements; each window solves local motion.
    points_b, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, points_a, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if points_b is None or status is None:
        raise ValueError("Lucas-Kanade could not track any points.")
    valid = (status.ravel() == 1) & np.isfinite(points_b.reshape(-1, 2)).all(axis=1)
    # A valid pixel location must also lie inside frame B.
    height, width = gray_b.shape
    locations_b = points_b.reshape(-1, 2)
    valid &= ((locations_b[:, 0] >= 0) & (locations_b[:, 0] <= width - 1)
              & (locations_b[:, 1] >= 0) & (locations_b[:, 1] <= height - 1))
    tracked_a = points_a.reshape(-1, 2)[valid]
    tracked_b = points_b.reshape(-1, 2)[valid]
    if len(tracked_a) == 0:
        raise ValueError("No successfully tracked points.")

    columns = ["point_id", "x_frame1", "y_frame1", "x_frame2", "y_frame2",
               "dx", "dy", "motion_magnitude"]
    rows = []
    for point_id, (a, b) in enumerate(zip(tracked_a, tracked_b), 1):
        x1, y1 = map(float, a)
        x2, y2 = map(float, b)
        dx, dy = x2 - x1, y2 - y1
        rows.append([point_id, x1, y1, x2, y2, dx, dy, math.hypot(dx, dy)])
    csv_path = results_dir / f"{path.stem}_tracking_points.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)

    # Both panels use the original pixel coordinate system. Offset B only for drawing.
    visualization = np.hstack((frame_a, frame_b))
    width = frame_a.shape[1]
    for a, b in zip(tracked_a, tracked_b):
        start = tuple(np.rint(a).astype(int))
        end = (int(round(float(b[0]))) + width, int(round(float(b[1]))))
        cv2.line(visualization, start, end, (255, 180, 0), 1, cv2.LINE_AA)
        cv2.circle(visualization, start, 4, (0, 255, 0), -1)
        cv2.circle(visualization, end, 4, (0, 0, 255), -1)
    for text, x in (("Frame A (green)", 10), ("Frame B (red)", width + 10)):
        for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(visualization, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, thickness, cv2.LINE_AA)
    image_path = results_dir / f"{path.stem}_tracking_validation.png"
    if not cv2.imwrite(str(image_path), visualization):
        raise ValueError(f"Cannot save visualization: {image_path}")

    print(f"Successfully tracked {len(rows)} of {len(points_a)} detected points.")
    print("First 10 tracked points (coordinates and displacement in pixels):")
    print(f"{'ID':>4} {'x1':>10} {'y1':>10} {'x2':>10} {'y2':>10} "
          f"{'dx':>10} {'dy':>10} {'Magnitude':>10}")
    for row in rows[:10]:
        print(f"{row[0]:>4}" + "".join(f" {value:>10.3f}" for value in row[1:]))
    print(f"Saved {csv_path}\nSaved {image_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    args = parser.parse_args()
    try:
        validate_tracking(args.video.expanduser())
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
