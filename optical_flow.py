"""Visualize dense Farneback optical flow for an input video."""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def motion_brightness(magnitude):
    """Scale visible motion robustly while keeping negligible motion black."""
    # Ignore tiny flow estimates (under 0.1 pixels/frame) from numerical noise.
    moving = np.isfinite(magnitude) & (magnitude > 0.1)
    brightness = np.zeros(magnitude.shape, dtype=np.uint8)
    if np.any(moving):
        # Sample moving pixels so a mostly static scene does not dominate.
        # The percentile limits outliers; the floor avoids amplifying weak noise.
        scale = max(float(np.percentile(magnitude[moving], 95)), 1.0)
        brightness[moving] = np.clip(magnitude[moving] / scale * 255, 0, 255).astype(np.uint8)
    return brightness


def labeled_panels(original, visualization):
    """Label a combined copy without modifying the original frames."""
    combined = np.hstack((original, visualization))
    width = original.shape[1]
    for label, x in (("Original", 10), ("Optical Flow", width + 10)):
        # Black outline keeps white text readable over bright or dark scenes.
        cv2.putText(combined, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combined, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return combined


def process_video(path, display=True):
    if not path.is_file():
        raise ValueError(f"Video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    writer = None
    window_open = False
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video FPS; cannot preserve timing.")
        success, first = capture.read()
        if not success:
            raise ValueError("Video has no readable frames.")
        height, width = first.shape[:2]
        # The MP4 codec requires even height; avoid silently cropping a row.
        if height % 2:
            raise ValueError("MP4 output requires an even frame height; this video's height is odd.")
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        # Anchor outputs to the project, regardless of the working directory.
        results = Path(__file__).resolve().parent / "results"
        results.mkdir(parents=True, exist_ok=True)
        output = results / f"{path.stem}_optical_flow.mp4"
        if path.resolve() == output.resolve():
            raise ValueError("Input and output paths must differ.")
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (2 * width, height))
        if not writer.isOpened():
            raise ValueError(f"Cannot create MP4: {output}")
        duration = count / fps if math.isfinite(count) and count > 1 else 30
        sample_span = min(30, max(0, duration - 1 / fps))
        targets = [max(1, round(sample_span * fps * f)) for f in (1 / 3, 2 / 3, 1)]
        saved = set()
        previous = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(first)
        hsv[..., 1] = 255
        # Preserve one output frame per input frame; initial flow is unavailable.
        writer.write(labeled_panels(first, np.zeros_like(first)))
        processed = 1
        progress_interval = max(1, round(fps * 5))
        print(f"Input: {width}x{height}, {fps:.3f} FPS. Output: {output}")
        if display:
            cv2.namedWindow("Original | Optical flow", cv2.WINDOW_NORMAL)
            window_open = True
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError("Video frame dimensions changed.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Estimate a motion vector at each pixel between consecutive frames.
            flow = cv2.calcOpticalFlowFarneback(
                previous, gray, None, pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # OpenCV uint8 hue spans 0..179; value is relative magnitude.
            hsv[..., 0] = angle * 180 / (2 * np.pi)
            hsv[..., 2] = motion_brightness(magnitude)
            visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            combined = labeled_panels(frame, visualization)
            writer.write(combined)
            for number, target in enumerate(targets, 1):
                if processed >= target and number not in saved:
                    screenshot = results / f"{path.stem}_flow_frame_{number:02d}.png"
                    if not cv2.imwrite(str(screenshot), visualization):
                        raise ValueError(f"Cannot save screenshot: {screenshot}")
                    saved.add(number)
                    print(f"Saved {screenshot.name} at {processed / fps:.2f}s")
            previous = gray
            processed += 1
            if processed % progress_interval == 0:
                print(f"Processed {processed} frames ({processed / fps:.1f}s)")
            if display:
                cv2.imshow("Original | Optical flow", combined)
                # Q hides the preview while the full video continues processing.
                if cv2.waitKey(max(1, round(1000 / fps))) & 0xFF == ord("q"):
                    cv2.destroyWindow("Original | Optical flow")
                    window_open = False
                    display = False
        if processed < 2:
            raise ValueError("Optical flow requires at least two readable frames.")
        print(f"Done: {processed} frames ({processed / fps:.2f}s).")
        if processed / fps < 30:
            print("Note: this video is shorter than the required 30 seconds.")
        if len(saved) < 3:
            print("Note: fewer than three screenshots saved; check video metadata/duration.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if window_open:
            cv2.destroyWindow("Original | Optical flow")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--no-display", action="store_true", help="Save without a preview")
    args = parser.parse_args()
    try:
        process_video(args.video.expanduser(), not args.no_display)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
