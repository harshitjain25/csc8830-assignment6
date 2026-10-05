"""Planar four-view boundary mapping. Run: python sfm_planar.py.

Click the same four physical back-cover corners in every view. Ignore the spine,
book thickness, and pages. View 1 defines the reference image coordinates.
"""
import csv
from pathlib import Path

import cv2
import numpy as np


CLICK_ORDER = "1 = top-left | 2 = top-right | 3 = bottom-right | 4 = bottom-left"


def draw_text(image, text, position, color=(255, 255, 255), scale=0.6):
    """Outline text so it remains readable over the photograph."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def select_points(image, view, validation=False):
    """Select fitting corners or separate validation points in original coordinates."""
    required = 3 if validation else 4
    kind = "validation points" if validation else "back-cover corners"
    instructions = ("Validation point 1 | Validation point 2 | Validation point 3"
                    if validation else CLICK_ORDER)
    height, width = image.shape[:2]
    scale = min(1.0, 1100 / width, 650 / height)
    shown_width = max(1, round(width * scale))
    shown_height = max(1, round(height * scale))
    preview = cv2.resize(image, (shown_width, shown_height))
    header_height = 165
    points = []
    confirmation_requested = False
    message = "Click inside this window to focus keyboard input."
    window = f"Select {kind}: {view}"

    def render():
        canvas = np.zeros((shown_height + header_height, max(850, shown_width), 3), np.uint8)
        canvas[header_height:, :shown_width] = preview
        draw_text(canvas, f"{view}: {kind}; same physical points in every view", (10, 25))
        draw_text(canvas, instructions, (10, 52), scale=0.55)
        draw_text(canvas, f"R = reset | Enter/Space = accept {required} points | Esc = quit", (10, 79))
        draw_text(canvas, f"Selected: {len(points)}/{required} | {message}", (10, 106), scale=0.5)
        # Mouse confirmation also works when Cocoa does not deliver key events.
        cv2.rectangle(canvas, (10, 120), (240, 155), (60, 90, 60), -1)
        draw_text(canvas, f"Accept {required} points (click)", (18, 144), scale=0.5)
        for number, (x, y) in enumerate(points, 1):
            position = (round(x * shown_width / width),
                        round(y * shown_height / height) + header_height)
            cv2.circle(canvas, position, 6, (0, 255, 255), -1)
            draw_text(canvas, str(number), (position[0] + 8, position[1] - 8), (0, 255, 255))
        cv2.imshow(window, canvas)

    def on_mouse(event, x, y, flags, parameter):
        nonlocal confirmation_requested
        if event == cv2.EVENT_LBUTTONDOWN and 10 <= x <= 240 and 120 <= y <= 155:
            confirmation_requested = True
            return
        if (event == cv2.EVENT_LBUTTONDOWN and len(points) < required
                and 0 <= x < shown_width and header_height <= y < header_height + shown_height):
            # Undo preview resizing and header offset before saving coordinates.
            points.append((min(width - 1, x * width / shown_width),
                           min(height - 1, (y - header_height) * height / shown_height)))
            render()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    print(f"{view}: {instructions}")
    if validation:
        print("Choose three distinct interior details, not the four fitting corners.")
    try:
        render()
        while True:
            # Poll continuously so OpenCV dispatches macOS keyboard/UI events.
            key = cv2.waitKeyEx(20)
            if key >= 0:
                key &= 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (ord('r'), ord('R')):
                points.clear()
                confirmation_requested = False
                message = "Selection reset."
                render()
            elif key in (3, 10, 13, 32) or confirmation_requested:
                # Cocoa may report keypad Enter as 3; LF/CR and Space also work.
                confirmation_requested = False
                if len(points) != required:
                    message = f"Select exactly {required} points first."
                    print(message)
                    render()
                    continue
                selected = np.asarray(points, dtype=np.float32)
                if not validation and (not cv2.isContourConvex(selected.reshape(-1, 1, 2))
                                       or abs(cv2.contourArea(selected)) < 1):
                    message = "Invalid corner order/boundary. Press R and retry."
                    print(message)
                    render()
                    continue
                return selected
    finally:
        # Closing the window manually is also a safe cancellation.
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def save_csv(path, columns, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Cannot save image: {path}")


def reconstruct(images, points, results, validation_points):
    """Map source corners into view1 and save comparison results."""
    reference = points["view1"]
    projections = {}
    rows = []
    validation_rows = []
    for view in ("view2", "view3", "view4"):
        # The surface is planar, so corresponding points obey a homography:
        # s * [x', y', 1]^T = H * [x, y, 1]^T.
        # This maps image coordinates; it does not recover 3D or camera poses.
        homography, _ = cv2.findHomography(points[view], reference, method=0)
        if homography is None or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
            raise ValueError(f"Cannot estimate a valid homography for {view}.")
        print(f"\nHomography {view} -> view1:")
        print(np.array2string(homography, precision=6, suppress_small=True))
        projected = cv2.perspectiveTransform(points[view].reshape(-1, 1, 2), homography).reshape(-1, 2)
        if not np.isfinite(projected).all():
            raise ValueError(f"Invalid projected coordinates for {view}.")
        projections[view] = projected
        errors = np.linalg.norm(projected - reference, axis=1)
        for number, (source, predicted, target, error) in enumerate(
                zip(points[view], projected, reference, errors), 1):
            rows.append([view, number, *map(float, source), *map(float, predicted),
                         *map(float, target), float(error)])
        print(f"Corner fitting error — mean for {view}: {np.mean(errors):.6f} pixels")
        # These three extra correspondences are never passed to findHomography.
        predicted_validation = cv2.perspectiveTransform(
            validation_points[view].reshape(-1, 1, 2), homography).reshape(-1, 2)
        if not np.isfinite(predicted_validation).all():
            raise ValueError(f"Invalid validation projections for {view}.")
        actual_validation = validation_points["view1"]
        independent_errors = np.linalg.norm(predicted_validation - actual_validation, axis=1)
        for number, (source, predicted, actual, error) in enumerate(zip(
                validation_points[view], predicted_validation, actual_validation, independent_errors), 1):
            validation_rows.append([view, number, *map(float, source), *map(float, predicted),
                                    *map(float, actual), float(error)])
        print(f"Independent validation error — mean for {view}: {np.mean(independent_errors):.6f} pixels")

    columns = ["view", "point_id", "source_x", "source_y", "projected_x", "projected_y",
               "reference_x", "reference_y", "reprojection_error"]
    print("\nCorner fitting error")
    print(" ".join(f"{name:>18}" for name in columns))
    for row in rows:
        print(f"{row[0]:>18} {row[1]:>18}" + "".join(f" {value:>18.6f}" for value in row[2:]))
    print(f"Corner fitting error — overall mean: {np.mean([row[-1] for row in rows]):.6f} pixels")
    # Four corners determine H exactly. Testing these fitting points usually gives
    # near-zero error; independent extra correspondences are needed for validation.
    print("Note: these four corners also fit H; near-zero errors are not independent validation.")
    validation_columns = ["view", "validation_point", "source_x", "source_y",
                          "predicted_x", "predicted_y", "actual_x", "actual_y", "reprojection_error"]
    print("\nIndependent validation error")
    print(" ".join(f"{name:>18}" for name in validation_columns))
    for row in validation_rows:
        print(f"{row[0]:>18} {row[1]:>18}" + "".join(f" {value:>18.6f}" for value in row[2:]))
    print(f"Independent validation error — overall mean: "
          f"{np.mean([row[-1] for row in validation_rows]):.6f} pixels")
    results.mkdir(parents=True, exist_ok=True)
    save_csv(results / "sfm_validation_results.csv", validation_columns, validation_rows)
    clicked_rows = [[view, number, float(x), float(y)]
                    for view, coordinates in points.items()
                    for number, (x, y) in enumerate(coordinates, 1)]
    save_csv(results / "sfm_clicked_points.csv", ["view", "point_id", "x", "y"], clicked_rows)
    save_csv(results / "sfm_reprojection_results.csv", columns, rows)

    reference_image = images["view1"].copy()
    for number, (x, y) in enumerate(reference, 1):
        position = (round(float(x)), round(float(y)))
        cv2.circle(reference_image, position, 8, (0, 255, 255), -1)
        draw_text(reference_image, str(number), (position[0] + 10, position[1] - 10), (0, 255, 255))
    save_image(results / "sfm_reference_points.png", reference_image)

    boundary_image = images["view1"].copy()
    boundaries = [("view1 reference", reference, (0, 255, 0), 9),
                  ("view2 projected", projections["view2"], (255, 0, 0), 7),
                  ("view3 projected", projections["view3"], (0, 255, 255), 5),
                  ("view4 projected", projections["view4"], (255, 0, 255), 2)]
    # Nested stroke widths make coincident boundaries visible without moving them.
    for index, (label, coordinates, color, thickness) in enumerate(boundaries):
        contour = np.rint(coordinates).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(boundary_image, [contour], True, color, thickness, cv2.LINE_AA)
        draw_text(boundary_image, label, (10, 30 + index * 30), color)
    save_image(results / "sfm_reconstructed_boundary.png", boundary_image)
    print(f"\nSaved coordinates, reprojection results, and images in {results}")


def main():
    project = Path(__file__).resolve().parent
    try:
        images = {}
        # Load all images before asking for clicks, so missing inputs fail early.
        for number in range(1, 5):
            view = f"view{number}"
            path = project / "sfm_images" / f"{view}.png"
            if not path.is_file():
                raise ValueError(f"Missing image: {path}")
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            images[view] = image
        points = {}
        validation_points = {}
        for view, image in images.items():
            selected = select_points(image, view)
            if selected is None:
                print("Selection cancelled. Existing outputs were not changed.")
                return
            points[view] = selected
            validation_selected = select_points(image, view, validation=True)
            if validation_selected is None:
                print("Selection cancelled. Existing outputs were not changed.")
                return
            validation_points[view] = validation_selected
        reconstruct(images, points, project / "results", validation_points)
    except (ValueError, OSError, cv2.error) as error:
        raise SystemExit(f"Error: {error}")


if __name__ == "__main__":
    main()
