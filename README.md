# CSC 8830 Computer Vision — Assignment 6

Final submission report: [Harshit Jain — Assignment 6](report_assets/Harshit_Jain_CSC8830_Assignment6_Final_Submission_Corrected.pdf).

This assignment explores motion estimation using Python and OpenCV. Part 1 uses
dense Farneback optical flow without machine learning or deep learning. Provide
two videos, each containing at least 30 seconds of motion.

## Current folder structure

```text
csc8830-assignment6/
├── README.md
├── requirements.txt
├── .gitignore
├── optical_flow.py
├── tracking_validation.py   # Consecutive-frame feature tracking
├── sfm_planar.py            # Four-view planar homography comparison
├── videos/                 # Input videos
├── frames/                 # Selected tracking frame pairs
├── sfm_images/              # Four book back-cover photographs
├── results/                # Output videos and screenshots
└── report_assets/          # Report assets
```

## Installation on macOS

Open a terminal in the project folder, then run:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies: `opencv-python`, `numpy`, and `matplotlib`. Optical flow uses OpenCV
and NumPy; Matplotlib is available for assignment figures. Run `deactivate` to
leave the virtual environment.

## Run optical flow

Place your input videos in `videos/`, then run each separately:

```bash
python optical_flow.py videos/video1.mp4
python optical_flow.py videos/video2.mp4
```

The entire video is processed, including at least the first 30 seconds when the
input is long enough. Shorter clips are processed with a terminal notice.
Progress is printed about every five seconds of video time.

The preview displays the original on the left and optical flow on the right.
Press **Q** to hide the preview while processing continues. To save without a
preview window:

```bash
python optical_flow.py videos/video1.mp4 --no-display
```

Consecutive frames are converted to grayscale. Farneback estimates a motion
vector at each pixel. The HSV visualization uses motion direction as hue,
saturation 255, and normalized motion magnitude as value. Magnitude is normalized
within each frame, so brightness indicates relative motion strength rather than
absolute speed across frames.

## Outputs

Outputs are saved in `results/` beside the script, even when launched from another
working directory. Input paths are relative to your terminal's working directory.

```text
results/video1_optical_flow.mp4
results/video1_flow_frame_01.png
results/video1_flow_frame_02.png
results/video1_flow_frame_03.png
```

The MP4 preserves the original FPS and height. Each panel retains the original
frame dimensions, so the side-by-side video has twice the original width. The
first flow panel is black because there is no preceding frame. Audio is not
included. Running the same input again replaces its outputs.
The MP4 codec requires an even frame height; odd-height inputs produce a clear
error to prevent silent cropping.

The PNGs contain the flow visualization at approximately 10, 20, and 30 seconds.
Shorter videos use timestamps spread across their available duration. Actual
timestamps are printed. Very short clips may reuse a frame; use the required
30-second inputs for distinct samples.

## Run tracking validation (Part 1)

```bash
python tracking_validation.py videos/video1.mp4
python tracking_validation.py videos/video2.mp4
```

The script samples consecutive frame pairs approximately once per second within
the first 30 seconds and selects the largest mean absolute grayscale difference.
This is a motion proxy; illumination changes can also increase the score.
Frame numbers and timestamps are printed. It converts both frames to grayscale,
detects corners with `cv2.goodFeaturesToTrack()`, and tracks them with pyramidal
Lucas–Kanade using `cv2.calcOpticalFlowPyrLK()`. Only successful, finite tracks
are retained. No GUI window is required.

For `video1.mp4`, it creates:

```text
frames/video1_frame_a.png
frames/video1_frame_b.png
results/video1_tracking_points.csv
results/video1_tracking_validation.png
```

The CSV columns are `point_id`, `x_frame1`, `y_frame1`, `x_frame2`, `y_frame2`,
`dx`, `dy`, and `motion_magnitude`. Coordinates are in each original frame:
x increases rightward and y downward. Displacement is `dx = x2 - x1`,
`dy = y2 - y1`, and magnitude is `sqrt(dx^2 + dy^2)` in pixels per frame pair.
The first ten tracks are printed as a terminal table.

The visualization places frame A on the left and frame B on the right. Green
points mark initial locations, red points mark tracked locations, and lines
connect the matching points. The horizontal panel offset is only for drawing;
it is not included in the CSV displacement. Outputs are saved beside the script
and replaced when the same video is processed again.

These coordinates support comparison with the tracking equations in the report.
Lucas–Kanade locations are estimates, not independently measured ground truth;
inspect corresponding features in the saved frames to check actual locations.
The theoretical derivation and bilinear interpolation explanation are still
required in the report.

## Run the planar example (Part 2)

Place the four photographs of the same book in `sfm_images/` as `view1.png`,
`view2.png`, `view3.png`, and `view4.png`. Then run:

```bash
python sfm_planar.py
```

The images appear one at a time. Click exactly four corners of the **flat back
cover**, ignoring thickness, the spine, and protruding pages. Use this order:

1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

Identify these corners in view1, then select the same physical corners in the
same order in every other view, even if the book is rotated in the photograph.
Each click immediately adds a numbered marker. Press **R** to reset the current
image, **Enter/Space** to accept four corners, or **Esc** to cancel without replacing
existing outputs. Closing the selection window also cancels. The preview may
be resized to fit the screen; saved coordinates use the original image size.
The clickable accept button provides a fallback for macOS keyboard input.
After the corners in each view, select three distinct interior validation details
in the same physical order in all four images. These are excluded from fitting H.

View1 supplies the reference coordinate system. Because the cover is planar,
corresponding image points obey `s * [x', y', 1]^T = H * [x, y, 1]^T`.
The script uses `cv2.findHomography()` to map views 2, 3, and 4 into view1,
then uses `cv2.perspectiveTransform()` to project their corners. It prints the
three 3×3 matrices, all point comparisons, and mean reprojection errors in pixels
for each source view and overall.

Generated outputs in `results/`:

- `sfm_clicked_points.csv`: 16 selected points with columns `view`, `point_id`, `x`, `y`.
- `sfm_reprojection_results.csv`: 12 source-point comparisons with columns `view`,
  `point_id`, `source_x`, `source_y`, `projected_x`, `projected_y`, `reference_x`,
  `reference_y`, `reprojection_error`.
- `sfm_reference_points.png`: view1 with its four numbered reference corners.
- `sfm_reconstructed_boundary.png`: view1 with the reference boundary and three
  projected boundaries, identified by colored labels. Different line widths
  help show boundaries that overlap.
- `sfm_validation_results.csv`: nine independent validation comparisons, including
  source coordinates, predicted coordinates, actual view1 coordinates, and error.

Reprojection error is the Euclidean distance between each projected corner and
its reference corner. **Four corners determine the homography, so errors on those
same fitting corners are normally almost zero.** They check mapping consistency,
not independent reconstruction accuracy. The three additional interior points
provide independent validation. This example maps a planar boundary into a
reference image; it does not recover 3D coordinates or camera poses. Camera
positions/parameters and the mathematical calculations still belong in the report.

## Build the report and independent tracking cross-check

After generating the outputs for both videos and the book images, run:

```bash
python report_assets/build_report.py
```

This creates `report_assets/Harshit_Jain_CSC8830_Assignment6_Report_corrected.pdf`
and `results/video1_independent_tracking_validation.csv` and
`results/video2_independent_tracking_validation.csv`. The report includes typed
derivations, numerical calculations, source/result images, and EXIF camera data.

The tracking cross-check searches 15x15 image patches within +/-45 pixels using
normalized correlation, without using Lucas–Kanade destinations to guide the
search. Comparison locations use integer pixel centers and are automated
measurements, not manual ground truth. Correlation and peak-separation checks
flag ambiguous matches. The report includes an erroneous door-handle track as
a failure case rather than presenting it as physical motion.

Set `GITHUB_URL` near the top of `report_assets/build_report.py` to your accessible
repository link and rerun it before submission. Camera distances/angles were
not measured; the report supplies qualitative viewpoints and available EXIF
parameters without inventing calibrated poses. Submit a real working-system
screen recording separately in Classroom, along with the PDF and source images.
