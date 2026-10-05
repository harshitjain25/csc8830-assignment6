"""Build the evidence-based PDF: python report_assets/build_report.py.

Reads existing assignment results, cross-checks tracking with independent template
matching, and writes the report and validation CSVs. Does not alter core scripts.
"""
import csv
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / 'report_assets'
OUTPUT = ASSETS / 'Harshit_Jain_CSC8830_Assignment6_Report_corrected.pdf'
GITHUB_URL = 'ADD ACCESSIBLE LINK BEFORE SUBMISSION'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})


def read_csv(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot read {path}')
    return image


def independent_tracks(name):
    """Find pixel correspondences without using LK destinations to guide search.

    A 15x15 patch around each initial feature is searched within +/-45 pixels
    using normalized correlation. Reference locations are integer pixel centers,
    so they do not resolve subpixel motion. These are independent measurements,
    not manually clicked ground truth. Ambiguous matches are marked explicitly.
    """
    a = read_image(PROJECT / 'frames' / f'{name}_frame_a.png')
    b = read_image(PROJECT / 'frames' / f'{name}_frame_b.png')
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    height, width = gray_a.shape
    tracks = read_csv(PROJECT / 'results' / f'{name}_tracking_points.csv')
    rows = []
    radius, search = 7, 45
    for track in tracks:
        x, y = round(float(track['x_frame1'])), round(float(track['y_frame1']))
        if min(x, y, width - 1 - x, height - 1 - y) < radius:
            continue
        patch = gray_a[y-radius:y+radius+1, x-radius:x+radius+1]
        if patch.std() < 5:
            continue
        x0, y0 = max(0, x-radius-search), max(0, y-radius-search)
        xend, yend = min(width, x+radius+search+1), min(height, y+radius+search+1)
        scores = cv2.matchTemplate(gray_b[y0:yend, x0:xend], patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        rx, ry = x0+location[0]+radius, y0+location[1]+radius
        # Exclude the best peak's immediate neighbors before testing ambiguity.
        alternatives = scores.copy()
        lx, ly = location
        alternatives[max(0, ly-3):ly+4, max(0, lx-3):lx+4] = -1
        margin = score - float(alternatives.max())
        predicted_x, predicted_y = float(track['x_frame2']), float(track['y_frame2'])
        error = float(np.hypot(predicted_x-rx, predicted_y-ry))
        rows.append({'point_id': int(track['point_id']), 'x_frame1': x, 'y_frame1': y,
                     'lk_x': predicted_x, 'lk_y': predicted_y,
                     'reference_x': rx, 'reference_y': ry,
                     'reference_dx': rx-x, 'reference_dy': ry-y,
                     'correlation': score, 'peak_margin': margin,
                     'reprojection_error': error,
                     'high_confidence': score >= .95 and margin >= .05,
                     'lk_magnitude': float(track['motion_magnitude'])})
    if not rows:
        raise ValueError(f'No independently measurable patches for {name}')
    with (PROJECT / 'results' / f'{name}_independent_tracking_validation.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return a, b, tracks, rows


class Report:
    """Simple fixed-page layout with explicit overflow checks."""
    def __init__(self, pdf):
        self.pdf = pdf
        self.page_number = 0

    def page(self, title):
        self.page_number += 1
        self.fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        self.fig.text(.075, .95, title, fontsize=15, weight='bold', va='top')
        self.fig.text(.075, .025, 'CSC 8830 Assignment 6 | Harshit Jain', fontsize=8, color='#555555')
        self.fig.text(.925, .025, str(self.page_number), fontsize=8, ha='right')
        self.y = .90

    def text(self, text, size=10, width=100, color='black'):
        lines = []
        for paragraph in text.split('\n'):
            lines.extend(textwrap.wrap(paragraph, width=width) or [''])
        height = len(lines) * size * 1.40 / (11.69*72)
        if self.y-height < .065:
            raise ValueError(f'Text overflow on page {self.page_number}')
        self.fig.text(.075, self.y, '\n'.join(lines), fontsize=size,
                      va='top', linespacing=1.4, color=color)
        self.y -= height + .015

    def heading(self, text):
        self.text(text, size=11, width=90)

    def math(self, formula):
        self.fig.text(.10, self.y, f'${formula}$', fontsize=12, va='top')
        self.y -= .048

    def table(self, headers, rows, height=.20, font=8):
        if self.y-height < .065:
            raise ValueError(f'Table overflow on page {self.page_number}')
        ax = self.fig.add_axes([.075, self.y-height, .85, height])
        ax.axis('off')
        table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center', bbox=[0,0,1,1])
        table.auto_set_font_size(False)
        table.set_fontsize(font)
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(.4)
            if row == 0:
                cell.set_facecolor('#e8eef4')
                cell.set_text_props(weight='bold')
        self.y -= height + .025

    def image(self, image, height=.28, caption=''):
        if self.y-height < .065:
            raise ValueError(f'Image overflow on page {self.page_number}')
        ax = self.fig.add_axes([.075, self.y-height, .85, height])
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        self.y -= height + .008
        if caption:
            self.text(caption, size=8, width=120)

    def end(self):
        self.pdf.savefig(self.fig, dpi=150)
        plt.close(self.fig)


def video_info(name):
    cap = cv2.VideoCapture(str(PROJECT / 'videos' / f'{name}.mp4'))
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return fps, int(count), size


def locate_saved_pair(name, frame_a, frame_b):
    """Recover actual source indices instead of assuming a cached timestamp."""
    cap = cv2.VideoCapture(str(PROJECT / 'videos' / f'{name}.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    previous = None
    try:
        for index in range(int(np.ceil(30 * fps))):
            success, frame = cap.read()
            if not success:
                break
            if previous is not None and np.array_equal(previous, frame_a) and np.array_equal(frame, frame_b):
                return index - 1
            previous = frame
    finally:
        cap.release()
    raise ValueError(f'{name}: saved frames do not match a source consecutive pair in the first 30 seconds')


def flow_example(name, second):
    cap = cv2.VideoCapture(str(PROJECT / 'results' / f'{name}_optical_flow.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(second * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f'Cannot read flow output for {name}')
    return frame


def selected_examples(rows):
    # Keep an initial feature and two distinct, confidently matched features.
    eligible = [row for row in rows if row['high_confidence']]
    if len(eligible) < 3:
        raise ValueError('Not enough high-confidence independent matches')
    selected = []
    for row in sorted(eligible, key=lambda r: r['point_id']):
        if all(np.hypot(row['x_frame1']-r['x_frame1'], row['y_frame1']-r['y_frame1']) > 35 for r in selected):
            selected.append(row)
        if len(selected) == 3:
            return selected
    return eligible[:3]


def annotated_pair(a, b, examples):
    a, b = a.copy(), b.copy()
    for row in examples:
        x, y = row['x_frame1'], row['y_frame1']
        rx, ry = row['reference_x'], row['reference_y']
        px, py = round(row['lk_x']), round(row['lk_y'])
        cv2.circle(a, (x,y), 7, (0,255,255), 2)
        cv2.putText(a, str(row['point_id']), (x+9,y-9), cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
        cv2.circle(b, (rx,ry), 7, (0,255,0), 2)
        cv2.drawMarker(b, (px,py), (255,0,255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(b, str(row['point_id']), (rx+9,ry-9), cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
    return np.hstack((a,b))


def main():
    data = {name: independent_tracks(name) for name in ('video1','video2')}
    info = {name: video_info(name) for name in data}
    clicked = read_csv(PROJECT / 'results' / 'sfm_clicked_points.csv')
    corners = {view: np.float32([[float(r['x']),float(r['y'])] for r in clicked if r['view']==view])
               for view in ('view1','view2','view3','view4')}
    validation = read_csv(PROJECT / 'results' / 'sfm_validation_results.csv')
    homographies = {view: cv2.findHomography(corners[view],corners['view1'],0)[0]
                    for view in ('view2','view3','view4')}
    book_images = {view: read_image(PROJECT / 'sfm_images' / f'{view}.png') for view in corners}
    exif = Image.open(PROJECT / 'sfm_images' / 'view1.png').getexif()
    lens = exif.get_ifd(34665)
    camera = f"{exif.get(271)} {exif.get(272)}"
    with PdfPages(OUTPUT) as pdf:
        pdf.infodict().update({'Title':'CSC 8830 Assignment 6 - corrected report', 'Author':'Harshit Jain'})
        report = Report(pdf)
        report.page('CSC 8830: Computer Vision - Assignment 6')
        report.text(f'Optical Flow, Motion Tracking, and Planar Multi-view Geometry\nHarshit Jain | Panther ID: 002916196\nGitHub repository: {GITHUB_URL}', size=12, width=80)
        report.heading('1. Purpose and reproducible execution')
        report.text('Two videos are processed with dense Farneback optical flow [1,2]. Consecutive-frame feature tracking uses pyramidal Lucas-Kanade [1,3]. Four photographs of a flat book back cover are related by homographies [4]. All methods use classical image processing, with Python, OpenCV, and NumPy. Matplotlib builds this report.')
        report.text('From the project folder on macOS:\npython3 -m venv venv\nsource venv/bin/activate\npython -m pip install -r requirements.txt\npython optical_flow.py videos/video1.mp4\npython optical_flow.py videos/video2.mp4\npython tracking_validation.py videos/video1.mp4\npython tracking_validation.py videos/video2.mp4\npython sfm_planar.py\npython report_assets/build_report.py', size=10, width=95)
        report.heading('Input video measurements')
        report.table(['Video','FPS','Frames','Duration (s)','Size (w x h)'],
                     [[name,f'{fps:.6f}',str(count),f'{count/fps:.3f}',f'{size[0]} x {size[1]}']
                      for name,(fps,count,size) in info.items()],height=.12)
        report.text('Both videos exceed 30 seconds. Optical flow processes the complete clips; this report analyzes samples within their first 30 seconds. Video length alone does not prove continuous object motion. The selected tracking pairs are dominated by stationary scene features, an important limitation discussed below.')
        report.text('Outputs: results/*_optical_flow.mp4, three flow PNGs per video, frames/*_frame_a.png and *_frame_b.png, tracking CSVs and visualizations, and the planar geometry CSVs and PNGs. The four original book PNGs are included in sfm_images/. The recorded working demonstration must be submitted separately with this PDF.')
        report.end()

        report.page('2. Optical flow: method and evidence')
        report.text('Farneback estimates a dense two-component displacement field using local polynomial image models [2]. Each frame is converted to grayscale and compared with its predecessor. Direction is atan2(dy,dx), and magnitude is sqrt(dx^2+dy^2). Image coordinates increase rightward in x and downward in y.')
        report.text('The HSV display uses hue for direction and saturation 255. Value uses the 95th percentile of magnitudes above 0.1 pixels/frame, with a minimum scale of 1 pixel/frame, then clips to 0-255. Smaller estimates remain black. This suppresses noise and reduces sensitivity to extreme magnitudes. Brightness is relative within a frame and cannot establish absolute speed differences across timestamps or videos.')
        report.image(flow_example('video1',10),height=.22,caption='Figure 1. Video1 at approximately 10 s: original and dense flow. The small floor-level vehicle is the moving object; broad background regions have weak flow.')
        report.image(flow_example('video2',10),height=.22,caption='Figure 2. Video2 at approximately 10 s: original and dense flow. Inspect the person and any locally colored regions against the darker background; flow estimates apparent image motion, not physical 3D speed.')
        report.text('In Figure 1, the vehicle region is predominantly red: the implemented hue mapping associates red with rightward motion, yellow-green with downward, cyan with leftward, and violet with upward motion. Figure 2 contains several directions and scattered background estimates, so not every bright patch identifies a moving object. Black means below the display threshold, not proof of zero motion. Camera motion, illumination, occlusion, and compression affect estimates. Mean grayscale difference is a selection heuristic, not physical velocity [1].')
        report.end()

        report.page('3. Tracking equations derived from fundamentals')
        report.text('Brightness constancy assumes the same physical point retains intensity as it moves. Let displacement between the two images be d=(dx,dy) and the time interval be dt [1,3].')
        report.math(r'I(x,y,t)=I(x+\Delta x,y+\Delta y,t+\Delta t)')
        report.text('Expand the right side about (x,y,t), retaining first-order terms:')
        report.math(r'I(x+\Delta x,y+\Delta y,t+\Delta t)\approx I+I_x\Delta x+I_y\Delta y+I_t\Delta t')
        report.text('Cancel I, divide by dt, and define u=dx/dt and v=dy/dt:')
        report.math(r'I_xu+I_yv+I_t=0')
        report.text('One pixel gives one equation for two unknowns (the aperture problem). In a local window, assume the same motion for each pixel i. For consecutive-frame displacements, set b_i=-(I_B(i)-I_A(i)) and A_i=[I_x(i),I_y(i)]. Then A d approximately equals b. Here d has units pixels/frame pair; u and v are pixels/second only if dt is expressed in seconds.')
        report.math(r'E(d)=\|Ad-b\|^2,\qquad \nabla_d E=2A^T(Ad-b)=0')
        report.math(r'(A^TA)d=A^Tb,\qquad d=(A^TA)^{-1}A^Tb')
        report.text('The inverse requires a well-conditioned gradient matrix: intensity variation in two directions. This explains selecting corners with goodFeaturesToTrack. Iterative refinement warps the second image, estimates a correction, and updates d. Image pyramids reduce large image displacement at coarser scales before refinement [1,3].')
        report.heading('Typed worked example of the normal equations')
        report.text('Illustrative gradients (not measured from these videos): A rows are [1,0], [0,1], [1,1]; temporal residual b=[2,1,3]. Thus A^T A=[[2,1],[1,2]] and A^T b=[5,4]. Solving 2dx+dy=5 and dx+2dy=4 gives dx=2, dy=1 pixels. An initial point (100,80) is predicted at (102,81), with magnitude sqrt(5)=2.236 pixels. Actual video comparisons appear on the next pages.')
        report.end()

        report.page('3.1 Normal equations evaluated on actual video pixels')
        report.text('The theoretical linear system can also be assembled directly from the saved frame pairs, without calling Lucas-Kanade. For point 1 in each video, use a 21x21 window (441 pixels), centered at its initial integer location. Central differences in frame A give Ix=(I(x+1,y)-I(x-1,y))/2 and Iy=(I(x,y+1)-I(x,y-1))/2. Set b=-(IB-IA) and accumulate the normal equations from these measured intensities [1,3].')
        report.math(r'G_{11}=\sum I_x^2,\quad G_{12}=G_{21}=\sum I_xI_y,\quad G_{22}=\sum I_y^2')
        report.math(r'c_1=-\sum I_x(I_B-I_A),\quad c_2=-\sum I_y(I_B-I_A),\quad Gd=c')
        for name in ('video1', 'video2'):
            frame_a, frame_b, tracks, measurements = data[name]
            track = tracks[0]
            x, y = round(float(track['x_frame1'])), round(float(track['y_frame1']))
            a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float64)
            b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float64)
            gx = (a[1:-1, 2:] - a[1:-1, :-2]) / 2
            gy = (a[2:, 1:-1] - a[:-2, 1:-1]) / 2
            ix = gx[y-11:y+10, x-11:x+10].ravel()
            iy = gy[y-11:y+10, x-11:x+10].ravel()
            A = np.column_stack((ix, iy))
            residual = -(b-a)[y-10:y+11, x-10:x+11].ravel()
            G, c = A.T @ A, A.T @ residual
            displacement = np.linalg.solve(G, c)
            predicted = np.array([x, y]) + displacement
            measured = next(row for row in measurements if row['point_id'] == 1)
            reference = np.array([measured['reference_x'], measured['reference_y']])
            report.heading(f'{name}: measured point 1 at ({x},{y})')
            report.text(f'G=[[{G[0,0]:.2f}, {G[0,1]:.2f}], [{G[1,0]:.2f}, {G[1,1]:.2f}]]\nc=[{c[0]:.2f}, {c[1]:.2f}]\nSolve Gd=c: d=({displacement[0]:.6f}, {displacement[1]:.6f}) pixels.\nTheoretical first-order prediction=({predicted[0]:.6f}, {predicted[1]:.6f}).\nIndependent patch location=({reference[0]},{reference[1]}); discrepancy={np.linalg.norm(predicted-reference):.6f} pixels.\nPyramidal LK prediction=({float(track["x_frame2"]):.6f}, {float(track["y_frame2"]):.6f}).', size=10, width=98)
        report.text('These are calculations from actual image intensities, not illustrative gradients. The direct first-order solve and OpenCV pyramidal iterative solve need not be identical: their gradients, interpolation, refinement, and scale handling differ. Both are compared with independently matched integer pixel positions. This supplies numerical evidence for the theoretical two-frame tracking formulation while retaining the reference-quantization limitation.')
        report.end()

        report.page('4. Bilinear interpolation: derivation and example')
        report.text('A warped point generally lies between integer pixel centers. Let x0=floor(x), y0=floor(y), a=x-x0, and b=y-y0. Define I00=I(x0,y0), I10=I(x0+1,y0), I01=I(x0,y0+1), and I11=I(x0+1,y0+1). Bilinear interpolation performs linear interpolation in each direction [5].')
        report.math(r'T=(1-a)I_{00}+aI_{10},\qquad B=(1-a)I_{01}+aI_{11}')
        report.math(r'I(x,y)=(1-b)T+bB')
        report.math(r'I(x,y)=(1-a)(1-b)I_{00}+a(1-b)I_{10}')
        report.math(r'\qquad +(1-a)bI_{01}+abI_{11}')
        report.text('The four weights sum to one and are nonnegative for 0<=a,b<=1. This gives a continuous, piecewise bilinear estimate used when sampling image intensities at subpixel positions. It does not create new measured image detail.')
        report.heading('Numerical example (illustrative intensities)')
        report.table(['Neighbor','Location','Intensity','Weight'],
                     [['I00','(10,20)','100','0.375'],['I10','(11,20)','120','0.125'],
                      ['I01','(10,21)','140','0.375'],['I11','(11,21)','160','0.125']],height=.22)
        report.text('At (x,y)=(10.25,20.5), a=0.25 and b=0.5. Interpolate the top row: T=0.75*100+0.25*120=105. Interpolate the bottom row: B=0.75*140+0.25*160=145. Interpolate vertically: I=0.5*105+0.5*145=125. The expanded formula gives 37.5+15+52.5+20=125.')
        report.heading('Two-frame tracking validation setup')
        report.text('For each saved feature p_A, Lucas-Kanade produces p_LK=p_A+d. An independent patch search measures a comparison location q_B from frame B pixels. The discrepancy is ||p_LK-q_B||. This cross-check uses a different estimation method; it is not manually annotated ground truth. The comparison locations are integer-valued, so errors below about one pixel cannot establish accurate subpixel motion.')
        report.end()

        for name in ('video1','video2'):
            a,b,tracks,measurements = data[name]
            examples = selected_examples(measurements)
            fps = info[name][0]
            index = locate_saved_pair(name, a, b)
            report.page(f'5. Consecutive-frame tracking validation: {name}')
            report.text(f'Frame pair: {index} -> {index+1}, zero-based. Timestamps: {index/fps:.6f} s -> {(index+1)/fps:.6f} s. The current script samples approximately once per second within the first 30 seconds and selects the largest mean absolute grayscale difference. It detects and successfully tracks {len(tracks)} points; a successful status is not a guarantee of accuracy.')
            report.text('Independent measurement: search each 15x15 grayscale patch in frame B within +/-45 pixels of its frame A position using TM_CCOEFF_NORMED [6]. The search never uses the LK destination. Require patch standard deviation >=5, correlation >=0.95, and a best-peak advantage >=0.05 over matches more than 3 pixels from the best peak. Uncertain matches are excluded from reported high-confidence means.')
            report.image(annotated_pair(a,b,examples),height=.23,caption=f'Figure {3 if name=="video1" else 4}. Frame A (left): yellow initial points and CSV IDs. Frame B (right): green independently matched locations and magenta LK predictions. Overlap reflects agreement at display scale.')
            report.table(['ID','A (x,y)','LK in B (x,y)','Pixel match (x,y)','NCC','Error px'],
                         [[str(r['point_id']),f"{r['x_frame1']},{r['y_frame1']}",f"{r['lk_x']:.3f},{r['lk_y']:.3f}",f"{r['reference_x']},{r['reference_y']}",f"{r['correlation']:.4f}",f"{r['reprojection_error']:.3f}"] for r in examples],height=.15,font=7)
            r=examples[0]
            dx=r['lk_x']-r['x_frame1'];dy=r['lk_y']-r['y_frame1']
            report.text(f"Worked actual-image comparison, point {r['point_id']}: dx={r['lk_x']:.6f}-{r['x_frame1']}={dx:.6f}; dy={r['lk_y']:.6f}-{r['y_frame1']}={dy:.6f}. Magnitude={np.hypot(dx,dy):.6f} pixels. Independent frame B pixel location is ({r['reference_x']},{r['reference_y']}); prediction error=sqrt(({r['lk_x']:.6f}-{r['reference_x']})^2+({r['lk_y']:.6f}-{r['reference_y']})^2)={r['reprojection_error']:.6f} pixels.",size=9,width=106)
            eligible=[r for r in measurements if r['high_confidence']]
            errors=[r['reprojection_error'] for r in eligible]
            magnitudes=[float(r['motion_magnitude']) for r in tracks]
            report.text(f'High-confidence independent matches: {len(eligible)}; mean discrepancy={np.mean(errors):.3f} pixels; median={np.median(errors):.3f} pixels. LK median magnitude over all {len(tracks)} tracks={np.median(magnitudes):.3f} pixels. Static background corners dominate these feature sets. Small displacements do not demonstrate object tracking on the moving object.',size=9,width=106)
            report.end()

        report.page('6. Tracking failure analysis and limitations')
        measurements = data['video2'][3]
        failed=next(r for r in measurements if r['point_id']==5)
        report.text(f"Video2 point 5 illustrates why LK status alone is insufficient. It begins at ({failed['x_frame1']},{failed['y_frame1']}) on the door handle. LK predicts ({failed['lk_x']:.3f},{failed['lk_y']:.3f}), a magnitude of {failed['lk_magnitude']:.3f} pixels. Independent patch matching instead finds ({failed['reference_x']},{failed['reference_y']}) with correlation {failed['correlation']:.4f}. The methods disagree by {failed['reprojection_error']:.3f} pixels. The unchanged door-handle patch supports interpreting this track as a likely mismatch, not evidence of fast physical movement.")
        a,b=data['video2'][:2]
        # The same crop coordinates in both images preserve pixel comparability.
        crop_a=a[210:290,280:340].copy();crop_b=b[210:290,280:340].copy()
        cv2.circle(crop_a,(failed['x_frame1']-280,failed['y_frame1']-210),4,(0,255,255),1)
        cv2.circle(crop_b,(failed['reference_x']-280,failed['reference_y']-210),4,(0,255,0),1)
        cv2.drawMarker(crop_b,(round(failed['lk_x'])-280,round(failed['lk_y'])-210),(255,0,255),cv2.MARKER_CROSS,8,1)
        report.image(np.hstack((crop_a,crop_b)),height=.32,caption='Figure 5. Enlarged identical-coordinate crops around the door handle: initial feature (yellow), independently matched location (green), and LK prediction (magenta).')
        report.text('Independent patch matching can also fail on repeated textures, illumination changes, occlusion, or nonrigid motion. Its confidence rules reduce ambiguity but do not establish ground truth. No manually clicked tracking measurements were available. Reported errors are disagreements between two estimators and include integer reference quantization. A manually annotated feature correspondence would provide a stronger check if required by the instructor.')
        report.text('Video1 has mean frame difference 0.793/255 at the selected pair, while video2 has 6.059/255. This can reflect brightness variation as well as motion. Because the tracked points are mostly static background features, these pairs validate small displacement estimates and expose mismatches rather than demonstrating robust moving-object trajectories. No cross-video physical speed comparison is inferred.')
        report.end()

        report.page('7. Planar object, four viewpoints, and camera data')
        report.text('The book back cover is treated as a plane. Only its four cover corners and three interior printed details are used. The spine, thickness, and protruding pages are excluded. View1 defines the projective reference coordinates. The photographs are shown below with their orientation as decoded by OpenCV.')
        # Assemble a labeled contact sheet without changing source files.
        sheet=np.full((800,1000,3),255,np.uint8)
        for i,(view,image) in enumerate(book_images.items()):
            scale=min(480/image.shape[1],350/image.shape[0])
            thumb=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
            x=(i%2)*500+(500-thumb.shape[1])//2;y=(i//2)*400+35
            sheet[y:y+thumb.shape[0],x:x+thumb.shape[1]]=thumb
            cv2.putText(sheet,view,((i%2)*500+20,(i//2)*400+25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
        report.image(sheet,height=.43,caption='Figure 6. Four source photographs. Relative positions are qualitative observations; distances and angles were not measured.')
        report.table(['View','Relative viewpoint','Decoded size (w x h)'],
                     [['view1','Oblique overhead','3024 x 4032'],['view2','Closer to overhead','3024 x 4032'],['view3','Rotated oblique','3024 x 4032'],['view4','Lower grazing angle','4032 x 3024']],height=.15)
        report.text(f"Camera metadata retained in all four PNGs: {camera}; lens '{lens.get(42036)}'; focal length {float(lens.get(37386)):.1f} mm; 35 mm equivalent {lens.get(41989)} mm; aperture f/{float(lens.get(33437)):.1f}; exposure 1/60 s; ISO {lens.get(34855)}. Views 1-3 have EXIF orientation 6; view4 has orientation 1. Stored raster size is 4032 x 3024; the table gives oriented OpenCV coordinates used by the CSVs.",size=9,width=108)
        report.end()

        report.page('8. Camera limitations and planar homography derivation')
        report.text('EXIF gives focal length in millimeters, not a calibrated focal length in pixels. Principal point, pixel focal lengths, distortion coefficients, camera-to-object distance, and metric poses were not measured. No calibrated intrinsic matrix K or metric camera position is claimed. Relative viewpoints and source images document the acquisition; exact camera positions remain unavailable. This is a planar projective boundary reconstruction, not metric 3D structure recovery [4].')
        report.math(r's[x^\prime,y^\prime,1]^T=H[x,y,1]^T')
        report.text('Write the rows of H as h11,h12,h13; h21,h22,h23; h31,h32,h33. Fix h33=1 for these fitted matrices, removing arbitrary scale. Multiplying H by the source point gives q=(X,Y,W); divide by W to obtain image coordinates.')
        report.math(r'x^\prime=\frac{h_{11}x+h_{12}y+h_{13}}{h_{31}x+h_{32}y+1}')
        report.math(r'y^\prime=\frac{h_{21}x+h_{22}y+h_{23}}{h_{31}x+h_{32}y+1}')
        report.text('Cross-multiply and rearrange each correspondence to obtain two linear equations:')
        report.math(r'xh_{11}+yh_{12}+h_{13}-xx^\prime h_{31}-yx^\prime h_{32}=x^\prime')
        report.math(r'xh_{21}+yh_{22}+h_{23}-xy^\prime h_{31}-yy^\prime h_{32}=y^\prime')
        report.text('With h=[h11,h12,h13,h21,h22,h23,h31,h32]^T, the two rows in the 8x8 system M h=c are:')
        report.text('[x, y, 1, 0, 0, 0, -x*x\', -y*x\']  -> x\'\n[0, 0, 0, x, y, 1, -x*y\', -y*y\']  -> y\'',size=10,width=95)
        report.text('Stack four nondegenerate corner correspondences, solve for h, and reshape into H. The actual implementation uses cv2.findHomography(source_corners,reference_corners,method=0), which handles the numerical estimation. No validation points enter this calculation. cv2.perspectiveTransform applies the resulting H [4].')
        r=clicked[4]
        target=corners['view1'][0];x,y=float(r['x']),float(r['y'])
        report.text(f"Numerical row example, view2 corner 1: source=({x:.6f},{y:.6f}), reference=({target[0]:.6f},{target[1]:.6f}). The x-equation includes -x*x'={-x*float(target[0]):.6f} and -y*x'={-y*float(target[0]):.6f}, with right-hand side x'={target[0]:.6f}. The remaining seven rows follow the same construction.",size=9,width=108)
        report.end()

        report.page('9. Corner correspondences and fitted matrices')
        report.table(['View','Corner','Source x','Source y'],
                     [[r['view'],r['point_id'],f"{float(r['x']):.3f}",f"{float(r['y']):.3f}"] for r in clicked],height=.37,font=8)
        report.text('Corner IDs retain the same physical order across rotations of the book. The reference coordinates are the four view1 rows above. Matrices below map each named source view into view1; entries are printed with sufficient precision for the worked example.',size=9,width=108)
        for view,H in homographies.items():
            report.text(f'{view} -> view1\n'+np.array2string(H,precision=10,suppress_small=False),size=9,width=110)
        report.end()

        report.page('10. Independent planar validation: worked calculation')
        report.text('Three additional physical details on the flat cover were manually selected in each view. The same validation point IDs identify the same details in every image. These points are excluded from homography estimation. Predicted locations are compared with the actual manually clicked reference locations in view1.')
        report.math(r'e_i=\sqrt{(x_{pred,i}-x_{actual,i})^2+(y_{pred,i}-y_{actual,i})^2}')
        first=validation[0];H=homographies['view2'];source=np.array([float(first['source_x']),float(first['source_y']),1.]);q=H@source
        predicted=q[:2]/q[2];actual=np.array([float(first['actual_x']),float(first['actual_y'])]);error=np.linalg.norm(predicted-actual)
        report.text(f'View2 validation point 1: p=[{source[0]:.6f}, {source[1]:.6f}, 1]^T.\nUsing the full-precision H printed on the previous page:\nX={H[0,0]:.10f}*{source[0]:.6f}+({H[0,1]:.10f})*{source[1]:.6f}+{H[0,2]:.10f}={q[0]:.6f}\nY={H[1,0]:.10f}*{source[0]:.6f}+{H[1,1]:.10f}*{source[1]:.6f}+({H[1,2]:.10f})={q[1]:.6f}\nW={H[2,0]:.10f}*{source[0]:.6f}+{H[2,1]:.10f}*{source[1]:.6f}+1={q[2]:.9f}',size=9,width=108)
        report.text(f"Divide by W: predicted=({predicted[0]:.6f}, {predicted[1]:.6f}).\nActual manually selected view1 location=({actual[0]:.6f}, {actual[1]:.6f}).\nResidual=({predicted[0]-actual[0]:.6f}, {predicted[1]-actual[1]:.6f}).\ne=sqrt(({predicted[0]-actual[0]:.6f})^2+({predicted[1]-actual[1]:.6f})^2)={error:.6f} pixels.\nThe saved float32 perspectiveTransform result rounds to {float(first['reprojection_error']):.6f} pixels.",size=10,width=95)
        means={view:np.mean([float(r['reprojection_error']) for r in validation if r['view']==view]) for view in homographies}
        overall=np.mean([float(r['reprojection_error']) for r in validation])
        report.table(['View','Corner fitting error (px)','Independent error (px)'],
                     [[view,'0.000',f'{value:.3f}'] for view,value in means.items()]+[['Overall','0.000',f'{overall:.3f}']],height=.18)
        report.text('Corner fitting errors are zero at the saved precision because the same four corners define H. They verify fitting consistency, not independent accuracy. Independent errors measure agreement at three held-out, manually identified points per source view. The sample is small; no universal accuracy threshold or metric 3D accuracy is inferred.')
        report.end()

        report.page('11. Planar validation correspondences and evidence')
        report.table(['View','ID','Source (x,y)','Predicted (x,y)','Actual view1 (x,y)','Error px'],
                     [[r['view'],r['validation_point'],f"{float(r['source_x']):.1f},{float(r['source_y']):.1f}",f"{float(r['predicted_x']):.1f},{float(r['predicted_y']):.1f}",f"{float(r['actual_x']):.1f},{float(r['actual_y']):.1f}",f"{float(r['reprojection_error']):.3f}"] for r in validation],height=.25,font=7)
        sheet=np.full((800,1000,3),255,np.uint8)
        for i,(view,image) in enumerate(book_images.items()):
            image=image.copy()
            if view=='view1':
                extra=[(float(r['actual_x']),float(r['actual_y'])) for r in validation if r['view']=='view2']
            else:
                extra=[(float(r['source_x']),float(r['source_y'])) for r in validation if r['view']==view]
            for label,coordinates,color in [('C',corners[view],(0,255,0)),('V',extra,(0,0,255))]:
                for number,(x,y) in enumerate(coordinates,1):
                    cv2.circle(image,(round(float(x)),round(float(y))),28,color,5)
                    cv2.putText(image,f'{label}{number}',(round(float(x))+35,round(float(y))-25),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),11)
                    cv2.putText(image,f'{label}{number}',(round(float(x))+35,round(float(y))-25),cv2.FONT_HERSHEY_SIMPLEX,3,color,7)
            scale=min(480/image.shape[1],350/image.shape[0]);thumb=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
            x=(i%2)*500+(500-thumb.shape[1])//2;y=(i//2)*400+35
            sheet[y:y+thumb.shape[0],x:x+thumb.shape[1]]=thumb
            cv2.putText(sheet,view,((i%2)*500+20,(i//2)*400+25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
        report.image(sheet,height=.43,caption='Figure 7. Fitting corners C1-C4 (green) and independent validation details V1-V3 (red). Reference validation locations come from the saved actual_x/actual_y columns. Identical labels designate physical correspondences.')
        report.text('Exact validation source/reference coordinates and errors are included in results/sfm_validation_results.csv. The table here rounds coordinates for readability; calculations use unrounded saved values.',size=9,width=108)
        report.end()

        report.page('12. Reconstructed boundary and interpretation')
        boundary = read_image(PROJECT / 'results' / 'sfm_reconstructed_boundary.png')
        # Enlarge only the legend in the report copy; original result stays intact.
        for index, (label, color) in enumerate([
                ('view1 reference', (0,255,0)), ('view2 projected', (255,0,0)),
                ('view3 projected', (0,255,255)), ('view4 projected', (255,0,255))]):
            position = (100, 130 + index * 110)
            cv2.putText(boundary, label, position, cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,0), 11)
            cv2.putText(boundary, label, position, cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)
        report.image(boundary,height=.59,caption='Figure 8. View1 reference boundary and projected boundaries from views 2, 3, and 4. Colored labels and nested line widths distinguish nearly coincident outlines. This reconstructs a planar boundary in view1 pixel coordinates, not in physical units.')
        report.text(f'The independent mean error is {overall:.3f} pixels, about {100*overall/np.hypot(3024,4032):.3f}% of the reference image diagonal. Individual errors range from {min(float(r["reprojection_error"]) for r in validation):.3f} to {max(float(r["reprojection_error"]) for r in validation):.3f} pixels. Reporting relative scale provides context; it does not establish that the accuracy is acceptable for an unspecified application.')
        report.text('Possible contributors include manual point uncertainty, imperfect planarity, image resampling, and lens distortion. The experiment does not isolate their contributions. Boundary coincidence is expected because its endpoints were used to fit H. The held-out interior correspondences provide the independent accuracy evidence. Physical 3D reconstruction would require additional geometric constraints and appropriate camera information.')
        report.end()

        report.page('13. Conclusions, references, and submission')
        report.text('Dense Farneback flow supplies apparent image-motion direction and magnitude. Consecutive-frame Lucas-Kanade tracking estimates subpixel displacement; independent image-patch comparisons show agreement on many background features and identify a likely erroneous large track. The selected pairs provide limited evidence of moving-object tracking. Four-view homographies map a planar book boundary into view1; held-out manual points give an independent mean discrepancy of 26.989 pixels. No metric 3D structure or calibrated camera poses are claimed.')
        refs=[
            '[1] OpenCV. Optical Flow. https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html',
            '[2] G. Farneback. Two-Frame Motion Estimation Based on Polynomial Expansion. SCIA 2003, pp. 363-370. DOI: 10.1007/3-540-45103-X_50.',
            '[3] B. D. Lucas and T. Kanade. An Iterative Image Registration Technique with an Application to Stereo Vision. IJCAI 1981, pp. 674-679. https://publications.ri.cmu.edu/an-iterative-image-registration-technique-with-an-application-to-stereo-vision-ijcai',
            '[4] OpenCV. Basic concepts of the homography explained with code. https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html',
            '[5] OpenCV. Geometric Image Transformations (INTER_LINEAR: bilinear interpolation). https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html',
            '[6] OpenCV. Template Matching. https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html',
            'Course-provided supplementary resource: First Principles of Computer Vision, Optical Flow | Structure from Motion | Object Tracking playlist. https://www.youtube.com/playlist?list=PL2zRqk16wsdoYzrWStffqBAoUY8XdvatV . Listed as a course resource; no specific unverified lecture content is attributed to it.'
        ]
        for reference in refs:
            report.text(reference,size=8,width=120)
        report.heading('Submission checklist')
        report.text('Add the accessible GitHub URL on page 1. Upload the repository code, README, source images, and relevant outputs. Submit this PDF and a screen-recording or properly captured working-system demonstration in Classroom. Generated flow MP4s show results, but are not by themselves proof of the complete interactive workflow. Camera positions are qualitative; if measured distances/angles are available, add them. The mathematical workouts here are typed; no photographs of handwritten worksheets are used.',size=9,width=108)
        report.end()
    print(f'Created {OUTPUT}')


if __name__ == '__main__':
    main()
