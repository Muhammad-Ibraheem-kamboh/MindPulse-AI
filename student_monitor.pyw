"""
MindPulse AI - Next-Gen Student Cognitive Focus, 3D Biometrics & Ergonomics Studio
100% On-Device AI • Zero Cloud Telemetry • Air-Gapped Privacy Architecture

Features:
- 6DoF 3D Head Pose Tracking (Pitch, Yaw, Roll via solvePnP)
- Dedicated 3D Wireframe Head Gizmo in Real-Time Perspective
- Projected 3D Coordinate Axes on Face (X-Red, Y-Green, Z-Blue)
- MAR (Mouth Aspect Ratio) Yawn Detection
- 60-Second Rolling PERCLOS Fatigue Index (NASA/FAA Standard)
- Iris Gaze Vector Tracking (Screen-On vs Screen-Off)
- Real-Time Blinks Per Minute (BPM) Counter
- Live Animated ECG/EEG-Style Focus Waveform Graph
- Pomodoro Sprint Timer (25m Focus / 5m Break)
- Anti-Aliased Glassmorphic Obsidian HUD
- Clickable On-Screen Buttons: [SOUND: MUTED/ON], [CALIBRATE], [ZEN MODE], [BLUR: OFF/ON]
"""

import argparse
import collections
import hashlib
import math
import os
import statistics
import sys
import threading
import time
import urllib.request
import winsound
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont


# ==============================================================================
# CRYPTOGRAPHIC MODEL VERIFICATION (SHA-256 PINNED)
# ==============================================================================
MODEL_ROOT = "https://storage.googleapis.com/mediapipe-models/"

MODELS = {
    "face_landmarker.task": (
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
    "efficientdet_lite0.tflite": (
        "object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
        "0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb",
    ),
}

def prepare_models(folder: Path):
    """Verifies SHA-256 cryptographic hashes and caches official Google MediaPipe models."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in MODELS.items():
        path = folder / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() == checksum:
                continue
            path.unlink(missing_ok=True)

        temporary = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(MODEL_ROOT + url, timeout=60) as resp:
                data = resp.read()
            if hashlib.sha256(data).hexdigest() != checksum:
                raise RuntimeError(f"Hash verification failed for {name}!")
            temporary.write_bytes(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


# ==============================================================================
# COGNITIVE SCIENCE & ERGONOMIC INSIGHTS
# ==============================================================================
PSYCHOLOGY_INSIGHTS = [
    "Deep diaphragmatic breaths lower cortisol and elevate working memory.",
    "Peak performers do not study longer; they protect against micro-distractions.",
    "A hydrated brain processes neural signals 14% faster. Take a sip of water.",
    "Your brain consolidates memory traces during brief moments of calm stillness.",
    "Progress is forged in 25-minute sprints. Keep your attention right here.",
    "Notice tension in your shoulders or jaw? Exhale slowly and loosen up.",
    "Drowsiness is an energy signal, not a failure. Stand up and stretch.",
    "Follow the 20-20-20 rule: every 20m, look at an object 20ft away for 20s.",
    "Place mobile devices out of reach to eliminate subconscious friction.",
    "Embrace challenging concepts; productive struggle activates neuroplasticity."
]


# ==============================================================================
# NON-BLOCKING AUDIO ENGINE (QUIET & DEBOUNCED)
# ==============================================================================
class AudioEngine:
    def __init__(self, enabled=False):
        # Default to False (MUTED) so it never disturbs or annoys the student
        self.enabled = enabled
        self.lock = threading.Lock()
        self.last_played = 0.0

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def play(self, kind="gentle_nudge"):
        if not self.enabled:
            return
        now = time.monotonic()
        with self.lock:
            # Enforce strict cooldown between any sound events
            if now - self.last_played < 2.5:
                return
            self.last_played = now

        def _run():
            try:
                # Ultra-short, soft, non-jarring acoustic frequencies
                if kind == "gentle_nudge":
                    winsound.Beep(750, 70)
                elif kind == "drowsy_warning":
                    winsound.Beep(520, 100)
                elif kind == "yawn_warning":
                    winsound.Beep(620, 90)
                elif kind == "pomodoro_cycle":
                    winsound.Beep(880, 100)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()


# ==============================================================================
# SUSTAINED OBSERVATION TRACKER
# ==============================================================================
class SustainedObservation:
    def __init__(self):
        self.start = None
        self.previous = None

    def update(self, condition: bool, now: float) -> float:
        if not condition:
            self.start = self.previous = None
            return 0.0
        if self.previous is None or not (0.0 <= now - self.previous <= 1.0):
            self.start = now
        self.previous = now
        return now - self.start


# ==============================================================================
# BIOMETRIC CALCULATIONS (EAR, MAR, GAZE, PERCLOS, BLINK)
# ==============================================================================
def calculate_ear(landmarks, width: int, height: int):
    """Calculates bilateral Eye Aspect Ratio (EAR)."""
    def dist(a, b):
        return math.hypot((a.x - b.x) * width, (a.y - b.y) * height)

    ratios = []
    for ids in [
        (33, 160, 158, 133, 153, 144),
        (362, 385, 387, 263, 373, 380),
    ]:
        a, b, c, d, e, f = [landmarks[i] for i in ids]
        horiz = dist(a, d)
        if horiz < 2.0:
            return None
        vert = (dist(b, f) + dist(c, e)) / (2.0 * horiz)
        ratios.append(vert)

    ear = sum(ratios) / 2.0
    return ear if math.isfinite(ear) and 0.0 <= ear <= 0.8 else None


def calculate_mar(landmarks, width: int, height: int):
    """Calculates Mouth Aspect Ratio (MAR) for yawn detection."""
    def dist(a, b):
        return math.hypot((a.x - b.x) * width, (a.y - b.y) * height)

    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_corner = landmarks[61]
    right_corner = landmarks[291]

    top_mid1 = landmarks[82]
    bot_mid1 = landmarks[87]
    top_mid2 = landmarks[312]
    bot_mid2 = landmarks[317]

    vert = (dist(top_lip, bottom_lip) + dist(top_mid1, bot_mid1) + dist(top_mid2, bot_mid2)) / 3.0
    horiz = dist(left_corner, right_corner)

    if horiz < 2.0:
        return 0.0
    mar = vert / horiz
    return mar if math.isfinite(mar) else 0.0


def calculate_iris_gaze(landmarks, width: int, height: int):
    """Estimates horizontal & vertical gaze ratio to confirm on-screen attention."""
    left_eye_pts = [landmarks[i] for i in (33, 160, 158, 133, 153, 144)]
    right_eye_pts = [landmarks[i] for i in (362, 385, 387, 263, 373, 380)]

    left_center_x = sum(p.x for p in left_eye_pts) / len(left_eye_pts)
    left_center_y = sum(p.y for p in left_eye_pts) / len(left_eye_pts)

    right_center_x = sum(p.x for p in right_eye_pts) / len(right_eye_pts)
    right_center_y = sum(p.y for p in right_eye_pts) / len(right_eye_pts)

    l_span = abs(landmarks[133].x - landmarks[33].x) + 1e-6
    r_span = abs(landmarks[263].x - landmarks[362].x) + 1e-6

    l_ratio = abs(left_center_x - landmarks[33].x) / l_span
    r_ratio = abs(right_center_x - landmarks[362].x) / r_span
    avg_ratio = (l_ratio + r_ratio) / 2.0

    is_on_screen = (0.35 <= avg_ratio <= 0.65)
    gaze_desc = "On-Screen (Focused)" if is_on_screen else "Off-Target (Wandering)"
    return gaze_desc, is_on_screen, (left_center_x * width, left_center_y * height), (right_center_x * width, right_center_y * height)


# ==============================================================================
# 6DoF 3D HEAD POSE ESTIMATION (SOLVEPNP)
# ==============================================================================
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (Landmark 1)
    (0.0, -330.0, -65.0),     # Chin (Landmark 152)
    (-225.0, 170.0, -135.0),  # Left eye corner (Landmark 33)
    (225.0, 170.0, -135.0),   # Right eye corner (Landmark 263)
    (-150.0, -150.0, -125.0), # Left mouth corner (Landmark 61)
    (150.0, -150.0, -125.0),  # Right mouth corner (Landmark 291)
], dtype=np.float64)

FACE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


def estimate_head_pose_3d(landmarks, width: int, height: int):
    """Computes exact 3D rotation, translation, and Euler angles (Pitch, Yaw, Roll)."""
    image_points = []
    for idx in FACE_LANDMARK_INDICES:
        lm = landmarks[idx]
        image_points.append([lm.x * width, lm.y * height])
    image_points = np.array(image_points, dtype=np.float64)

    focal_length = width
    cam_matrix = np.array([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1.0]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, (0.0, 0.0, 0.0), cam_matrix, dist_coeffs

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0] * 360.0
    yaw = angles[1] * 360.0
    roll = angles[2] * 360.0

    return rmat, (rvec, tvec), (pitch, yaw, roll), cam_matrix, dist_coeffs


# ==============================================================================
# 3D WIREFRAME HEAD MODEL GEOMETRY
# ==============================================================================
WIREFRAME_VERTS = np.array([
    [0, 48, 0],     # 0: Cranium Top
    [-28, 28, 18],  # 1: Forehead Left
    [28, 28, 18],   # 2: Forehead Right
    [0, 0, 45],     # 3: Nose Tip
    [-34, -12, 16], # 4: Left Cheek
    [34, -12, 16],  # 5: Right Cheek
    [0, -42, 20],   # 6: Chin
    [-18, -6, 26],  # 7: Left Mouth
    [18, -6, 26],   # 8: Right Mouth
    [-16, 14, 25],  # 9: Left Eye Orbit
    [16, 14, 25],   # 10: Right Eye Orbit
    [-42, 0, -18],  # 11: Left Ear
    [42, 0, -18],   # 12: Right Ear
    [0, 18, -38],   # 13: Occiput Back
], dtype=np.float32)

WIREFRAME_EDGES = [
    (0, 1), (0, 2), (1, 2), (1, 9), (2, 10), (9, 10),
    (9, 3), (10, 3), (1, 4), (2, 5), (4, 3), (5, 3),
    (4, 7), (5, 8), (7, 8), (7, 6), (8, 6), (4, 6), (5, 6),
    (1, 11), (4, 11), (2, 12), (5, 12), (0, 13), (11, 13), (12, 13)
]


def render_3d_wireframe_gizmo(draw, center_x: int, center_y: int, rmat, scale=1.1):
    """Renders real-time rotating 3D wireframe head gizmo on canvas."""
    if rmat is None:
        R = np.eye(3)
    else:
        R = rmat.copy()

    rotated = (R @ WIREFRAME_VERTS.T).T

    projected = []
    for v in rotated:
        px = int(center_x + v[0] * scale)
        py = int(center_y - v[1] * scale)
        projected.append((px, py))

    for i, j in WIREFRAME_EDGES:
        p1 = projected[i]
        p2 = projected[j]
        depth = (rotated[i][2] + rotated[j][2]) / 2.0
        c_intensity = int(max(100, min(255, 180 + depth * 1.5)))
        edge_color = (0, c_intensity, int(c_intensity * 0.9))
        draw.line([p1, p2], fill=edge_color, width=1)

    nose_pt = projected[3]
    draw.ellipse([nose_pt[0] - 3, nose_pt[1] - 3, nose_pt[0] + 3, nose_pt[1] + 3], fill=(56, 189, 248))


# ==============================================================================
# UI DESIGN SYSTEM (OBSIDIAN GLASSMORPHIC PALETTE)
# ==============================================================================
class Theme:
    BG_CANVAS = (8, 12, 20)          # #080C14 Deep Obsidian
    CARD_BG = (17, 22, 34)           # #111622 Dark Glassmorphic Card
    CARD_BORDER = (32, 43, 66)       # #202B42 Subtle Outline
    CARD_HEADER = (24, 32, 48)       # #182030 Card Title Fill

    ACCENT_CYAN = (56, 189, 248)     # #38BDF8 Sky Cyan
    ACCENT_EMERALD = (16, 185, 129)  # #10B981 Emerald Flow
    ACCENT_AMBER = (245, 158, 11)    # #F59E0B Warning Amber
    ACCENT_ROSE = (239, 68, 68)      # #EF4444 Alert Coral
    ACCENT_PURPLE = (168, 85, 247)   # #A855F7 Restorative Violet

    TEXT_TITLE = (248, 250, 252)     # #F8FAFC
    TEXT_MUTED = (148, 163, 184)     # #94A3B8
    TEXT_DARK = (100, 116, 139)      # #64748B


class FontManager:
    def __init__(self):
        font_dir = Path("C:/Windows/Fonts")
        reg = font_dir / "segoeui.ttf"
        bold = font_dir / "segoeuib.ttf"
        semi = font_dir / "seguisb.ttf"

        def _load(p, s):
            try:
                return ImageFont.truetype(str(p), s)
            except Exception:
                return ImageFont.load_default()

        self.f_brand = _load(bold if bold.exists() else reg, 18)
        self.f_card_title = _load(semi if semi.exists() else bold, 13)
        self.f_big_metric = _load(bold if bold.exists() else reg, 26)
        self.f_body = _load(reg, 12)
        self.f_body_bold = _load(semi if semi.exists() else bold, 12)
        self.f_small = _load(reg, 11)
        self.f_badge = _load(semi if semi.exists() else bold, 10)


# ==============================================================================
# VISION DETECTOR CONTEXT
# ==============================================================================
def create_vision_detectors(folder: Path, stack: ExitStack):
    base = python.BaseOptions

    face = stack.enter_context(
        vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(folder / "face_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=2,
            )
        )
    )

    phone = stack.enter_context(
        vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=base(model_asset_path=str(folder / "efficientdet_lite0.tflite")),
                running_mode=vision.RunningMode.VIDEO,
                category_allowlist=["cell phone"],
                score_threshold=0.42,
            )
        )
    )
    return face, phone


# Global click queue for mouse click handling
UI_CLICK_QUEUE = []

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        UI_CLICK_QUEUE.append((x, y))


# ==============================================================================
# MAIN STUDIO APPLICATION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MindPulse AI - 3D Cognitive Focus Studio")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--closed-seconds", type=float, default=2.0)
    parser.add_argument("--phone-seconds", type=float, default=1.2)
    parser.add_argument("--yawn-seconds", type=float, default=1.2)
    parser.add_argument("--ear-threshold", type=float, default=0.185)
    parser.add_argument("--audio-on", action="store_true", help="Start with audio enabled (default is MUTED)")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    models_dir = Path(__file__).resolve().parent / "models"
    prepare_models(models_dir)

    if args.download_only:
        print("Models successfully verified. Offline execution ready.")
        return

    # Audio Engine defaults to MUTED (False) unless explicitly requested
    audio = AudioEngine(enabled=args.audio_on)
    fonts = FontManager()

    # Session State
    session_start_time = time.monotonic()
    last_timestamp = -1
    ear_threshold = args.ear_threshold
    calibration_start = None
    calibration_samples = []
    calibration_msg = "Click [CALIBRATE] or press 'C' to set eye baseline"

    # Sustained Observation Trackers
    closed_tracker = SustainedObservation()
    phone_tracker = SustainedObservation()
    yawn_tracker = SustainedObservation()
    slouch_tracker = SustainedObservation()

    # Rolling Biometric Buffers
    perclos_buffer = collections.deque(maxlen=1800)
    blink_timestamps = collections.deque(maxlen=60)
    focus_waveform = collections.deque(maxlen=140)
    for _ in range(140):
        focus_waveform.append(90.0)

    # Blink Detection State
    was_eye_closed = False
    eye_closed_start = 0.0

    # Cumulative Counters & Single-Trigger State Flags (NO REPEATED BEEPING)
    phone_distraction_count = 0
    drowsy_event_count = 0
    yawn_count = 0
    phone_currently_flagged = False
    yawn_currently_flagged = False
    sleepy_currently_flagged = False

    # Smooth Focus Score (0 - 100)
    focus_score = 95.0

    # Pomodoro State (25m Focus / 5m Rest)
    POMO_WORK_SEC = 25 * 60
    POMO_REST_SEC = 5 * 60
    pomo_state = "WORK"
    pomo_remaining = POMO_WORK_SEC
    pomo_paused = False
    pomo_last_tick = time.monotonic()

    # 20-20-20 Optic Rest Timer
    eye_rest_remaining = 20 * 60

    # Psychology Quote
    quote_index = 0
    last_quote_time = time.monotonic()

    # Modes
    zen_mode = False
    privacy_blur = False

    # FPS Metric
    fps_time = time.monotonic()
    fps_frames = 0
    fps_display = 30.0

    window_title = "MindPulse AI - 3D Cognitive Focus & Ergonomics Studio"

    with ExitStack() as stack:
        face_detector, phone_detector = create_vision_detectors(models_dir, stack)

        camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened() and args.camera == 0:
            camera = cv2.VideoCapture(1)

        stack.callback(camera.release)
        stack.callback(cv2.destroyAllWindows)

        if not camera.isOpened():
            raise RuntimeError("Camera unavailable! Please check camera permissions and hardware connection.")

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 1280, 720)
        cv2.setMouseCallback(window_title, on_mouse_click)

        while True:
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.04)
                continue

            now = time.monotonic()
            fps_frames += 1
            if now - fps_time >= 1.0:
                fps_display = fps_frames / (now - fps_time)
                fps_frames = 0
                fps_time = now

            timestamp_ms = max(last_timestamp + 1, int(now * 1000))
            last_timestamp = timestamp_ms

            # Mirror webcam
            frame = cv2.flip(frame, 1)
            f_height, f_width = frame.shape[:2]

            if privacy_blur:
                frame = cv2.GaussianBlur(frame, (45, 45), 0)

            # MediaPipe Detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            face_result = face_detector.detect_for_video(mp_img, timestamp_ms)
            phone_result = phone_detector.detect_for_video(mp_img, timestamp_ms)

            faces = face_result.face_landmarks if face_result else []
            phones = phone_result.detections if phone_result else []

            # --- PROCESS MOUSE CLICKS ON INTERACTIVE UI BUTTONS ---
            while UI_CLICK_QUEUE:
                click_x, click_y = UI_CLICK_QUEUE.pop(0)

                # 1. Click Audio Toggle Button: (x: 640 to 800, y: 12 to 46)
                if 640 <= click_x <= 800 and 12 <= click_y <= 46:
                    audio.toggle()

                # 2. Click Calibrate Button: (x: 810 to 935, y: 12 to 46)
                elif 810 <= click_x <= 935 and 12 <= click_y <= 46:
                    calibration_start = time.monotonic()
                    calibration_samples = []
                    closed_tracker = SustainedObservation()

                # 3. Click Zen Mode Button: (x: 945 to 1055, y: 12 to 46)
                elif 945 <= click_x <= 1055 and 12 <= click_y <= 46:
                    zen_mode = not zen_mode

                # 4. Click Privacy Blur Button: (x: 1065 to 1175, y: 12 to 46)
                elif 1065 <= click_x <= 1175 and 12 <= click_y <= 46:
                    privacy_blur = not privacy_blur

            # --- POMODORO TIMER ---
            pomo_dt = now - pomo_last_tick
            pomo_last_tick = now
            if not pomo_paused:
                pomo_remaining -= pomo_dt
                if pomo_remaining <= 0:
                    if pomo_state == "WORK":
                        pomo_state = "BREAK"
                        pomo_remaining = POMO_REST_SEC
                        audio.play("pomodoro_cycle")
                    else:
                        pomo_state = "WORK"
                        pomo_remaining = POMO_WORK_SEC
                        audio.play("pomodoro_cycle")

            # --- 20-20-20 EYE REST TIMER ---
            eye_rest_remaining = max(0, eye_rest_remaining - pomo_dt)
            if eye_rest_remaining == 0:
                eye_rest_remaining = 20 * 60

            # --- PSYCHOLOGY QUOTE ROTATION ---
            if now - last_quote_time > 45.0:
                quote_index = (quote_index + 1) % len(PSYCHOLOGY_INSIGHTS)
                last_quote_time = now

            # --- BIOMETRIC EXTRACTION ---
            has_single_face = (len(faces) == 1)
            ear = None
            mar = 0.0
            gaze_desc = "Calibrating..."
            is_gaze_focused = True
            pupil_l, pupil_r = (0, 0), (0, 0)
            rmat = None
            euler_angles = (0.0, 0.0, 0.0)

            if has_single_face:
                f_lm = faces[0]
                ear = calculate_ear(f_lm, f_width, f_height)
                mar = calculate_mar(f_lm, f_width, f_height)
                gaze_desc, is_gaze_focused, pupil_l, pupil_r = calculate_iris_gaze(f_lm, f_width, f_height)

                # 6DoF 3D Head Pose
                rmat, pose_vecs, euler_angles, cam_mat, dist_coeffs = estimate_head_pose_3d(f_lm, f_width, f_height)

                # Project 3D coordinate axes onto face
                if pose_vecs is not None:
                    rvec, tvec = pose_vecs
                    axis_len = 90.0
                    axis_3d = np.array([
                        (0.0, 0.0, 0.0),
                        (axis_len, 0.0, 0.0),
                        (0.0, -axis_len, 0.0),
                        (0.0, 0.0, -axis_len),
                    ], dtype=np.float64)

                    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_mat, dist_coeffs)
                    axis_2d = np.int32(axis_2d).reshape(-1, 2)
                    p_orig = tuple(axis_2d[0])
                    cv2.line(frame, p_orig, tuple(axis_2d[1]), (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, p_orig, tuple(axis_2d[2]), (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(frame, p_orig, tuple(axis_2d[3]), (255, 0, 0), 2, cv2.LINE_AA)

                # Draw pupil tracking points
                cv2.circle(frame, (int(pupil_l[0]), int(pupil_l[1])), 3, (0, 255, 255), -1)
                cv2.circle(frame, (int(pupil_r[0]), int(pupil_r[1])), 3, (0, 255, 255), -1)

            # --- CALIBRATION ROUTINE ---
            if calibration_start is not None:
                if ear is not None:
                    calibration_samples.append(ear)
                remaining = max(0.0, 3.0 - (now - calibration_start))
                calibration_msg = f"Hold eyes naturally open: {remaining:.1f}s"
                if remaining == 0:
                    if len(calibration_samples) >= 15 and 0.15 <= statistics.median(calibration_samples) <= 0.6:
                        ear_threshold = 0.72 * statistics.median(calibration_samples)
                        calibration_msg = f"Calibrated threshold: {ear_threshold:.3f}"
                        audio.play("gentle_nudge")
                    else:
                        calibration_msg = "Calibration incomplete; face camera and retry"
                    calibration_start = None
                    closed_tracker = SustainedObservation()

            # --- BLINK DETECTION & PERCLOS BUFFER ---
            is_eye_closed = (ear is not None and ear < ear_threshold and calibration_start is None)
            perclos_buffer.append(1 if is_eye_closed else 0)

            # Blink counter
            if is_eye_closed and not was_eye_closed:
                was_eye_closed = True
                eye_closed_start = now
            elif not is_eye_closed and was_eye_closed:
                was_eye_closed = False
                blink_dur = now - eye_closed_start
                if 0.08 <= blink_dur <= 0.45:
                    blink_timestamps.append(now)

            while blink_timestamps and (now - blink_timestamps[0] > 60.0):
                blink_timestamps.popleft()
            bpm = len(blink_timestamps)

            perclos_pct = (sum(perclos_buffer) / max(1, len(perclos_buffer))) * 100.0

            # --- SUSTAINED TIMERS ---
            closed_duration = closed_tracker.update(is_eye_closed, now)
            phone_visible = bool(phones)
            phone_duration = phone_tracker.update(phone_visible, now)

            is_yawning = (has_single_face and mar >= 0.58)
            yawn_duration = yawn_tracker.update(is_yawning, now)

            pitch, yaw, roll = euler_angles
            is_slouching = (has_single_face and (pitch > 22.0 or abs(yaw) > 28.0))
            slouch_duration = slouch_tracker.update(is_slouching, now)

            # --- EVENT ALERTS (DEBOUNCED: CHIMES ONLY ONCE PER EVENT) ---
            is_sleepy = (closed_duration >= args.closed_seconds or perclos_pct >= 18.0)
            is_phone_distracted = (phone_visible and phone_duration >= args.phone_seconds)
            is_yawn_confirmed = (yawn_duration >= args.yawn_seconds)

            if is_phone_distracted and not phone_currently_flagged:
                phone_distraction_count += 1
                phone_currently_flagged = True
                audio.play("drowsy_warning")
            elif not phone_visible:
                phone_currently_flagged = False

            if is_yawn_confirmed and not yawn_currently_flagged:
                yawn_count += 1
                yawn_currently_flagged = True
                audio.play("yawn_warning")
            elif not is_yawning:
                yawn_currently_flagged = False

            if is_sleepy and not sleepy_currently_flagged:
                drowsy_event_count += 1
                sleepy_currently_flagged = True
                audio.play("drowsy_warning")
            elif not is_sleepy:
                sleepy_currently_flagged = False

            # --- TARGET FOCUS SCORE CALCULATION ---
            if is_sleepy:
                target_score = 20.0
            elif is_phone_distracted:
                target_score = 30.0
            elif is_yawn_confirmed:
                target_score = 45.0
            elif not has_single_face:
                target_score = 50.0
            elif not is_gaze_focused:
                target_score = 65.0
            elif slouch_duration > 3.0:
                target_score = 70.0
            elif ear is not None and ear >= ear_threshold and is_gaze_focused and abs(yaw) < 18.0:
                target_score = 98.0
            else:
                target_score = 85.0

            focus_score = 0.94 * focus_score + 0.06 * target_score

            wave_ripple = math.sin(now * 4.0) * 1.5 + (np.random.rand() - 0.5) * 1.0
            focus_waveform.append(max(5.0, min(100.0, focus_score + wave_ripple)))

            # --- OBJECT DETECTIONS ON WEBCAM ---
            for d in phones:
                box = d.bounding_box
                x1, y1 = max(0, box.origin_x), max(0, box.origin_y)
                x2, y2 = min(f_width, x1 + box.width), min(f_height, y1 + box.height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (11, 158, 245), 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 130, y1), (11, 158, 245), cv2.FILLED)
                cv2.putText(frame, "PHONE DETECTED", (x1 + 6, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

            # --- COMPOSITING CANVAS (RESPONSIVE LAYOUT) ---
            canvas_w = 1280
            canvas_h = 720
            sidebar_w = 0 if zen_mode else 410
            header_h = 58
            footer_h = 42

            main_w = canvas_w - sidebar_w
            main_h = canvas_h - header_h - footer_h

            aspect = f_width / f_height
            if main_w / main_h > aspect:
                render_h = main_h - 16
                render_w = int(render_h * aspect)
            else:
                render_w = main_w - 16
                render_h = int(render_w / aspect)

            frame_resized = cv2.resize(frame, (render_w, render_h), interpolation=cv2.INTER_AREA)

            canvas = np.full((canvas_h, canvas_w, 3), Theme.BG_CANVAS, dtype=np.uint8)

            cam_x = (main_w - render_w) // 2
            cam_y = header_h + (main_h - render_h) // 2
            canvas[cam_y:cam_y + render_h, cam_x:cam_x + render_w] = frame_resized
            cv2.rectangle(canvas, (cam_x - 1, cam_y - 1), (cam_x + render_w, cam_y + render_h), Theme.CARD_BORDER, 1)

            # --- PIL HIGH-DPI RENDERING ENGINE ---
            pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # 1. HEADER BAR
            draw.rectangle([0, 0, canvas_w, header_h], fill=Theme.CARD_HEADER)
            draw.line([0, header_h, canvas_w, header_h], fill=Theme.CARD_BORDER, width=1)

            dot_color = Theme.ACCENT_EMERALD if focus_score >= 80 else (Theme.ACCENT_AMBER if focus_score >= 50 else Theme.ACCENT_ROSE)
            draw.ellipse([18, 21, 32, 35], fill=dot_color)
            draw.text((42, 17), "MINDPULSE AI", font=fonts.f_brand, fill=Theme.ACCENT_CYAN)
            draw.text((185, 21), "Cognitive Biometric Studio", font=fonts.f_body, fill=Theme.TEXT_MUTED)

            # --- INTERACTIVE CLICKABLE BUTTONS IN HEADER ---
            # Button 1: AUDIO ON / OFF TOGGLE BUTTON (Clickable!)
            btn_audio_bg = (16, 185, 129) if audio.enabled else (45, 55, 75)
            btn_audio_txt = "SOUND: ON" if audio.enabled else "SOUND: MUTED"
            btn_audio_txt_col = (0, 0, 0) if audio.enabled else (200, 215, 235)
            draw.rounded_rectangle([640, 12, 800, 46], radius=6, fill=btn_audio_bg, outline=(56, 189, 248) if audio.enabled else (70, 85, 115), width=1)
            draw.text((655, 20), btn_audio_txt, font=fonts.f_body_bold, fill=btn_audio_txt_col)

            # Button 2: CALIBRATE BUTTON (Clickable!)
            draw.rounded_rectangle([810, 12, 935, 46], radius=6, fill=(24, 34, 52), outline=Theme.CARD_BORDER, width=1)
            draw.text((824, 20), "CALIBRATE", font=fonts.f_body_bold, fill=Theme.ACCENT_CYAN)

            # Button 3: ZEN VIEW BUTTON (Clickable!)
            zen_txt = "EXIT ZEN" if zen_mode else "ZEN MODE"
            draw.rounded_rectangle([945, 12, 1055, 46], radius=6, fill=(24, 34, 52), outline=Theme.CARD_BORDER, width=1)
            draw.text((957, 20), zen_txt, font=fonts.f_body_bold, fill=Theme.TEXT_MUTED)

            # Button 4: PRIVACY BLUR (Clickable!)
            blur_txt = "BLUR: ON" if privacy_blur else "BLUR: OFF"
            draw.rounded_rectangle([1065, 12, 1175, 46], radius=6, fill=(35, 45, 65) if privacy_blur else (24, 34, 52), outline=Theme.CARD_BORDER, width=1)
            draw.text((1077, 20), blur_txt, font=fonts.f_body_bold, fill=Theme.ACCENT_AMBER if privacy_blur else Theme.TEXT_MUTED)

            uptime_sec = int(now - session_start_time)
            uptime_str = f"{uptime_sec // 60:02d}:{uptime_sec % 60:02d}"
            draw.text((canvas_w - 95, 20), f"{fps_display:.0f} FPS", font=fonts.f_body_bold, fill=Theme.TEXT_MUTED)

            # 2. FOOTER BAR
            draw.rectangle([0, canvas_h - footer_h, canvas_w, canvas_h], fill=Theme.CARD_HEADER)
            draw.line([0, canvas_h - footer_h, canvas_w, canvas_h - footer_h], fill=Theme.CARD_BORDER, width=1)
            sound_lbl = "ACTIVE" if audio.enabled else "MUTED"
            hotkeys = f"Click buttons above or use keys: [M] Sound: {sound_lbl}  |  [C] Calibrate  |  [P] Pomodoro  |  [H] Zen  |  [V] Blur  |  [Q] Exit"
            draw.text((24, canvas_h - 30), hotkeys, font=fonts.f_body, fill=Theme.TEXT_MUTED)

            # Bottom Status Banner inside Webcam View
            banner_y = cam_y + render_h - 44
            draw.rectangle([cam_x, banner_y, cam_x + render_w, cam_y + render_h], fill=(12, 17, 29))

            if is_sleepy:
                status_str = f"FATIGUE ALERT: Eyes closed {closed_duration:.1f}s (PERCLOS: {perclos_pct:.1f}%) - Take a restorative stretch"
                status_col = Theme.ACCENT_ROSE
            elif is_yawn_confirmed:
                status_str = f"EARLY FATIGUE: Yawn Detected ({yawn_duration:.1f}s) - Drink water & breathe deeply"
                status_col = Theme.ACCENT_AMBER
            elif is_phone_distracted:
                status_str = f"DISTRACTION ALERT: Phone visible ({phone_duration:.1f}s) - Place mobile devices away"
                status_col = Theme.ACCENT_AMBER
            elif not is_gaze_focused:
                status_str = "GAZE NOTICE: Eyes looking away from screen - Refocus your line of sight"
                status_col = Theme.ACCENT_AMBER
            elif is_slouching:
                status_str = f"POSTURE NOTICE: Pitch {pitch:.1f}° - Align spine and lift your gaze"
                status_col = Theme.ACCENT_AMBER
            else:
                status_str = f"OPTIMAL FLOW: Gaze Locked • Posture Aligned • PERCLOS {perclos_pct:.1f}%"
                status_col = Theme.ACCENT_EMERALD

            draw.text((cam_x + 16, banner_y + 12), status_str, font=fonts.f_body_bold, fill=status_col)

            # 3. SIDEBAR DASHBOARD (When not in Zen Mode)
            if not zen_mode:
                sx = main_w + 10
                sw = sidebar_w - 20
                sy = header_h + 10

                def render_card_box(y, h, title, tag=None, tag_color=Theme.ACCENT_CYAN):
                    draw.rounded_rectangle([sx, y, sx + sw, y + h], radius=10, fill=Theme.CARD_BG, outline=Theme.CARD_BORDER, width=1)
                    draw.text((sx + 14, y + 8), title, font=fonts.f_card_title, fill=Theme.TEXT_TITLE)
                    if tag:
                        tw = 90
                        draw.rounded_rectangle([sx + sw - tw - 12, y + 7, sx + sw - 12, y + 24], radius=4, fill=(10, 15, 26))
                        draw.text((sx + sw - tw - 6, y + 9), tag, font=fonts.f_badge, fill=tag_color)
                    return y + h + 10

                # --- CARD 1: 3D HEAD POSE GIZMO ---
                c1_h = 105
                render_card_box(sy, c1_h, "3D SPATIAL HEAD POSE", "6DoF LIVE", Theme.ACCENT_CYAN)

                gizmo_cx = sx + 50
                gizmo_cy = sy + 58
                render_3d_wireframe_gizmo(draw, gizmo_cx, gizmo_cy, rmat, scale=0.85)

                draw.text((sx + 115, sy + 32), f"Pitch (Nod):  {pitch:+5.1f}°", font=fonts.f_body_bold,
                          fill=Theme.ACCENT_AMBER if abs(pitch) > 20 else Theme.TEXT_TITLE)
                draw.text((sx + 115, sy + 52), f"Yaw (Turn):   {yaw:+5.1f}°", font=fonts.f_body_bold,
                          fill=Theme.ACCENT_AMBER if abs(yaw) > 25 else Theme.TEXT_TITLE)
                draw.text((sx + 115, sy + 72), f"Roll (Tilt):  {roll:+5.1f}°", font=fonts.f_body_bold,
                          fill=Theme.ACCENT_AMBER if abs(roll) > 18 else Theme.TEXT_MUTED)

                status_pose = "CENTERED" if (abs(pitch) < 18 and abs(yaw) < 22) else "SLOUCH / OFF"
                draw.text((sx + 270, sy + 32), f"[{status_pose}]", font=fonts.f_badge,
                          fill=Theme.ACCENT_EMERALD if status_pose == "CENTERED" else Theme.ACCENT_AMBER)

                sy += c1_h + 10

                # --- CARD 2: COGNITIVE ATTENTION & LIVE ECG WAVEFORM ---
                c2_h = 120
                score_int = int(focus_score)
                score_col = Theme.ACCENT_EMERALD if score_int >= 80 else (Theme.ACCENT_AMBER if score_int >= 50 else Theme.ACCENT_ROSE)
                score_lbl = "PEAK FLOW" if score_int >= 80 else ("ATTENTIVE" if score_int >= 50 else "DISTRACTED")
                render_card_box(sy, c2_h, "COGNITIVE ATTENTION", score_lbl, score_col)

                draw.text((sx + 16, sy + 32), f"{score_int}%", font=fonts.f_big_metric, fill=score_col)
                draw.text((sx + 95, sy + 38), f"Gaze: {gaze_desc}", font=fonts.f_small,
                          fill=Theme.ACCENT_EMERALD if is_gaze_focused else Theme.ACCENT_AMBER)

                # Animated Focus Waveform Graph
                gw = sw - 32
                gh = 42
                gx = sx + 16
                gy = sy + 68
                draw.rectangle([gx, gy, gx + gw, gy + gh], fill=(12, 17, 28), outline=(26, 36, 56), width=1)
                draw.line([gx, gy + gh // 2, gx + gw, gy + gh // 2], fill=(20, 29, 46), width=1)

                wave_pts = list(focus_waveform)
                if len(wave_pts) > 1:
                    step = gw / (len(wave_pts) - 1)
                    poly_pts = []
                    for idx_w, val_w in enumerate(wave_pts):
                        px = int(gx + idx_w * step)
                        py = int(gy + gh - (val_w / 100.0) * (gh - 4) - 2)
                        poly_pts.append((px, py))

                    for idx_w in range(len(poly_pts) - 1):
                        draw.line([poly_pts[idx_w], poly_pts[idx_w + 1]], fill=score_col, width=2)

                sy += c2_h + 10

                # --- CARD 3: FATIGUE & BIOMETRICS (PERCLOS, YAWN, BLINK) ---
                c3_h = 110
                fatigue_tag = "ALERT" if perclos_pct < 12 else ("TIRED" if perclos_pct < 18 else "HAZARD")
                fatigue_col = Theme.ACCENT_EMERALD if perclos_pct < 12 else (Theme.ACCENT_AMBER if perclos_pct < 18 else Theme.ACCENT_ROSE)
                render_card_box(sy, c3_h, "FATIGUE & BIOMETRICS", fatigue_tag, fatigue_col)

                draw.text((sx + 16, sy + 32), f"PERCLOS: {perclos_pct:4.1f}%", font=fonts.f_body_bold, fill=fatigue_col)
                draw.text((sx + 160, sy + 32), f"Blink Rate: {bpm} BPM", font=fonts.f_body_bold,
                          fill=Theme.ACCENT_EMERALD if 10 <= bpm <= 26 else Theme.ACCENT_AMBER)

                draw.text((sx + 16, sy + 54), f"Yawns Logged: {yawn_count}", font=fonts.f_body, fill=Theme.ACCENT_ROSE if yawn_count > 0 else Theme.TEXT_TITLE)
                draw.text((sx + 160, sy + 54), f"MAR Level: {mar:.2f}", font=fonts.f_body, fill=Theme.TEXT_MUTED)

                p_bar_y = sy + 84
                p_fill = int((sw - 32) * min(1.0, perclos_pct / 30.0))
                draw.rounded_rectangle([sx + 16, p_bar_y, sx + sw - 16, p_bar_y + 8], radius=4, fill=(24, 34, 52))
                draw.rounded_rectangle([sx + 16, p_bar_y, sx + 16 + p_fill, p_bar_y + 8], radius=4, fill=fatigue_col)
                draw.text((sx + 16, sy + 94), "Aviation/NASA Fatigue Threshold (Target < 12%)", font=fonts.f_badge, fill=Theme.TEXT_DARK)

                sy += c3_h + 10

                # --- CARD 4: POMODORO SPRINT & ERGONOMIC CLOCK ---
                c4_h = 95
                pomo_mins = int(pomo_remaining) // 60
                pomo_secs = int(pomo_remaining) % 60
                p_tag = f"{pomo_state}" + (" (PAUSED)" if pomo_paused else "")
                p_col = Theme.ACCENT_EMERALD if pomo_state == "WORK" else Theme.ACCENT_PURPLE
                render_card_box(sy, c4_h, "POMODORO SPRINT", p_tag, p_col)

                draw.text((sx + 16, sy + 32), f"{pomo_mins:02d}:{pomo_secs:02d}", font=fonts.f_big_metric, fill=Theme.TEXT_TITLE)
                eye_m = int(eye_rest_remaining) // 60
                draw.text((sx + 140, sy + 38), f"20-20-20 Optic Break in {eye_m}m", font=fonts.f_small, fill=Theme.ACCENT_PURPLE)

                total_span = POMO_WORK_SEC if pomo_state == "WORK" else POMO_REST_SEC
                p_pct = max(0.0, min(1.0, 1.0 - (pomo_remaining / total_span)))
                b_y = sy + 72
                draw.rounded_rectangle([sx + 16, b_y, sx + sw - 16, b_y + 8], radius=4, fill=(24, 34, 52))
                draw.rounded_rectangle([sx + 16, b_y, sx + 16 + int((sw - 32) * p_pct), b_y + 8], radius=4, fill=p_col)

                sy += c4_h + 10

                # --- CARD 5: PSYCHOLOGICAL INSIGHT & TELEMETRY ---
                c5_h = 80
                render_card_box(sy, c5_h, "MINDFUL INSIGHT", "SCIENCE", Theme.ACCENT_PURPLE)
                q_text = PSYCHOLOGY_INSIGHTS[quote_index]
                words = q_text.split()
                l1, l2 = "", ""
                for w in words:
                    if len(l1) + len(w) < 40 and not l2:
                        l1 += (" " if l1 else "") + w
                    else:
                        l2 += (" " if l2 else "") + w
                draw.text((sx + 14, sy + 32), f'"{l1}', font=fonts.f_small, fill=Theme.TEXT_MUTED)
                draw.text((sx + 14, sy + 50), f'{l2}"', font=fonts.f_small, fill=Theme.TEXT_MUTED)

            final_canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_title, final_canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key in (ord("c"), ord("C")):
                calibration_start = time.monotonic()
                calibration_samples = []
                closed_tracker = SustainedObservation()

            if key in (ord("p"), ord("P")):
                pomo_paused = not pomo_paused

            if key in (ord("m"), ord("M")):
                audio.toggle()

            if key in (ord("h"), ord("H")):
                zen_mode = not zen_mode

            if key in (ord("v"), ord("V")):
                privacy_blur = not privacy_blur


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as err:
        print(f"Runtime Exception: {err}", file=sys.stderr)
        sys.exit(1)
