import cv2
import os
import subprocess
import sys
import time
from typing import Optional, Tuple

##################### CUSTOMIZE SETTINGS #####################
# seconds before playing video
looking_away_threshold = 0.8 

# Adjust these settings to your eye shape and blink speed
EAR_CLOSED_THRESHOLD = 0.3      # lower = more strict
EYES_CLOSED_GRACE_SECONDS = 0.1


# Adjust these settings to your "safe" gaze box
GAZE_CENTER_MIN_X = 0.40        
GAZE_CENTER_MAX_X = 0.60
GAZE_CENTER_MIN_Y = 0.30
GAZE_CENTER_MAX_Y = 0.50

##################### CUSTOMIZE SETTINGS #####################


# Avoid MediaPipe/Matplotlib warnings & slow first import by forcing a writable cache dir.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MPL_CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache", "matplotlib")
try:
    os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)
except Exception:
    # If this fails, Matplotlib will fall back to a temp cache directory.
    pass

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None

if sys.platform != "darwin":
    raise SystemExit("No support for windows or linux yet.")

if mp is None or not hasattr(mp, "tasks") or not hasattr(mp.tasks, "vision"):
    raise SystemExit(
        "MediaPipe Tasks (mp.tasks.vision) is required. Install dependencies with:\n"
        "  pip install -r requirements.txt\n"
    )

def play_video(video_path: str) -> None:

    ext = os.path.splitext(video_path)[1] or ".mp4"
    runs_dir = os.path.join(SCRIPT_DIR, ".cache", "quicktime_runs")
    os.makedirs(runs_dir, exist_ok=True)

    unique_name = f"play_{int(time.time() * 1000)}{ext}"
    unique_path = os.path.join(runs_dir, unique_name)
    # `cp -c` makes an APFS clone
    try:
        subprocess.run(["cp", "-c", video_path, unique_path], check=True)
    except Exception:
        subprocess.run(["cp", video_path, unique_path], check=True)

    # Keep the cache directory from growing forever.
    try:
        items = [os.path.join(runs_dir, f) for f in os.listdir(runs_dir)]
        items = [p for p in items if os.path.isfile(p)]
        items.sort(key=lambda p: os.path.getmtime(p))
        for old in items[:-20]:
            try:
                os.remove(old)
            except Exception:
                pass
    except Exception:
        pass

    subprocess.run(['open', '-a', 'QuickTime Player', unique_path], check=True)
    applescript = '\n'.join([
        'tell application "QuickTime Player"',
        '  activate',
        '  delay 0.3',
        '  try',
        '    play front document',
        '  end try',
        'end tell',
    ])
    subprocess.run(['osascript', '-e', applescript], check=True)

def _lm_to_xy(lm, w: int, h: int) -> Tuple[float, float]:
    return (lm.x * w, lm.y * h)

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5

def _avg_xy(points: list[Tuple[float, float]]) -> Tuple[float, float]:
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    n = max(len(points), 1)
    return (sx / n, sy / n)

def _clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x

def _safe_div(n: float, d: float) -> float:
    return n / d if abs(d) > 1e-9 else 0.0

def _eye_ear(landmarks_xy: list[Tuple[float, float]], outer_i: int, inner_i: int, top_i1: int, bot_i1: int, top_i2: int, bot_i2: int) -> float:
    outer = landmarks_xy[outer_i]
    inner = landmarks_xy[inner_i]
    a = _dist(landmarks_xy[top_i1], landmarks_xy[bot_i1])
    b = _dist(landmarks_xy[top_i2], landmarks_xy[bot_i2])
    c = _dist(outer, inner)
    return _safe_div(a + b, 2.0 * c)

def _gaze_ratio(
    iris_xy: Tuple[float, float],
    outer_xy: Tuple[float, float],
    inner_xy: Tuple[float, float],
    top_xy: Tuple[float, float],
    bottom_xy: Tuple[float, float],
) -> Tuple[float, float]:
    hx = _safe_div(iris_xy[0] - outer_xy[0], inner_xy[0] - outer_xy[0])
    vy = _safe_div(iris_xy[1] - top_xy[1], bottom_xy[1] - top_xy[1])
    return (_clamp01(hx), _clamp01(vy))

def _bbox_from_indices(landmarks_xy: list[Tuple[float, float]], indices: list[int]) -> Tuple[int, int, int, int]:
    xs = [landmarks_xy[i][0] for i in indices]
    ys = [landmarks_xy[i][1] for i in indices]
    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(xs))
    y2 = int(max(ys))
    return x1, y1, x2, y2

# 1. skeletonshield.mp4 video path
videos_folder = os.path.join(SCRIPT_DIR, 'motivational_videos')
VIDEO_PATH = os.path.abspath(os.path.join(videos_folder, 'skeletonshield.mp4'))

if not os.path.exists(VIDEO_PATH):
    raise SystemExit(
        f"Expected video file at '{VIDEO_PATH}'.\n"
        "Place skeletonshield.mp4 in the 'motivational_videos' folder."
    )

print(f"Loaded video: {VIDEO_PATH}")

# 2. Initialize the webcam
cap = cv2.VideoCapture(0)

USE_MEDIAPIPE = True
looking_away = False
looking_away_start_time = 0
video_playing = False

eyes_closed_start_time: float = 0.0

FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_LANDMARKER_PATH = os.path.join(SCRIPT_DIR, "models", "face_landmarker.task")

def ensure_face_landmarker_model(model_path: str) -> bool:
    if os.path.exists(model_path):
        return True
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import ssl
        import urllib.request
        import certifi  # type: ignore
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        print(f"Downloading face landmarker model to: {model_path}")
        with urllib.request.urlopen(FACE_LANDMARKER_URL, context=ssl_ctx) as resp, open(model_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        raise SystemExit(
            "Could not download Face Landmarker model.\n"
            f"Error: {e}\n"
            f"Download it manually from: {FACE_LANDMARKER_URL}\n"
            f"And save it to: {model_path}\n"
        )

face_landmarker = None
ensure_face_landmarker_model(FACE_LANDMARKER_PATH)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)
face_landmarker = FaceLandmarker.create_from_options(options)

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    unfocused_reason: Optional[str] = None

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.time() * 1000)
    results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    if not results.face_landmarks:
        unfocused_reason = "No face"
    else:
        lms = results.face_landmarks[0]
        landmarks_xy = [(_lm_to_xy(lm, w, h)) for lm in lms]

        # Indices for FaceMesh landmarks (with iris landmarks)
        # Left eye: outer=33, inner=133, vertical pairs (159,145) and (158,144), iris=468-472
        # Right eye: outer=362, inner=263, vertical pairs (386,374) and (385,373), iris=473-477
        left_ear = _eye_ear(landmarks_xy, 33, 133, 159, 145, 158, 144)
        right_ear = _eye_ear(landmarks_xy, 362, 263, 386, 374, 385, 373)
        ear = (left_ear + right_ear) / 2.0

        left_iris = _avg_xy([landmarks_xy[i] for i in [468, 469, 470, 471, 472]])
        right_iris = _avg_xy([landmarks_xy[i] for i in [473, 474, 475, 476, 477]])

        left_h, left_v = _gaze_ratio(
            left_iris, landmarks_xy[33], landmarks_xy[133], landmarks_xy[159], landmarks_xy[145]
        )
        right_h, right_v = _gaze_ratio(
            right_iris, landmarks_xy[362], landmarks_xy[263], landmarks_xy[386], landmarks_xy[374]
        )

        gaze_h = (left_h + right_h) / 2.0
        gaze_v = (left_v + right_v) / 2.0

        eyes_closed_now = ear < EAR_CLOSED_THRESHOLD
        gaze_away_now = (
            gaze_h < GAZE_CENTER_MIN_X
            or gaze_h > GAZE_CENTER_MAX_X
            or gaze_v < GAZE_CENTER_MIN_Y
            or gaze_v > GAZE_CENTER_MAX_Y
        )

        # Track "eyes closed" len to ignore normal blinks.
        now = time.time()
        if eyes_closed_now:
            if eyes_closed_start_time <= 0:
                eyes_closed_start_time = now
        else:
            eyes_closed_start_time = 0.0

        eyes_closed_too_long = eyes_closed_start_time > 0 and (now - eyes_closed_start_time) >= EYES_CLOSED_GRACE_SECONDS

        if eyes_closed_too_long:
            unfocused_reason = "Eyes closed"
        elif gaze_away_now:
            unfocused_reason = "Gaze away"

        # Draw eye boxes (left/right) using FaceMesh indices
        LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        lx1, ly1, lx2, ly2 = _bbox_from_indices(landmarks_xy, LEFT_EYE_IDX)
        rx1, ry1, rx2, ry2 = _bbox_from_indices(landmarks_xy, RIGHT_EYE_IDX)

        pad = 6
        cv2.rectangle(frame, (lx1 - pad, ly1 - pad), (lx2 + pad, ly2 + pad), (0, 255, 255), 2)
        cv2.rectangle(frame, (rx1 - pad, ry1 - pad), (rx2 + pad, ry2 + pad), (0, 255, 255), 2)

        # Debug overlay
        # cv2.putText(
        #     frame,
        #     f"gaze x {gaze_h:.2f} y {gaze_v:.2f}",
        #     (50, 85),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 255, 255),
        #     2,
        # )

    # Handle looking away detection
    is_unfocused = unfocused_reason is not None
    if is_unfocused:
        if not looking_away:
            looking_away = True
            looking_away_start_time = time.time()
        
        # Check if looking away for too long
        time_looking_away = time.time() - looking_away_start_time
        if time_looking_away > looking_away_threshold and not video_playing:
            video_path = VIDEO_PATH
            
            # Play video (and try to auto-start playback)
            try:
                play_video(video_path)
                video_playing = True
                print(f"Not locked in! Skeleton banging shield activated :)")
            except subprocess.CalledProcessError as e:
                video_playing = False
                print(f"Error playing video (exit {e.returncode}): {e}")
                print(f"Video path tried: {video_path}")
                print("If QuickTime didn't auto-play, mac may ask for permission.")
            except Exception as e:
                video_playing = False
                print(f"Error playing video: {e}")
                print(f"Video path tried: {video_path}")
        
        # Display warning on screen
        label = f"Unfocused ({unfocused_reason})! Lockkkkk innnnnnnnn :("
        cv2.putText(frame, label, (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Focused - reset the looking away state
        if looking_away:
            looking_away = False
            video_playing = False
        cv2.putText(frame, "Eyes Detected - I'm super proud of you :)", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Anti Reels Detector', frame)

    # Stop the program when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

try:
    if face_landmarker is not None:
        face_landmarker.close()
except Exception:
    pass