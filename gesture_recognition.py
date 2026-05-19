"""
gesture_recognition.py
Phase 1 — MediaPipe hand landmark extraction + CLAHE preprocessing + rule-based gesture classifier.

Uses MediaPipe Tasks API (mediapipe 0.10.30+) — the legacy mp.solutions API was removed
from ARM64 wheels. Model file (hand_landmarker.task) is auto-downloaded on first run.

CRITICAL (M1 ARM64): detector.detect_for_video() must ALWAYS be called from the main thread.
"""

import cv2
import numpy as np
import time
import os
import urllib.request
from collections import deque
from dotenv import load_dotenv

load_dotenv()

LOW_LIGHT_THRESHOLD    = int(os.getenv("LOW_LIGHT_THRESHOLD", "30"))
GESTURE_STABILITY_FRAMES = int(os.getenv("GESTURE_STABILITY_FRAMES", "6"))

# Hand landmark model — auto-downloaded if missing
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe hand connections for manual drawing (same topology as old API)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # Thumb
    (0,5),(5,6),(6,7),(7,8),           # Index
    (5,9),(9,10),(10,11),(11,12),      # Middle
    (9,13),(13,14),(14,15),(15,16),    # Ring
    (13,17),(17,18),(18,19),(19,20),   # Pinky
    (0,17),                            # Palm base
]


def _ensure_model() -> None:
    if not os.path.exists(_MODEL_PATH):
        print("[GestureRecognizer] Downloading hand_landmarker.task (~8 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[GestureRecognizer] Model downloaded.")


# ── CLAHE Preprocessing ────────────────────────────────────────────────────────

def apply_clahe(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Apply CLAHE to the L channel of the LAB colorspace.
    Returns (processed_frame, low_light_detected).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    mean_brightness = float(np.mean(l))
    low_light = mean_brightness < LOW_LIGHT_THRESHOLD

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)

    lab_eq    = cv2.merge([l_eq, a, b])
    processed = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return processed, low_light


# ── MediaPipe Hands (Tasks API) ───────────────────────────────────────────────

class GestureRecognizer:
    """
    Wraps MediaPipe HandLandmarker (Tasks API).
    Must be instantiated once in main() and all calls to process_frame()
    must happen on the main thread.
    """

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks import python as mp_python

        _ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector   = mp_vision.HandLandmarker.create_from_options(options)
        self._mp         = mp
        self._start_time = time.perf_counter()
        self.buffer      = GestureBuffer(GESTURE_STABILITY_FRAMES)

    def _ts(self) -> int:
        """Monotonically increasing timestamp in milliseconds."""
        return int((time.perf_counter() - self._start_time) * 1000)

    def process_frame(self, frame: np.ndarray):
        """
        Run MediaPipe on a BGR frame. Returns detection result.
        MUST be called from main thread only.
        """
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._detector.detect_for_video(mp_image, self._ts())

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw hand landmarks and connections on frame. Returns annotated frame."""
        if not results.hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        for hand_lms in results.hand_landmarks:
            # Draw connections
            for s, e in _HAND_CONNECTIONS:
                x1, y1 = int(hand_lms[s].x * w), int(hand_lms[s].y * h)
                x2, y2 = int(hand_lms[e].x * w), int(hand_lms[e].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 2, cv2.LINE_AA)
            # Draw landmark dots
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1, cv2.LINE_AA)
        return frame

    def get_landmarks(self, results) -> list | None:
        """Return list of 21 landmark objects for the first hand, or None."""
        if results.hand_landmarks:
            return results.hand_landmarks[0]
        return None

    def classify_and_stabilize(self, results) -> str | None:
        """Classify gesture and push through stability buffer."""
        lm          = self.get_landmarks(results)
        raw_gesture = classify_gesture(lm) if lm else None
        return self.buffer.update(raw_gesture)

    def close(self):
        self._detector.close()


# ── Gesture Classification (Rule-Based) ───────────────────────────────────────

def _is_finger_extended(tip_idx: int, pip_idx: int, lm: list) -> bool:
    """
    Returns True if the finger tip is above (lower y) the PIP joint.
    Uses normalized coordinates — y increases downward, so tip.y < pip.y = extended.
    """
    return lm[tip_idx].y < lm[pip_idx].y


def classify_gesture(lm: list) -> str | None:
    """
    Rule-based classification of 6 gestures using MediaPipe landmark indices.

    Landmark reference (key ones used here):
        0  = Wrist
        2  = Thumb MCP (knuckle)     4  = Thumb Tip
        5  = Index MCP               6  = Index PIP      8  = Index Tip
        9  = Middle MCP             10  = Middle PIP     12 = Middle Tip
        13 = Ring MCP               14  = Ring PIP       16 = Ring Tip
        17 = Pinky MCP              18  = Pinky PIP      20 = Pinky Tip

    y=0 is TOP of frame, y=1 is BOTTOM (normalized coords).

    Returns one of: 'THUMBS_UP', 'THUMBS_DOWN', 'OPEN_PALM', 'CLOSED_FIST',
                    'PEACE', 'THREE_FINGERS', or None.
    """
    if lm is None:
        return None

    index_ext  = _is_finger_extended(8,  6,  lm)
    middle_ext = _is_finger_extended(12, 10, lm)
    ring_ext   = _is_finger_extended(16, 14, lm)
    pinky_ext  = _is_finger_extended(20, 18, lm)

    fingers_curled = not index_ext and not middle_ext and not ring_ext and not pinky_ext

    # ── Thumb direction ───────────────────────────────────────────────────────
    # THUMB_MARGIN prevents borderline fist positions from bleeding into
    # THUMBS_UP / THUMBS_DOWN. Requires the thumb to be clearly above/below
    # the IP joint, not just fractionally past it.
    THUMB_MARGIN = 0.04   # ~4% of frame height

    thumb_up_check   = (lm[4].y < lm[3].y - THUMB_MARGIN) and (lm[4].y < lm[9].y)
    thumb_down_check = (lm[4].y > lm[3].y + THUMB_MARGIN) and (lm[4].y > lm[9].y)

    # ── THUMBS_UP ─────────────────────────────────────────────────────────────
    # Uses fingers_curled (PIP check) — permissive enough for the Tasks API
    # model which gives slightly different landmark positions than the old
    # solutions model. The THUMB_MARGIN above prevents false triggers.
    if thumb_up_check and fingers_curled:
        return "THUMBS_UP"

    # ── THUMBS_DOWN ───────────────────────────────────────────────────────────
    # THUMB_MARGIN prevents closed fist (thumb wrapped) from triggering this.
    if thumb_down_check and not (index_ext and middle_ext and ring_ext):
        return "THUMBS_DOWN"

    # ── OPEN_PALM ─────────────────────────────────────────────────────────────
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "OPEN_PALM"

    # ── THREE_FINGERS (emergency) ─────────────────────────────────────────────
    if index_ext and middle_ext and ring_ext and not pinky_ext:
        return "THREE_FINGERS"

    # ── PEACE / V Sign ────────────────────────────────────────────────────────
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "PEACE"

    # ── CLOSED_FIST ───────────────────────────────────────────────────────────
    if fingers_curled:
        return "CLOSED_FIST"

    return None


# ── Stability Buffer ───────────────────────────────────────────────────────────

class GestureBuffer:
    """
    Majority-vote stability buffer. Returns a gesture once it appears in at
    least VOTE_THRESHOLD of the last N frames (ignoring None frames).
    """

    VOTE_THRESHOLD = 5

    def __init__(self, stability_frames: int = 6):
        self._stability = stability_frames
        self._buffer: deque = deque(maxlen=stability_frames)

    def update(self, gesture: str | None) -> str | None:
        self._buffer.append(gesture)
        if len(self._buffer) < self._stability:
            return None

        readings = [g for g in self._buffer if g is not None]
        if not readings:
            return None

        most_common = max(set(readings), key=readings.count)
        if readings.count(most_common) >= self.VOTE_THRESHOLD:
            return most_common

        return None

    def fill_ratio(self) -> float:
        """Return how full the buffer is (0.0 – 1.0), for the UI progress bar."""
        if self._stability == 0:
            return 0.0
        readings = [g for g in self._buffer if g is not None]
        if not readings:
            return len(self._buffer) / self._stability
        most_common = max(set(readings), key=readings.count)
        return min(1.0, readings.count(most_common) / self.VOTE_THRESHOLD)
