"""
gesture_recognition.py
Phase 1 — MediaPipe hand landmark extraction + CLAHE preprocessing + rule-based gesture classifier.

CRITICAL (M1 ARM64): hands.process() must ALWAYS be called from the main thread.
Never move this module's methods into a thread or subprocess.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from dotenv import load_dotenv
import os

load_dotenv()

LOW_LIGHT_THRESHOLD = int(os.getenv("LOW_LIGHT_THRESHOLD", "30"))
GESTURE_STABILITY_FRAMES = int(os.getenv("GESTURE_STABILITY_FRAMES", "10"))


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
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    processed = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return processed, low_light


# ── MediaPipe Hands ────────────────────────────────────────────────────────────

class GestureRecognizer:
    """
    Wraps MediaPipe Hands. Must be instantiated once in main() and all
    calls to process() must happen on the main thread.
    """

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        # model_complexity=0 (lite) is faster on M1 and sufficient for rule-based classification
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.buffer = GestureBuffer(GESTURE_STABILITY_FRAMES)

    def process_frame(self, frame: np.ndarray):
        """
        Run MediaPipe on a BGR frame. Returns (results, landmarks_list).
        landmarks_list is a list of 21 landmark objects (or None if no hand found).
        MUST be called from main thread only.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw hand landmarks on frame in-place. Returns annotated frame."""
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame

    def get_landmarks(self, results) -> list | None:
        """Extract the 21 landmark objects from results. Returns None if no hand."""
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0].landmark
        return None

    def classify_and_stabilize(self, results) -> str | None:
        """
        Classify gesture from results, push through stability buffer.
        Returns stable gesture name (str) or None.
        """
        lm = self.get_landmarks(results)
        raw_gesture = classify_gesture(lm) if lm else None
        return self.buffer.update(raw_gesture)

    def close(self):
        self.hands.close()


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

    Landmark reference:
        0  = Wrist
        4  = Thumb tip      3  = Thumb IP
        8  = Index tip      6  = Index PIP
        12 = Middle tip     10 = Middle PIP
        16 = Ring tip       14 = Ring PIP
        20 = Pinky tip      18 = Pinky PIP

    Returns one of: 'THUMBS_UP', 'THUMBS_DOWN', 'OPEN_PALM', 'CLOSED_FIST',
                    'PEACE', 'THREE_FINGERS', or None.
    """
    if lm is None:
        return None

    index_ext = _is_finger_extended(8, 6, lm)
    middle_ext = _is_finger_extended(12, 10, lm)
    ring_ext = _is_finger_extended(16, 14, lm)
    pinky_ext = _is_finger_extended(20, 18, lm)

    # Thumbs Up: thumb tip clearly above index MCP, all 4 fingers curled
    if (lm[4].y < lm[5].y) and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "THUMBS_UP"

    # Thumbs Down: thumb tip clearly below wrist, all 4 fingers curled
    if (lm[4].y > lm[0].y) and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "THUMBS_DOWN"

    # Open Palm: all 5 fingers extended
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "OPEN_PALM"

    # Three Fingers (emergency): index + middle + ring extended, pinky curled
    if index_ext and middle_ext and ring_ext and not pinky_ext:
        return "THREE_FINGERS"

    # Peace / V Sign: index + middle extended, ring + pinky curled
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "PEACE"

    # Closed Fist: all fingertips curled below MCP joints, thumb not pointing down
    fist_index  = lm[8].y  > lm[5].y
    fist_middle = lm[12].y > lm[9].y
    fist_ring   = lm[16].y > lm[13].y
    fist_pinky  = lm[20].y > lm[17].y
    if fist_index and fist_middle and fist_ring and fist_pinky and lm[4].y <= lm[0].y:
        return "CLOSED_FIST"

    return None


# ── Stability Buffer ───────────────────────────────────────────────────────────

class GestureBuffer:
    """
    Only returns a gesture after STABILITY_FRAMES consecutive identical readings.
    Prevents jitter / false triggers.
    """

    def __init__(self, stability_frames: int = 10):
        self._stability = stability_frames
        self._buffer: deque = deque(maxlen=stability_frames)

    def update(self, gesture: str | None) -> str | None:
        self._buffer.append(gesture)
        if len(self._buffer) < self._stability:
            return None
        if len(set(self._buffer)) == 1 and self._buffer[0] is not None:
            return self._buffer[0]
        return None
