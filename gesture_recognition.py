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
GESTURE_STABILITY_FRAMES = int(os.getenv("GESTURE_STABILITY_FRAMES", "6"))


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

    # ── Thumb direction (IP joint is the reliable discriminator) ─────────────
    #
    # Thumb joint chain: CMC(1) → MCP(2) → IP(3) → Tip(4)
    #
    #  Pointing UP  : y(Tip) < y(IP) < y(MCP)   →  tip has the LOWEST y value
    #  Pointing DOWN: y(Tip) > y(IP) > y(MCP)   →  tip has the HIGHEST y value
    #  Wrapped/fist : y(Tip) ≈ y(IP) or y(Tip) < y(IP)  — IP joint bends inward
    #
    # lm[9] (middle-finger MCP) is used as the palm-centre sanity check.

    thumb_up_check   = (lm[4].y < lm[3].y) and (lm[4].y < lm[9].y)
    thumb_down_check = (lm[4].y > lm[3].y) and (lm[4].y > lm[9].y)

    # ── THUMBS_UP ─────────────────────────────────────────────────────────────
    # Thumb clearly above IP + fingers in a PROPER fist (tips past MCPs).
    # Using the stricter MCP check (not just PIP) prevents false THUMBS_UP
    # during OPEN_PALM→CLOSED_FIST transition, where fingers briefly pass the
    # PIP threshold (fingers_curled=True) before reaching the MCPs.
    fist_index  = lm[8].y  > lm[5].y
    fist_middle = lm[12].y > lm[9].y
    fist_ring   = lm[16].y > lm[13].y
    fist_pinky  = lm[20].y > lm[17].y
    fingers_fisted = fist_index and fist_middle and fist_ring and fist_pinky
    if thumb_up_check and fingers_fisted:
        return "THUMBS_UP"

    # ── THUMBS_DOWN ───────────────────────────────────────────────────────────
    # Thumb clearly below IP + NOT a multi-extended-finger pose.
    #
    # WHY we don't use fingers_curled here:
    #   When you rotate your fist so the thumb points downward the finger tips
    #   can appear ABOVE their PIP joints in screen y-coords even though the
    #   fingers are physically curled.  fingers_curled therefore returns False
    #   for many valid thumbs-down positions, making detection unreliable.
    #
    # WHY "not (index_ext and middle_ext and ring_ext)" is enough:
    #   Every multi-finger gesture (OPEN_PALM, THREE_FINGERS, PEACE) requires
    #   both index AND middle AND/OR ring to be extended.  If even one of those
    #   three is curled the pose cannot be any of those gestures, so calling it
    #   THUMBS_DOWN (given the thumb is genuinely pointing down) is correct.
    #   CLOSED_FIST has fingers curled so that condition is already False.
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
    # Use the same PIP-based soft check (fingers_curled) rather than the
    # stricter MCP check.  This is more rotation-invariant: a fist held at any
    # angle will have finger tips not clearly above their PIP joints.
    # THUMBS_UP and THUMBS_DOWN are already returned above, so a curled hand
    # with a neutral/side thumb correctly lands here.
    if fingers_curled:
        return "CLOSED_FIST"

    return None


# ── Stability Buffer ───────────────────────────────────────────────────────────

class GestureBuffer:
    """
    Returns a gesture once it reaches a majority vote across the window.

    Old logic: ALL N frames must be identical — any noise or transition frame
    resets the chain, so direct gesture-to-gesture transitions never stabilise.

    New logic: count only non-None frames in the window; fire when the
    dominant gesture appears in at least VOTE_THRESHOLD of them.
    This lets up to 2 transition/noise frames exist in the window while still
    detecting the new gesture quickly.
    """

    VOTE_THRESHOLD = 5   # minimum votes needed out of STABILITY_FRAMES window

    def __init__(self, stability_frames: int = 6):
        self._stability = stability_frames
        self._buffer: deque = deque(maxlen=stability_frames)

    def update(self, gesture: str | None) -> str | None:
        self._buffer.append(gesture)
        if len(self._buffer) < self._stability:
            return None

        # Consider only frames where a gesture was actually detected
        readings = [g for g in self._buffer if g is not None]
        if not readings:
            return None

        # Find most frequent gesture in this window
        most_common = max(set(readings), key=readings.count)
        if readings.count(most_common) >= self.VOTE_THRESHOLD:
            return most_common

        return None
