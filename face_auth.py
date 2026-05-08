"""
face_auth.py
Phase 3 — Single-photo face enrollment + batch-voting authentication.

Design decisions (performance on M1 ARM64):
  - face_recognition.compare_faces() takes ~100ms per call on M1.
  - Comparison runs in a daemon thread so the main OpenCV loop is NEVER blocked.
  - We collect 5 votes (one per completed thread), majority (3/5) = authorized.
  - Result is cached for 25 frames (~1 second at 15fps) before re-evaluating.
  - Only ONE comparison thread runs at a time (_vote_in_progress flag).

Note: MediaPipe hands.process() must stay on main thread — face_recognition
      has no such restriction and is safe to run in a background thread.

Known limitations:
  - Glasses vs no-glasses: encoding shifts; re-enroll if accuracy drops.
  - Single enrolled face only (one owner.jpg).
"""

import os
import threading
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ENROLLED_FACES_DIR = os.path.join(os.path.dirname(__file__), "enrolled_faces")
ENROLLMENT_PATH = os.path.join(ENROLLED_FACES_DIR, "owner.jpg")
FACE_TOLERANCE = float(os.getenv("FACE_TOLERANCE", "0.65"))
FACE_AUTH_ENABLED = os.getenv("FACE_AUTH_ENABLED", "true").lower() == "true"

_FRAMES_PER_BATCH = 10      # trigger a new comparison every Nth frame
_VOTES_NEEDED     = 5       # votes per evaluation batch
_MAJORITY_THRESHOLD = 3     # votes needed to be "authorized" (3/5)
_CACHE_FRAMES     = 25      # frames to hold a batch result before re-evaluating


class FaceAuthenticator:
    """
    Manages face enrollment and non-blocking per-batch authorization.
    check() must be called from the main thread every frame.
    The heavy face_recognition comparison runs in a daemon thread.
    """

    def __init__(self):
        self._enrolled_encoding: np.ndarray | None = None
        self._votes: list[bool] = []
        self._last_auth_result: bool = False
        self._cache_remaining: int = 0
        self._enrolled = False
        self._auth_enabled = FACE_AUTH_ENABLED

        # Thread-safety for async comparison
        self._lock = threading.Lock()
        self._vote_in_progress = False   # True while a daemon thread is running
        self._pending_vote: bool | None = None  # result delivered by the thread

        self._load_enrollment()

    # ── Enrollment ─────────────────────────────────────────────────────────────

    def _load_enrollment(self) -> None:
        if not self._auth_enabled:
            print("[FaceAuth] Auth disabled via FACE_AUTH_ENABLED=false")
            return
        if not os.path.exists(ENROLLMENT_PATH):
            print(f"[FaceAuth] No enrolled face at {ENROLLMENT_PATH} — run enroll_face.py.")
            return
        try:
            print("[FaceAuth] Loading face_recognition (may take a few seconds)...")
            import face_recognition
            img = face_recognition.load_image_file(ENROLLMENT_PATH)
            encodings = face_recognition.face_encodings(img)
            if not encodings:
                print("[FaceAuth] WARNING: No face in owner.jpg — re-enroll.")
                return
            self._enrolled_encoding = encodings[0]
            self._enrolled = True
            print("[FaceAuth] Enrolled face loaded successfully.")
        except Exception as e:
            print(f"[FaceAuth] Failed to load enrolled face: {e}")

    def enroll(self, image_path: str) -> bool:
        try:
            import face_recognition, shutil
            os.makedirs(ENROLLED_FACES_DIR, exist_ok=True)
            img = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(img)
            if not encodings:
                print("[FaceAuth] No face found in image.")
                return False
            self._enrolled_encoding = encodings[0]
            self._enrolled = True
            shutil.copy2(image_path, ENROLLMENT_PATH)
            print(f"[FaceAuth] Enrolled face saved to {ENROLLMENT_PATH}")
            return True
        except Exception as e:
            print(f"[FaceAuth] Enrollment failed: {e}")
            return False

    # ── Authorization Check (main thread, non-blocking) ────────────────────────

    def check(self, frame: np.ndarray, frame_counter: int) -> tuple[bool, str]:
        """
        Called every frame from main loop. Never blocks.
        Fires a background thread for the heavy comparison work.
        Returns (authorized: bool, status_text: str).
        """
        if not self._auth_enabled:
            return True, "Auth Disabled"
        if not self._enrolled:
            return True, "Not Enrolled (open)"

        # ── Pick up result from the last completed thread ──────────────────────
        with self._lock:
            if self._pending_vote is not None:
                self._votes.append(self._pending_vote)
                self._pending_vote = None
                self._vote_in_progress = False

        # ── Evaluate batch once enough votes are in ────────────────────────────
        if len(self._votes) >= _VOTES_NEEDED:
            authorized = self._votes.count(True) >= _MAJORITY_THRESHOLD
            self._last_auth_result = authorized
            self._cache_remaining = _CACHE_FRAMES
            self._votes = []

        # ── Use cached result if still fresh ──────────────────────────────────
        if self._cache_remaining > 0:
            self._cache_remaining -= 1
            status = "Authorized" if self._last_auth_result else "Unauthorized"
            return self._last_auth_result, status

        # ── Fire a new comparison thread if none is running ───────────────────
        with self._lock:
            start_thread = (
                frame_counter % _FRAMES_PER_BATCH == 0
                and not self._vote_in_progress
            )
            if start_thread:
                self._vote_in_progress = True

        if start_thread:
            frame_copy = np.ascontiguousarray(frame[:, :, ::-1], dtype=np.uint8)
            threading.Thread(
                target=self._compare_worker,
                args=(frame_copy,),
                daemon=True,
            ).start()

        # ── Return last known result while votes are being collected ──────────
        status = "Authorized" if self._last_auth_result else "Checking..."
        return self._last_auth_result, status

    # ── Background worker (daemon thread) ──────────────────────────────────────

    def _compare_worker(self, rgb: np.ndarray) -> None:
        """
        Runs in a daemon thread. Stores result in _pending_vote.
        rgb must already be contiguous uint8 (prepared by caller).
        """
        result = False
        try:
            import face_recognition
            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                distance = face_recognition.face_distance(
                    [self._enrolled_encoding], encodings[0]
                )[0]
                result = distance <= FACE_TOLERANCE
        except Exception as e:
            print(f"[FaceAuth] Comparison error: {e}")
        with self._lock:
            self._pending_vote = result

    @property
    def is_enrolled(self) -> bool:
        return self._enrolled
