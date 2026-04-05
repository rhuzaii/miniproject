"""
face_auth.py
Phase 3 — Single-photo face enrollment + batch-voting authentication.

Design decisions (performance on M1 ARM64):
  - face_recognition.compare_faces() takes ~100ms per call.
  - We run it only every 10th frame (frame_counter % 10 == 0).
  - We collect 5 votes, majority (3/5) = authorized.
  - Result is cached for 25 frames (~1 second at 25fps).
  - No threads — everything runs in the main OpenCV loop.

Known limitations (documented as required by spec):
  - Glasses vs no-glasses: encoding changes significantly; re-enroll if accuracy drops.
  - Drastic lighting changes can lower match confidence.
  - Single enrolled face only (one owner.jpg).
"""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ENROLLED_FACES_DIR = os.path.join(os.path.dirname(__file__), "enrolled_faces")
ENROLLMENT_PATH = os.path.join(ENROLLED_FACES_DIR, "owner.jpg")
FACE_TOLERANCE = float(os.getenv("FACE_TOLERANCE", "0.55"))
FACE_AUTH_ENABLED = os.getenv("FACE_AUTH_ENABLED", "true").lower() == "true"

# Batch voting parameters (from spec)
_FRAMES_PER_BATCH = 10      # run comparison every Nth frame
_VOTES_NEEDED = 5           # collect this many votes per batch
_MAJORITY_THRESHOLD = 3     # votes needed for "authorized" (3/5)
_CACHE_FRAMES = 25          # frames to reuse a batch result


class FaceAuthenticator:
    """
    Manages face enrollment and per-batch authorization checking.
    All public methods must be called from the main thread.
    """

    def __init__(self):
        self._enrolled_encoding: np.ndarray | None = None
        self._votes: list[bool] = []
        self._last_auth_result: bool = False
        self._cache_remaining: int = 0   # frames left before next batch
        self._enrolled = False
        self._auth_enabled = FACE_AUTH_ENABLED

        self._load_enrollment()

    # ── Enrollment ─────────────────────────────────────────────────────────────

    def _load_enrollment(self) -> None:
        """Try to load the enrolled face encoding from disk at startup."""
        if not self._auth_enabled:
            print("[FaceAuth] Auth disabled via FACE_AUTH_ENABLED=false")
            return

        if not os.path.exists(ENROLLMENT_PATH):
            print(f"[FaceAuth] No enrolled face found at {ENROLLMENT_PATH}.")
            print("[FaceAuth] Run enroll_face.py to register the owner.")
            print("[FaceAuth] Allowing all commands until enrollment is done.")
            return

        try:
            print("[FaceAuth] Loading face_recognition (may take a few seconds)...")
            import face_recognition
            img = face_recognition.load_image_file(ENROLLMENT_PATH)
            encodings = face_recognition.face_encodings(img)
            if not encodings:
                print("[FaceAuth] WARNING: No face detected in owner.jpg. Re-enroll.")
                return
            self._enrolled_encoding = encodings[0]
            self._enrolled = True
            print("[FaceAuth] Enrolled face loaded successfully.")
        except Exception as e:
            print(f"[FaceAuth] Failed to load enrolled face: {e}")

    def enroll(self, image_path: str) -> bool:
        """
        Enroll a new face from image_path. Saves encoding and copies image to enrolled_faces/.
        Returns True on success.
        """
        try:
            import face_recognition
            import shutil
            os.makedirs(ENROLLED_FACES_DIR, exist_ok=True)
            img = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(img)
            if not encodings:
                print("[FaceAuth] No face found in the provided image.")
                return False
            self._enrolled_encoding = encodings[0]
            self._enrolled = True
            shutil.copy2(image_path, ENROLLMENT_PATH)
            print(f"[FaceAuth] Enrolled face saved to {ENROLLMENT_PATH}")
            return True
        except Exception as e:
            print(f"[FaceAuth] Enrollment failed: {e}")
            return False

    # ── Authorization Check ─────────────────────────────────────────────────────

    def check(self, frame: np.ndarray, frame_counter: int) -> tuple[bool, str]:
        """
        Called every frame from the main loop.
        Returns (authorized: bool, status_text: str).

        Batch voting logic:
          - Every _FRAMES_PER_BATCH frames: run one comparison vote.
          - After _VOTES_NEEDED votes: compute majority → cache for _CACHE_FRAMES.
        """
        if not self._auth_enabled:
            return True, "Auth Disabled"

        if not self._enrolled:
            return True, "Not Enrolled (open)"

        # Use cached result if still valid
        if self._cache_remaining > 0:
            self._cache_remaining -= 1
            status = "Authorized" if self._last_auth_result else "Unauthorized"
            return self._last_auth_result, status

        # Collect a vote every _FRAMES_PER_BATCH frames
        if frame_counter % _FRAMES_PER_BATCH == 0:
            vote = self._run_comparison(frame)
            self._votes.append(vote)

        # Once we have enough votes, decide and cache
        if len(self._votes) >= _VOTES_NEEDED:
            authorized = self._votes.count(True) >= _MAJORITY_THRESHOLD
            self._last_auth_result = authorized
            self._cache_remaining = _CACHE_FRAMES
            self._votes = []
            status = "Authorized" if authorized else "Unauthorized"
            return authorized, status

        # Still collecting votes — use last known result
        status = "Authorized" if self._last_auth_result else "Checking..."
        return self._last_auth_result, status

    def _run_comparison(self, frame: np.ndarray) -> bool:
        """Run face_recognition.compare_faces() on a single frame. Returns True if match."""
        try:
            import face_recognition
            rgb = frame[:, :, ::-1]  # BGR → RGB
            face_locations = face_recognition.face_locations(rgb, model="hog")
            if not face_locations:
                return False
            encodings = face_recognition.face_encodings(rgb, face_locations)
            if not encodings:
                return False
            results = face_recognition.compare_faces(
                [self._enrolled_encoding], encodings[0], tolerance=FACE_TOLERANCE
            )
            return results[0]
        except Exception as e:
            print(f"[FaceAuth] Comparison error: {e}")
            return False

    @property
    def is_enrolled(self) -> bool:
        return self._enrolled
