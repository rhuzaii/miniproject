"""
face_auth.py
Multi-user face enrollment + batch-voting authentication.

Enrollment:
  Run enroll_face.py for each user. Each user's photo is saved as
  enrolled_faces/<Name>.jpg  (e.g. Rana.jpg, Anannya.jpg, Janet.jpg).
  The old single-user owner.jpg is still loaded as "Owner" for backward compat.

Authentication:
  All enrolled faces are loaded on startup. Each comparison finds the
  CLOSEST matching enrolled face below the distance threshold (τ).
  check() returns (authorized: bool, matched_name: str) where
  matched_name is the actual person's name (e.g. "Rana") when
  authorized, or "Unauthorized" / "Checking..." when not.

Design (M1 ARM64 performance):
  - face_recognition.compare_faces() takes ~100ms per call on M1.
  - Comparison runs in a daemon thread — main OpenCV loop never blocks.
  - We collect 5 votes (one per thread), majority (3/5) = authorized.
  - Result cached for 25 frames (~1 second at 15fps) before re-evaluating.
  - Only ONE comparison thread runs at a time (_vote_in_progress flag).
"""

import os
import threading
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ENROLLED_FACES_DIR = os.path.join(os.path.dirname(__file__), "enrolled_faces")
FACE_TOLERANCE = float(os.getenv("FACE_TOLERANCE", "0.65"))
FACE_AUTH_ENABLED = os.getenv("FACE_AUTH_ENABLED", "true").lower() == "true"

_FRAMES_PER_BATCH   = 10   # trigger a new comparison every Nth frame
_VOTES_NEEDED       = 5    # votes per evaluation batch
_MAJORITY_THRESHOLD = 3    # votes for a single name needed to be "authorized"
_CACHE_FRAMES       = 25   # frames to hold a batch result before re-evaluating


class FaceAuthenticator:
    """
    Manages multi-user face enrollment and non-blocking per-batch authorization.
    check() must be called from the main thread every frame.
    The heavy face_recognition comparison runs in a daemon thread.
    """

    def __init__(self):
        # {name: encoding}  e.g. {"Rana": array(...), "Anannya": array(...)}
        self._enrolled: dict[str, np.ndarray] = {}

        self._votes: list[str] = []       # each vote is a name ("") = no match
        self._last_auth_result: bool = False
        self._last_matched_name: str = ""
        self._cache_remaining: int = 0
        self._auth_enabled = FACE_AUTH_ENABLED

        # Thread-safety for async comparison
        self._lock = threading.Lock()
        self._vote_in_progress = False
        # None = thread not finished yet
        # ""   = thread finished, no match found
        # str  = thread finished, matched this name
        self._pending_vote: str | None = None

        self._load_all_enrollments()

    # ── Enrollment ─────────────────────────────────────────────────────────────

    def _load_all_enrollments(self) -> None:
        """Load every .jpg / .png in enrolled_faces/ as a named encoding."""
        if not self._auth_enabled:
            print("[FaceAuth] Auth disabled via FACE_AUTH_ENABLED=false")
            return
        if not os.path.isdir(ENROLLED_FACES_DIR):
            print(f"[FaceAuth] No enrolled_faces/ directory — run enroll_face.py.")
            return

        try:
            import face_recognition
        except ImportError:
            print("[FaceAuth] face_recognition not installed — auth disabled.")
            self._auth_enabled = False
            return

        print("[FaceAuth] Loading enrolled faces...")
        loaded = 0
        for filename in sorted(os.listdir(ENROLLED_FACES_DIR)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # File name (without extension) becomes the person's display name
            raw_name = os.path.splitext(filename)[0]
            name = raw_name.title()   # "rana" → "Rana", "owner" → "Owner"
            path = os.path.join(ENROLLED_FACES_DIR, filename)
            try:
                img = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(img)
                if not encodings:
                    print(f"  [FaceAuth] WARNING: no face found in {filename} — skipped.")
                    continue
                self._enrolled[name] = encodings[0]
                print(f"  [FaceAuth] Loaded: {name} ({filename})")
                loaded += 1
            except Exception as e:
                print(f"  [FaceAuth] Could not load {filename}: {e}")

        if loaded == 0:
            print("[FaceAuth] No valid enrolled faces found — run enroll_face.py.")
        else:
            print(f"[FaceAuth] {loaded} enrolled user(s): {', '.join(self._enrolled)}")

    def enroll(self, image_path: str, name: str) -> bool:
        """Enroll a face from image_path under the given name."""
        import face_recognition, shutil
        name = name.strip().title()
        if not name:
            print("[FaceAuth] Name cannot be empty.")
            return False
        os.makedirs(ENROLLED_FACES_DIR, exist_ok=True)
        try:
            img = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(img)
            if not encodings:
                print("[FaceAuth] No face found in image.")
                return False
            dest = os.path.join(ENROLLED_FACES_DIR, f"{name}.jpg")
            self._enrolled[name] = encodings[0]
            shutil.copy2(image_path, dest)
            print(f"[FaceAuth] Enrolled '{name}' saved to {dest}")
            return True
        except Exception as e:
            print(f"[FaceAuth] Enrollment failed: {e}")
            return False

    # ── Authorization Check (main thread, non-blocking) ────────────────────────

    def check(self, frame: np.ndarray, frame_counter: int) -> tuple[bool, str]:
        """
        Called every frame from main loop. Never blocks.
        Returns (authorized: bool, status_text: str).
        When authorized, status_text is the matched person's name (e.g. "Rana").
        When not, it is "Unauthorized" or "Checking...".
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
            # Count how many votes each name received (empty string = no match)
            name_counts: dict[str, int] = {}
            for v in self._votes:
                if v:   # non-empty = a real name matched
                    name_counts[v] = name_counts.get(v, 0) + 1

            # Pick the name with the most votes
            if name_counts:
                best_name = max(name_counts, key=name_counts.get)
                best_count = name_counts[best_name]
            else:
                best_name, best_count = "", 0

            authorized = best_count >= _MAJORITY_THRESHOLD
            self._last_auth_result = authorized
            self._last_matched_name = best_name if authorized else ""
            self._cache_remaining = _CACHE_FRAMES
            self._votes = []

        # ── Use cached result if still fresh ──────────────────────────────────
        if self._cache_remaining > 0:
            self._cache_remaining -= 1
            if self._last_auth_result:
                return True, self._last_matched_name
            return False, "Unauthorized"

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
        if self._last_auth_result:
            return True, self._last_matched_name
        return False, "Checking..."

    # ── Background worker (daemon thread) ──────────────────────────────────────

    def _compare_worker(self, rgb: np.ndarray) -> None:
        """
        Runs in a daemon thread. Finds the closest enrolled face and stores
        the matched name (or "" for no match) in _pending_vote.
        rgb must already be contiguous uint8 in RGB order (prepared by caller).
        """
        matched_name = ""
        try:
            import face_recognition
            query_encodings = face_recognition.face_encodings(rgb)
            if query_encodings:
                query_enc = query_encodings[0]
                best_name = ""
                best_dist = FACE_TOLERANCE   # only accept matches below threshold

                for name, enrolled_enc in self._enrolled.items():
                    dist = face_recognition.face_distance([enrolled_enc], query_enc)[0]
                    if dist < best_dist:
                        best_dist = dist
                        best_name = name

                matched_name = best_name   # "" if no one matched
        except Exception as e:
            print(f"[FaceAuth] Comparison error: {e}")

        with self._lock:
            self._pending_vote = matched_name

    # ── Utility ────────────────────────────────────────────────────────────────

    @property
    def enrolled_users(self) -> list[str]:
        """Returns names of all enrolled users."""
        return sorted(self._enrolled.keys())

    @property
    def is_enrolled(self) -> bool:
        return bool(self._enrolled)
