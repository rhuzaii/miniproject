"""
main.py — Entry point for the Gesture-Based Alexa Control System.

Run: python main.py
Controls:
  S — start gesture processing
  Q — quit

CRITICAL (M1 ARM64): ALL MediaPipe calls happen on THIS main thread.
Never offload hands.process() to any thread or subprocess. See face_auth.py and
gesture_recognition.py for batch-voting and stability patterns that keep everything
on the main thread while remaining performant.
"""

import cv2
import time
import requests
import os
import sys
import threading
import queue
sys.stdout.reconfigure(line_buffering=True)

# Disable CoreML GPU delegate on M1 — prevents 5-15min first-run compilation hang
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

from dotenv import load_dotenv

load_dotenv()

from gesture_recognition import GestureRecognizer, apply_clahe
from command_mapper import get_command, get_display_label, get_command_label
from face_auth import FaceAuthenticator
from emergency import EmergencySystem

# ── Config ─────────────────────────────────────────────────────────────────────
FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))
FLASK_URL = f"http://localhost:{FLASK_PORT}/trigger-command"
COMMAND_COOLDOWN = 2.0        # seconds before the SAME gesture can re-fire
DIFFERENT_GESTURE_COOLDOWN = 0.8  # seconds before a DIFFERENT gesture can fire

# ── UI Helpers ─────────────────────────────────────────────────────────────────

def _put_text(frame, text: str, y: int, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_progress_bar(frame, progress: float, y: int, color=(0, 165, 255)):
    w = frame.shape[1]
    bar_w = int((w - 20) * min(1.0, max(0.0, progress)))
    cv2.rectangle(frame, (10, y - 15), (10 + bar_w, y), color, -1)
    cv2.rectangle(frame, (10, y - 15), (w - 10, y), color, 1)


def _send_command(command: str) -> bool:
    """POST the command to the local Flask backend. Returns True on success."""
    try:
        resp = requests.post(FLASK_URL, json={"command": command}, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("status") == "success"
        print(f"[Main] Backend error: {resp.status_code} {resp.text}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"[Main] Flask not running at {FLASK_URL}. Start backend/app.py first.")
        return False
    except Exception as e:
        print(f"[Main] Request error: {e}")
        return False


# Thread-safe queue for async command results
_cmd_result_queue: queue.Queue = queue.Queue()


def _send_command_async(command: str) -> None:
    """
    Dispatch _send_command in a daemon thread so the OpenCV UI loop never blocks.
    Result arrives via _cmd_result_queue.
    We intentionally do NOT gate on an in-flight flag — that caused gesture
    transitions to block for the entire HTTP timeout (2s) before a new gesture
    could fire. The cooldown + stability buffer already prevent double-triggers.
    """
    def _worker():
        result = _send_command(command)
        _cmd_result_queue.put((command, result))

    threading.Thread(target=_worker, daemon=True).start()


# ── Main Loop ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Gesture-Based Alexa Control System")
    print("  MSRIT CSP67 — Team: Rana, Anannya, Janet, Jersha")
    print("=" * 55)
    print("Press S in the OpenCV window to start, Q to quit.")

    # Initialise modules ONCE in main thread
    print("[Main] Loading MediaPipe (first run may take 30-60s)...")
    recognizer = GestureRecognizer()
    print("[Main] MediaPipe ready.")
    print("[Main] Loading face auth...")
    face_auth = FaceAuthenticator()
    print("[Main] Face auth ready.")
    emergency = EmergencySystem()
    print("[Main] All modules loaded. Opening webcam...")

    # Webcam setup (M1 rules: cv2.VideoCapture(0) only, no backend flag)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[Main] ERROR: Cannot open webcam.")
        sys.exit(1)

    processing = False          # toggle with S key
    frame_counter = 0
    last_stable_gesture = None
    last_command_time = 0.0
    last_command_sent = ""      # last command string sent (e.g. "play_music")
    last_sent_gesture = ""      # last gesture NAME that fired (e.g. "THUMBS_UP")
    command_status = ""         # feedback shown on screen
    command_status_until = 0.0

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("[Main] Frame read failed. Check webcam.")
                break

            # Always flip for mirror effect
            raw_frame = cv2.flip(raw_frame, 1)
            frame = raw_frame.copy()

            if not processing:
                _put_text(frame, "Press S to start  |  Q to quit", 40, color=(200, 200, 200))
                cv2.imshow("Gesture Alexa Control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') or key == ord('S'):
                    processing = True
                    print("[Main] Started gesture processing.")
                elif key == ord('q') or key == ord('Q'):
                    break
                continue

            # ── Pre-processing ──────────────────────────────────────────────
            frame, low_light = apply_clahe(frame)

            # ── MediaPipe (main thread) ─────────────────────────────────────
            results = recognizer.process_frame(frame)
            frame = recognizer.draw_landmarks(frame, results)

            # Raw gesture for emergency (unstabilized)
            lm = recognizer.get_landmarks(results)
            from gesture_recognition import classify_gesture
            raw_gesture = classify_gesture(lm) if lm else None

            # ── Emergency (checked before auth gate) ───────────────────────
            emergency_fired = emergency.update(raw_gesture)
            em_text, em_progress = emergency.get_ui_state()

            # ── Stable Gesture ──────────────────────────────────────────────
            stable_gesture = recognizer.classify_and_stabilize(results)

            # ── Face Auth (main thread, batch-voted) ────────────────────────
            # Use raw_frame (pre-CLAHE) — enrollment was also captured raw.
            # CLAHE alters pixel values enough to shift face encodings and cause mismatches.
            authorized, auth_status = face_auth.check(raw_frame, frame_counter)

            # ── Command dispatch ────────────────────────────────────────────
            now = time.time()
            elapsed = now - last_command_time
            same_gesture = (stable_gesture == last_sent_gesture)

            # THUMBS_UP after a fist gesture (CLOSED_FIST / THUMBS_DOWN) is almost
            # always a release artifact — the hand naturally passes through thumbs-up
            # when uncurling. Apply the full 2s cooldown for that specific transition.
            _FIST_GESTURES = {"CLOSED_FIST", "THUMBS_DOWN"}
            release_artifact = (
                stable_gesture == "THUMBS_UP"
                and last_sent_gesture in _FIST_GESTURES
            )

            if same_gesture or release_artifact:
                required_cooldown = COMMAND_COOLDOWN          # 2.0s
            else:
                required_cooldown = DIFFERENT_GESTURE_COOLDOWN  # 0.8s

            cooldown_ok = elapsed >= required_cooldown
            if (
                stable_gesture
                and stable_gesture != "THREE_FINGERS"       # emergency handled separately
                and stable_gesture != last_stable_gesture   # only fire on gesture change
                and authorized
                and cooldown_ok
            ):
                command = get_command(stable_gesture)
                if command:
                    print(f"[Main] Gesture: {stable_gesture} → Command: {command}")
                    last_command_time = now
                    last_command_sent = command
                    last_sent_gesture = stable_gesture      # store gesture name, not command
                    command_status = "Sending..."
                    command_status_until = now + 6.0
                    _send_command_async(command)  # non-blocking — UI stays smooth

            # Poll for async command result
            try:
                cmd, success = _cmd_result_queue.get_nowait()
                command_status = (
                    f"Sent: {get_command_label(cmd)}" if success
                    else "API Error — check backend"
                )
                command_status_until = time.time() + 3.0
            except queue.Empty:
                pass

            last_stable_gesture = stable_gesture

            # ── UI Overlay ──────────────────────────────────────────────────
            h, w = frame.shape[:2]
            y = 35

            # Low light warning
            if low_light:
                _put_text(frame, "WARNING: Low Light", y, color=(0, 165, 255)); y += 30

            # Emergency overlay
            if em_text:
                color = (0, 0, 255) if "TRIGGERED" in em_text else (0, 165, 255)
                _put_text(frame, em_text, y, color=color, scale=0.75, thickness=2)
                if em_progress is not None:
                    _draw_progress_bar(frame, em_progress, y + 10)
                y += 40

            # Gesture label
            if stable_gesture:
                _put_text(frame, f"Gesture: {get_display_label(stable_gesture)}", y, color=(0, 255, 100))
            else:
                _put_text(frame, "Gesture: --", y, color=(150, 150, 150))
            y += 30

            # Auth status
            auth_color = (0, 255, 100) if authorized else (0, 0, 255)
            _put_text(frame, f"Auth: {auth_status}", y, color=auth_color)
            y += 30

            # Unauthorized blocker
            if not authorized and stable_gesture and stable_gesture != "THREE_FINGERS":
                _put_text(frame, "UNAUTHORIZED — Command Blocked", y, color=(0, 0, 255), scale=0.65)
                y += 28

            # Command status
            if now < command_status_until and command_status:
                color = (0, 255, 100) if "Sent" in command_status else (0, 0, 255)
                _put_text(frame, command_status, y, color=color)
                y += 28

            # Last command sent (persistent footer)
            if last_command_sent:
                _put_text(frame, f"Last: {get_command_label(last_command_sent)}", h - 20,
                          color=(180, 180, 180), scale=0.55, thickness=1)

            # FPS counter
            _put_text(frame, f"Frame: {frame_counter}", w - 120, color=(100, 100, 100), scale=0.5, thickness=1)

            cv2.imshow("Gesture Alexa Control", frame)
            frame_counter += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                processing = not processing
                print(f"[Main] Processing {'started' if processing else 'paused'}.")

    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[Main] Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()
