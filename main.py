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
OWNER_NAME = os.getenv("OWNER_NAME", "Owner")  # name logged to DynamoDB on auth success

# ── UI Helpers ─────────────────────────────────────────────────────────────────

def _put_text(frame, text: str, y: int, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_progress_bar(frame, progress: float, y: int, color=(0, 165, 255)):
    w = frame.shape[1]
    bar_w = int((w - 20) * min(1.0, max(0.0, progress)))
    cv2.rectangle(frame, (10, y - 15), (10 + bar_w, y), color, -1)
    cv2.rectangle(frame, (10, y - 15), (w - 10, y), color, 1)


def _send_command(command: str, user: str = "Unknown", auth_status: str = "Authorized") -> bool:
    """POST the command to the local Flask backend. Returns True on success."""
    try:
        payload = {
            "command":     command,
            "user":        user,
            "auth_status": auth_status,
        }
        resp = requests.post(FLASK_URL, json=payload, timeout=2)
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


def _send_command_async(command: str, user: str = "Unknown", auth_status: str = "Authorized") -> None:
    """
    Dispatch _send_command in a daemon thread so the OpenCV UI loop never blocks.
    Result arrives via _cmd_result_queue.
    We intentionally do NOT gate on an in-flight flag — that caused gesture
    transitions to block for the entire HTTP timeout (2s) before a new gesture
    could fire. The cooldown + stability buffer already prevent double-triggers.
    """
    def _worker():
        result = _send_command(command, user=user, auth_status=auth_status)
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
                    print(f"[Main] Gesture: {stable_gesture} → Command: {command} (user: {OWNER_NAME})")
                    last_command_time = now
                    last_command_sent = command
                    last_sent_gesture = stable_gesture      # store gesture name, not command
                    command_status = "Sending..."
                    command_status_until = now + 6.0
                    _send_command_async(command, user=OWNER_NAME, auth_status="Authorized")  # non-blocking

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


# ── Test Mode: Table 4 — Gesture Recognition Accuracy ─────────────────────────

def run_test_table4():
    """
    Table 4 — 6 gestures × 3 lighting conditions × 30 trials.
    Controls: SPACE = correct, X = wrong, Q = quit & save.
    """
    import csv

    GESTURES = ["THUMBS_UP", "THUMBS_DOWN", "OPEN_PALM", "CLOSED_FIST", "PEACE", "THREE_FINGERS"]
    LIGHTING  = ["Bright", "Normal", "Low-Light"]
    TRIALS    = 30
    DISPLAY   = {
        "THUMBS_UP": "Thumbs Up", "THUMBS_DOWN": "Thumbs Down",
        "OPEN_PALM": "Open Palm", "CLOSED_FIST": "Closed Fist",
        "PEACE": "Peace / V Sign", "THREE_FINGERS": "Three Fingers",
    }

    print("[Table4] Loading MediaPipe...")
    recognizer = GestureRecognizer()
    print("[Table4] Ready. Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    all_results = []
    quit_early  = False

    for lighting in LIGHTING:
        if quit_early:
            break
        print(f"\n[Table4] Lighting: {lighting} — adjust room then press SPACE in window.")

        # Wait screen
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            _put_text(frame, f"Lighting: {lighting}", 50, color=(0,255,255), scale=1.0, thickness=2)
            _put_text(frame, "Adjust lighting, then press SPACE to begin", 100)
            cv2.imshow("Table 4 — Gesture Accuracy", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            if key in (ord('q'), ord('Q')):
                quit_early = True
                break

        for gesture in GESTURES:
            if quit_early:
                break
            recognizer.buffer._buffer.clear()
            correct = incorrect = trial = 0
            print(f"  {lighting} / {DISPLAY[gesture]}")

            while trial < TRIALS:
                ret, raw = cap.read()
                if not ret:
                    break
                raw   = cv2.flip(raw, 1)
                frame, _ = apply_clahe(raw.copy())
                results  = recognizer.process_frame(frame)
                frame    = recognizer.draw_landmarks(frame, results)
                stable   = recognizer.classify_and_stabilize(results)

                h = frame.shape[0]
                y = 35
                _put_text(frame, f"Lighting: {lighting}", y, color=(0,200,255)); y += 35
                _put_text(frame, f"Do: {DISPLAY[gesture]}", y, color=(0,255,100), scale=0.9); y += 35
                _put_text(frame, f"Trial {trial+1}/{TRIALS}  Correct:{correct}  Wrong:{incorrect}", y); y += 35
                det = stable or "-- None --"
                col = (0,255,0) if stable == gesture else ((0,0,255) if stable else (150,150,150))
                _put_text(frame, f"Detected: {det}", y, color=col, scale=0.85); y += 35
                _put_text(frame, "SPACE=Correct   X=Wrong   Q=Quit", h-25,
                          color=(180,180,180), scale=0.55, thickness=1)

                cv2.imshow("Table 4 — Gesture Accuracy", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    correct += 1; trial += 1; recognizer.buffer._buffer.clear()
                elif key in (ord('x'), ord('X')):
                    incorrect += 1; trial += 1; recognizer.buffer._buffer.clear()
                elif key in (ord('q'), ord('Q')):
                    quit_early = True; break

            acc = round(correct / TRIALS * 100, 1)
            all_results.append({"Gesture": DISPLAY[gesture], "Lighting": lighting,
                                 "Correct": correct, "Incorrect": incorrect, "Accuracy (%)": acc})
            print(f"    {correct}/{TRIALS} = {acc}%")

    os.makedirs("tests/results", exist_ok=True)
    out = "tests/results/table4_gesture_accuracy.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Gesture","Lighting","Correct","Incorrect","Accuracy (%)"])
        w.writeheader(); w.writerows(all_results)
    print(f"\n[Table4] Saved → {out}")
    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()


# ── Test Mode: Table 6 — Face Authentication Accuracy ─────────────────────────

def run_test_table6():
    """
    Table 6 — 50+ attempts YOUR face, 50+ another person.
    Controls: Y = your face, O = other, TAB = switch, Q = quit & save.
    """
    import csv

    print("[Table6] Loading face_recognition...")
    auth = FaceAuthenticator()
    print("[Table6] Ready. Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    your_res = []; other_res = []; latencies = []
    frame_counter = 0
    phase = "YOUR"
    print("\nPhase YOUR → press Y each attempt. TAB to switch to OTHER. O to record. Q to quit.\n")

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        raw = cv2.flip(raw, 1)
        t0 = time.time()
        authorized, status = auth.check(raw, frame_counter)
        lat = (time.time() - t0) * 1000
        frame_counter += 1

        frame = raw.copy()
        h = frame.shape[0]
        y = 35
        pcol = (0,255,100) if phase == "YOUR" else (0,165,255)
        _put_text(frame, f"Phase: {phase} face", y, color=pcol, scale=0.9); y += 40
        _put_text(frame, f"Auth: {status}", y); y += 35

        tar = round(your_res.count(True)/len(your_res)*100,1) if your_res else "--"
        frr = round(your_res.count(False)/len(your_res)*100,1) if your_res else "--"
        far = round(other_res.count(True)/len(other_res)*100,1) if other_res else "--"
        _put_text(frame, f"YOUR:{len(your_res)}  TAR:{tar}%  FRR:{frr}%", y, scale=0.65, color=(200,255,200)); y += 28
        _put_text(frame, f"OTHER:{len(other_res)}  FAR:{far}%", y, scale=0.65, color=(255,200,150)); y += 28

        hint = "Y=record YOUR  TAB=switch  Q=quit" if phase=="YOUR" else "O=record OTHER  TAB=switch  Q=quit"
        _put_text(frame, hint, h-25, color=(180,180,180), scale=0.5, thickness=1)

        cv2.imshow("Table 6 — Face Auth Accuracy", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('y'), ord('Y')) and phase == "YOUR":
            your_res.append(authorized); latencies.append(lat)
            print(f"  YOUR #{len(your_res):>3}: {'ACCEPTED' if authorized else 'REJECTED'}  lat={lat:.0f}ms")
        elif key in (ord('o'), ord('O')) and phase == "OTHER":
            other_res.append(authorized)
            print(f"  OTHER #{len(other_res):>3}: {'ACCEPTED (FAR!)' if authorized else 'rejected'}")
        elif key == ord('\t'):
            phase = "OTHER" if phase == "YOUR" else "YOUR"
            print(f"\n── Switched to {phase} ──")
        elif key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

    TAR = round(your_res.count(True)/len(your_res)*100,1) if your_res else 0
    FRR = round(your_res.count(False)/len(your_res)*100,1) if your_res else 0
    FAR = round(other_res.count(True)/len(other_res)*100,1) if other_res else 0
    avg = round(sum(latencies)/len(latencies),1) if latencies else 0
    print(f"\n  TAR:{TAR}%  FRR:{FRR}%  FAR:{FAR}%  Avg latency:{avg}ms")

    os.makedirs("tests/results", exist_ok=True)
    out = "tests/results/table6_face_auth.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric","Value"])
        for row in [("TAR (%)",TAR),("FRR (%)",FRR),("FAR (%)",FAR),
                    ("Avg Latency (ms)",avg),("YOUR attempts",len(your_res)),("OTHER attempts",len(other_res))]:
            w.writerow(row)
    print(f"[Table6] Saved → {out}")


# ── Test Mode: Table 5 — Temporal Stability Buffer ─────────────────────────────

def run_test_table5():
    """
    Table 5 — buffer sizes N=1,5,10,15.
    Measures false trigger rate (%) and confirmation latency (ms).
    Controls: SPACE=start gesture, R=rest, Q=next buffer size.
    """
    import csv, math
    from gesture_recognition import GestureBuffer

    BUFFER_SIZES = [1, 5, 10, 15]
    TRIALS_PER   = 30

    print("[Table5] Loading MediaPipe...")
    all_results = []

    for buf_n in BUFFER_SIZES:
        print(f"\n[Table5] Buffer N={buf_n} — SPACE=gesture  R=rest  Q=next size")
        input(f"  Press Enter to open window for N={buf_n}...")

        recognizer = GestureRecognizer()
        recognizer.buffer = GestureBuffer(buf_n)
        recognizer.buffer.VOTE_THRESHOLD = max(1, math.ceil(buf_n * 5 / 6))

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        latencies      = []
        false_triggers = 0
        trial          = 0
        in_rest        = True
        gesture_start  = None
        last_stable    = None

        while trial < TRIALS_PER:
            ret, raw = cap.read()
            if not ret:
                break
            raw = cv2.flip(raw, 1)
            frame, _ = apply_clahe(raw.copy())
            results  = recognizer.process_frame(frame)
            frame    = recognizer.draw_landmarks(frame, results)
            stable   = recognizer.classify_and_stabilize(results)

            h = frame.shape[0]
            y = 35
            phase = "REST — hand down" if in_rest else "GESTURE — show one now!"
            pcol  = (120,120,120) if in_rest else (0,255,100)
            _put_text(frame, f"Buffer N={buf_n}   Trial {trial+1}/{TRIALS_PER}", y); y += 35
            _put_text(frame, f"Phase: {phase}", y, color=pcol, scale=0.75); y += 35
            _put_text(frame, f"Detected: {stable or '--'}", y, color=(0,200,255)); y += 35
            _put_text(frame, f"Done:{trial}  False triggers:{false_triggers}", y, color=(255,200,0), scale=0.65); y += 28
            _put_text(frame, "SPACE=start gesture  R=rest  Q=next buffer", h-25,
                      color=(180,180,180), scale=0.55, thickness=1)

            if in_rest and stable and stable != last_stable:
                false_triggers += 1

            if not in_rest and stable and gesture_start and stable != last_stable:
                ms = (time.perf_counter() - gesture_start) * 1000
                latencies.append(ms)
                trial += 1
                in_rest = True
                gesture_start = None
                recognizer.buffer._buffer.clear()
                print(f"    Trial {trial}: {ms:.0f}ms")

            last_stable = stable
            cv2.imshow(f"Table 5 — Buffer N={buf_n}", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                in_rest = False
                gesture_start = time.perf_counter()
                recognizer.buffer._buffer.clear()
            elif key in (ord('r'), ord('R')):
                in_rest = True
                gesture_start = None
                recognizer.buffer._buffer.clear()
            elif key in (ord('q'), ord('Q')):
                break

        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()

        avg  = round(sum(latencies)/len(latencies), 1) if latencies else 0
        rate = round(false_triggers / max(1, trial) * 100, 1)
        print(f"  N={buf_n}: false_rate={rate}%  avg_latency={avg}ms")
        all_results.append({"Buffer Size (N)": buf_n,
                             "False Trigger Rate (%)": rate,
                             "Avg Confirmation Latency (ms)": avg})

    os.makedirs("tests/results", exist_ok=True)
    out = "tests/results/table5_stability_buffer.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Buffer Size (N)","False Trigger Rate (%)","Avg Confirmation Latency (ms)"])
        w.writeheader(); w.writerows(all_results)
    print(f"[Table5] Saved → {out}")


# ── Test Mode: Table 7 — End-to-End Latency ───────────────────────────────────

def run_test_table7():
    """
    Table 7 — median + P95 latency for each pipeline stage over 50 trials.
    Requires backend/app.py to be running in another terminal.
    Controls: SPACE=timed trial (show gesture first), Q=quit & save.
    """
    import csv, statistics

    TARGET = 50
    print("[Table7] Loading MediaPipe + face_recognition...")
    recognizer = GestureRecognizer()
    face_auth  = FaceAuthenticator()
    print(f"[Table7] Ready. Make sure backend/app.py is running. Target: {TARGET} trials.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    t_cap_mp = []; t_mp_cls = []; t_cls_fa = []; t_fa_req = []; t_flask = []
    frame_counter = 0
    trials = 0

    def _pct(data, p):
        if not data: return 0
        s = sorted(data)
        k = (len(s)-1)*p/100
        lo, hi = int(k), min(int(k)+1, len(s)-1)
        return round(s[lo]+(k-lo)*(s[hi]-s[lo]), 1)

    while trials < TARGET:
        t0  = time.perf_counter()
        ret, raw = cap.read()
        if not ret: break
        raw = cv2.flip(raw, 1)
        frame, _ = apply_clahe(raw.copy())

        t1      = time.perf_counter()
        results = recognizer.process_frame(frame)
        frame   = recognizer.draw_landmarks(frame, results)

        t2     = time.perf_counter()
        stable = recognizer.classify_and_stabilize(results)

        t3             = time.perf_counter()
        authorized, auth_status = face_auth.check(raw, frame_counter)
        t4             = time.perf_counter()
        frame_counter += 1

        h = frame.shape[0]; y = 35
        _put_text(frame, f"Trial {trials+1}/{TARGET}", y, scale=0.8); y += 35
        _put_text(frame, f"Gesture: {stable or '--'}", y, color=(0,255,100)); y += 30
        _put_text(frame, f"Auth: {auth_status}", y, color=(0,200,255)); y += 30
        if t_flask:
            _put_text(frame, f"Last Flask RTT: {t_flask[-1]:.0f}ms", y, color=(255,200,0)); y += 28
        _put_text(frame, "SPACE=timed trial (need gesture+auth)   Q=quit & save",
                  h-25, color=(180,180,180), scale=0.5, thickness=1)

        cv2.imshow("Table 7 — Latency Benchmark", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and stable and authorized:
            try:
                command  = get_command(stable) or "play_music"
                t_req    = time.perf_counter()
                requests.post(FLASK_URL, json={"command": command}, timeout=15)
                t_done   = time.perf_counter()
                flask_ms = (t_done - t_req) * 1000

                t_cap_mp.append((t1-t0)*1000)
                t_mp_cls.append((t2-t1)*1000)
                t_cls_fa.append((t4-t3)*1000)
                t_fa_req.append((t_req-t4)*1000)
                t_flask.append(flask_ms)
                trials += 1
                print(f"  {trials:>2}: cap→mp={t_cap_mp[-1]:.1f}  mp→cls={t_mp_cls[-1]:.1f}  "
                      f"cls→fa={t_cls_fa[-1]:.1f}  flask={flask_ms:.1f}ms")
            except Exception as e:
                print(f"  Skipped — {e}")
        elif key in (ord('q'), ord('Q')):
            break

    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()

    if not t_flask:
        print("[Table7] No trials completed."); return

    stages = [
        ("Capture → MediaPipe",        t_cap_mp),
        ("MediaPipe → Classification", t_mp_cls),
        ("Classification → Face Auth", t_cls_fa),
        ("Auth → Flask POST",          t_fa_req),
        ("Flask → Response (total)",   t_flask),
    ]
    print(f"\n{'='*60}")
    print(f"  {'Stage':<35} {'Median':>8} {'P95':>8}")
    rows = []
    for name, data in stages:
        med = round(statistics.median(data), 1)
        p95 = _pct(data, 95)
        print(f"  {name:<35} {med:>7.1f}ms {p95:>7.1f}ms")
        rows.append({"Stage": name, "Median (ms)": med, "P95 (ms)": p95, "Trials": len(data)})
    print(f"{'='*60}")

    os.makedirs("tests/results", exist_ok=True)
    out = "tests/results/table7_latency.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Stage","Median (ms)","P95 (ms)","Trials"])
        w.writeheader(); w.writerows(rows)
    print(f"[Table7] Saved → {out}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["table4", "table5", "table6", "table7"], default=None,
                        help="Run a data-collection test instead of normal mode")
    args = parser.parse_args()

    if args.test == "table4":
        run_test_table4()
    elif args.test == "table5":
        run_test_table5()
    elif args.test == "table6":
        run_test_table6()
    elif args.test == "table7":
        run_test_table7()
    else:
        main()
