"""
enroll_face.py — Face enrollment helper script.

Run once before using the system:
  python enroll_face.py

Controls (OpenCV window):
  E — capture current frame and save as enrolled face
  Q — quit without saving

The captured photo is saved to enrolled_faces/owner.jpg and used by face_auth.py.
Only ONE face should be in the frame when you press E.
Tip: ensure good lighting and face the camera straight-on without glasses
if you normally do not wear glasses (or enroll with them if you do).
"""

import cv2
import os
import sys

ENROLLED_DIR = os.path.join(os.path.dirname(__file__), "enrolled_faces")
ENROLLMENT_PATH = os.path.join(ENROLLED_DIR, "owner.jpg")


def main():
    os.makedirs(ENROLLED_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    print("=" * 50)
    print("  Face Enrollment")
    print("  Press E to capture | Q to quit")
    print("=" * 50)

    saved = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            cv2.putText(display, "Press E to enroll your face  |  Q to quit",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2, cv2.LINE_AA)

            if os.path.exists(ENROLLMENT_PATH):
                cv2.putText(display, "Existing enrollment found — E will overwrite",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)

            cv2.imshow("Enroll Face", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('e') or key == ord('E'):
                # Verify face is detectable before saving
                try:
                    import face_recognition
                    rgb = frame[:, :, ::-1]
                    locations = face_recognition.face_locations(rgb, model="hog")
                    if not locations:
                        print("No face detected in frame. Move closer and try again.")
                        cv2.putText(display, "No face detected — try again",
                                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Enroll Face", display)
                        cv2.waitKey(1500)
                        continue

                    cv2.imwrite(ENROLLMENT_PATH, frame)
                    print(f"Enrolled face saved to {ENROLLMENT_PATH}")
                    print("You can now run main.py — face auth is active.")
                    saved = True
                    cv2.putText(display, "Enrolled! Starting in 2s...",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 100), 2)
                    cv2.imshow("Enroll Face", display)
                    cv2.waitKey(2000)
                    break

                except ImportError:
                    print("face_recognition not installed. Saving image anyway.")
                    cv2.imwrite(ENROLLMENT_PATH, frame)
                    print(f"Image saved to {ENROLLMENT_PATH}")
                    saved = True
                    break

            elif key == ord('q') or key == ord('Q'):
                print("Enrollment cancelled.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not saved:
        print("No face was enrolled. Run enroll_face.py again when ready.")


if __name__ == "__main__":
    main()
