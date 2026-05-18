"""
enroll_face.py — Multi-user face enrollment.

Run for EACH person who should be able to control the system:
  python enroll_face.py

You will be prompted for the person's name. Their face is saved as:
  enrolled_faces/<Name>.jpg   (e.g. enrolled_faces/Rana.jpg)

Multiple people can be enrolled — run the script once per person.
To list enrolled users:   python enroll_face.py --list
To delete a user:         python enroll_face.py --delete <Name>

Controls (OpenCV window):
  E — capture current frame and save
  Q — quit without saving
"""

import cv2
import os
import sys
import argparse

ENROLLED_DIR = os.path.join(os.path.dirname(__file__), "enrolled_faces")


def list_enrolled():
    if not os.path.isdir(ENROLLED_DIR):
        print("No enrolled_faces/ directory. No users enrolled yet.")
        return
    faces = [f for f in sorted(os.listdir(ENROLLED_DIR))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not faces:
        print("No enrolled users yet.")
    else:
        print(f"\nEnrolled users ({len(faces)}):")
        for f in faces:
            name = os.path.splitext(f)[0].title()
            path = os.path.join(ENROLLED_DIR, f)
            size = os.path.getsize(path)
            print(f"  • {name:<20} ({f}, {size//1024} KB)")
    print()


def delete_user(name: str):
    name = name.strip().title()
    path = os.path.join(ENROLLED_DIR, f"{name}.jpg")
    if not os.path.exists(path):
        # Try other extensions
        for ext in (".jpeg", ".png"):
            alt = os.path.join(ENROLLED_DIR, f"{name}{ext}")
            if os.path.exists(alt):
                path = alt
                break
        else:
            print(f"No enrollment found for '{name}'.")
            return
    os.remove(path)
    print(f"Deleted enrollment for '{name}'.")


def enroll_user():
    os.makedirs(ENROLLED_DIR, exist_ok=True)

    # ── Ask for name ──────────────────────────────────────────────────────────
    print("=" * 55)
    print("  Face Enrollment — Multi-User Gesture Alexa System")
    print("=" * 55)
    list_enrolled()

    name = ""
    while not name:
        raw = input("Enter the person's name to enroll (e.g. Rana): ").strip()
        name = raw.title()   # "rana" → "Rana"
        if not name:
            print("Name cannot be empty.")

    save_path = os.path.join(ENROLLED_DIR, f"{name}.jpg")
    overwrite_warning = os.path.exists(save_path)

    print(f"\nEnrolling: {name}")
    if overwrite_warning:
        print(f"  WARNING: {name} is already enrolled. Press E to overwrite.")
    print("  Stand in front of the camera. Press E to capture. Q to cancel.\n")

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    saved = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # UI overlay
            cv2.putText(display, f"Enrolling: {name}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(display, "Press E to capture  |  Q to cancel",
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
            if overwrite_warning:
                cv2.putText(display, f"Will overwrite existing {name} enrollment",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)

            # Show enrolled users count at the bottom
            n_enrolled = len([f for f in os.listdir(ENROLLED_DIR)
                               if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            cv2.putText(display, f"Total enrolled: {n_enrolled} user(s)",
                        (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imshow("Enroll Face", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('e'), ord('E')):
                try:
                    import face_recognition
                    rgb = frame[:, :, ::-1]
                    locations = face_recognition.face_locations(rgb, model="hog")
                    if not locations:
                        print("  No face detected — move closer and try again.")
                        err_display = display.copy()
                        cv2.putText(err_display, "No face detected — try again!",
                                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        cv2.imshow("Enroll Face", err_display)
                        cv2.waitKey(1500)
                        continue

                    cv2.imwrite(save_path, frame)
                    print(f"\n  Enrolled '{name}' saved to {save_path}")
                    print("  This user can now control the system.")
                    saved = True

                    ok_display = display.copy()
                    cv2.putText(ok_display, f"{name} enrolled!",
                                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 100), 2)
                    cv2.imshow("Enroll Face", ok_display)
                    cv2.waitKey(2000)
                    break

                except ImportError:
                    print("  face_recognition not installed. Saving image anyway.")
                    cv2.imwrite(save_path, frame)
                    saved = True
                    break

            elif key in (ord('q'), ord('Q')):
                print(f"  Cancelled. {name} was NOT enrolled.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    list_enrolled()

    if not saved:
        print("No face was enrolled. Run enroll_face.py again when ready.")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-user face enrollment for Gesture Alexa System"
    )
    parser.add_argument("--list",   action="store_true", help="List enrolled users")
    parser.add_argument("--delete", metavar="NAME",      help="Delete a user's enrollment")
    args = parser.parse_args()

    if args.list:
        list_enrolled()
    elif args.delete:
        delete_user(args.delete)
    else:
        enroll_user()


if __name__ == "__main__":
    main()
