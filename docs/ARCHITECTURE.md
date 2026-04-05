# System Architecture

## Data Flow

```
User performs hand gesture
        ↓
Webcam captures frame (OpenCV, 1280×720, flipped)
        ↓
CLAHE preprocessing (LAB colorspace, L channel)
  └─ if brightness < 30 → show ⚠ Low Light warning
        ↓
MediaPipe extracts 21 hand landmarks  ← MAIN THREAD ONLY
        ↓
  ┌─────────────────────────────┐    ┌────────────────────────────────┐
  │ Face Auth (every 10th frame) │    │ Gesture Classify (every frame)  │
  │ 5-vote batch, 3/5 majority  │    │ Rule-based, 6 gestures         │
  │ Cache 25 frames             │    │ 10-frame stability buffer       │
  └─────────────────────────────┘    └────────────────────────────────┘
        ↓
THREE_FINGERS held ≥ 75 frames? ── YES ──→ Twilio SMS + Call → END
        ↓ NO
User authorized? ──────────────── NO ───→ Block + show warning → END
        ↓ YES
Gesture stable (10 same frames)?
        ↓
Command Mapper: gesture → command string
        ↓
Flask API: POST /trigger-command {command: "play_music"}   (port 5001)
        ↓
AWS API Gateway (REST, POST /command)
        ↓
AWS Lambda (lambda_function.py, ASK SDK)
        ↓
Alexa Skill intent handler
        ↓
Alexa Device (Echo Dot) executes command
        ↓
OpenCV window shows execution feedback
```

## Module Responsibilities

| File | Responsibility |
|---|---|
| `main.py` | Entry point. OpenCV loop, key handling, coordinates all modules |
| `gesture_recognition.py` | MediaPipe setup, CLAHE, landmark extraction, rule-based classifier |
| `command_mapper.py` | Dict mapping gesture name → command string |
| `face_auth.py` | Single-photo enrollment, batch voting logic, authorization check |
| `emergency.py` | Three-finger hold timer, Twilio SMS + call trigger |
| `backend/app.py` | Flask API server, runs on port 5001 |
| `backend/routes.py` | POST /trigger-command handler, AWS API Gateway forwarding |
| `alexa_skill/lambda_function.py` | AWS Lambda handler, all 6 Alexa intent handlers |

## Critical M1 ARM64 Rules

1. **MediaPipe must run on main thread** — CoreML/Metal backend crashes on any other thread
2. **Library versions are locked** — do not upgrade mediapipe (0.10.9) or numpy (1.26.4)
3. **Python 3.11 required** — dlib (face_recognition dependency) does not build on 3.12+
4. **Webcam**: `cv2.VideoCapture(0)` only — no CAP_AVFOUNDATION flag
5. **Port 5001** — macOS Monterey+ uses 5000 for AirPlay

## Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Computer Vision | OpenCV | 4.9.0.80 |
| Hand Tracking | MediaPipe | 0.10.9 |
| Numerical | NumPy | 1.26.4 |
| Face Auth | face-recognition (dlib) | 1.3.0 |
| Emergency | Twilio | 8.13.0 |
| Backend | Flask + flask-cors | 3.0.2 / 4.0.0 |
| Cloud | AWS API Gateway + Lambda | us-east-1 |
| Alexa | ASK SDK (ask-sdk-core) | 2.x |
| Config | python-dotenv | 1.0.1 |
| Language | Python | 3.11.x |
