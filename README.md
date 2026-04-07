# Gesture-Based Alexa Control System

Control an Amazon Echo Dot using hand gestures — no voice commands needed.

**Team:** Rana Adiba, Anannya, Janet, Jersha
**Institution:** MSRIT — CSP67 Mini Project

---

## What It Does

Show a hand gesture to the webcam → Echo Dot performs the action automatically.

| Gesture | Action |
|---------|--------|
| Thumbs Up | Play music |
| Thumbs Down | Stop music |
| Open Palm | Turn lights on |
| Closed Fist | Turn lights off |
| Peace Sign | Get weather report |
| Three Fingers (hold 3s) | Emergency alert (SMS + call) |

---

## How It Works

The system has two complementary paths:

```
GESTURE PATH (Control):
Webcam → MediaPipe → Gesture Classifier → Face Auth
       → Flask API → Voice Monkey → Alexa Routine → Echo Dot

VOICE PATH (Monitor):
"Alexa, ask gesture control..." → Lambda → DynamoDB → Alexa Skill response
```

- **MediaPipe** detects 21 hand landmarks in real time
- **Rule-based classifier** identifies 6 gestures from landmark geometry
- **Face authentication** ensures only the enrolled owner triggers commands
- **Flask REST API** receives commands and dispatches to both paths
- **Voice Monkey** proactively triggers Alexa Routines on Echo Dot — no wake word needed
- **AWS Lambda + DynamoDB** logs every gesture command with timestamp for voice querying
- **Emergency system** triggers Twilio SMS + call when three-finger gesture is held for 3 seconds

---

## Dual Architecture

| Path | Purpose | How |
|------|---------|-----|
| Gesture → Voice Monkey → Echo Dot | **Control** — perform actions silently | Show a gesture |
| Voice → Lambda → DynamoDB → Alexa | **Monitor** — query gesture history | Ask Alexa |

**Voice query examples:**
- *"Alexa, ask gesture control what was the last gesture"*
  → *"The last gesture was Thumbs Up at 6:15 PM. It triggered the command to play music."*
- *"Alexa, ask gesture control how many gestures today"*
  → *"3 gestures detected today. Most recent was Peace Sign at 6:32 PM."*

---

## Project Structure

```
miniproj/
├── main.py                    # Entry point — run this
├── gesture_recognition.py     # MediaPipe + CLAHE + gesture classifier
├── command_mapper.py          # Gesture → command mapping
├── face_auth.py               # Face enrollment + batch-voting authentication
├── emergency.py               # Emergency trigger + Twilio
├── enroll_face.py             # One-time face registration
├── backend/
│   ├── app.py                 # Flask server (port 5001)
│   └── routes.py              # POST /trigger-command + dual delivery
├── alexa_skill/
│   ├── lambda_function.py     # AWS Lambda — DynamoDB logging + Alexa intents
│   └── interaction_model.json # Alexa skill intents (8 total)
├── docs/
│   ├── SETUP.md               # Full setup guide
│   ├── PROJECT_DOCUMENTATION.md # Technical documentation
│   └── architecture.html      # Visual architecture diagram
├── .env.template              # Copy to .env and fill credentials
└── requirements.txt           # Python dependencies
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/rhuzaii/miniproj.git
cd miniproj

# Requires Python 3.11
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate

pip install numpy==1.26.4
brew install cmake
pip install dlib==19.24.2
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.template .env
# Edit .env with your credentials (Twilio, Voice Monkey, AWS)
```

### 3. Enroll your face

```bash
python enroll_face.py
# Press E to capture, Q to quit
```

### 4. Run

```bash
# Terminal 1 — backend
python backend/app.py

# Terminal 2 — gesture system
python main.py
```

Press **S** in the OpenCV window to start. Press **Q** to quit.

---

## AWS Setup Required

| Service | Purpose |
|---------|---------|
| API Gateway | HTTP endpoint that triggers Lambda |
| Lambda | Logs gesture commands to DynamoDB + handles Alexa Skill |
| DynamoDB | Table `gesture_commands` — stores gesture history |

Lambda requires **AmazonDynamoDBFullAccess** IAM policy attached to its execution role.

See [`docs/SETUP.md`](docs/SETUP.md) for full AWS setup steps.

---

## Requirements

- macOS with Apple Silicon (M1/M2) — Python 3.11 required
- Amazon Echo Dot (any generation)
- Amazon Developer account (same as Echo Dot)
- Voice Monkey account — voicemonkey.io
- AWS account (API Gateway + Lambda + DynamoDB)
- Twilio account (for emergency alerts)
- Webcam

---

## Documentation

- Full setup instructions → [`docs/SETUP.md`](docs/SETUP.md)
- Technical documentation → [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md)
- Architecture diagram → [`docs/architecture.html`](docs/architecture.html)

---

## Tech Stack

Python 3.11 · OpenCV · MediaPipe · face_recognition · Flask · AWS Lambda · DynamoDB · Alexa Skills Kit · Voice Monkey · Twilio
