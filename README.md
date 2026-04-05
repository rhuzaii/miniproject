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

```
Webcam → MediaPipe (hand detection) → Gesture Classifier
       → Face Authentication → Flask API → Voice Monkey
       → Alexa Routine → Echo Dot performs action
```

- **MediaPipe** detects 21 hand landmarks in real time
- **Rule-based classifier** identifies 6 gestures from landmark geometry
- **Face authentication** ensures only the enrolled owner triggers commands
- **Flask REST API** receives commands and forwards to Echo Dot via Voice Monkey
- **AWS Lambda** logs every command to CloudWatch as a secondary path
- **Emergency system** triggers Twilio SMS + call when three-finger gesture is held for 3 seconds

---

## Project Structure

```
miniproj/
├── main.py                    # Entry point — run this
├── gesture_recognition.py     # MediaPipe + gesture classifier
├── command_mapper.py          # Gesture → command mapping
├── face_auth.py               # Face enrollment + authentication
├── emergency.py               # Emergency trigger + Twilio
├── enroll_face.py             # One-time face registration
├── backend/
│   ├── app.py                 # Flask server
│   └── routes.py              # POST /trigger-command
├── alexa_skill/
│   ├── lambda_function.py     # AWS Lambda handler
│   └── interaction_model.json # Alexa skill intents
├── docs/
│   ├── SETUP.md               # Full setup guide
│   └── PROJECT_DOCUMENTATION.md # Technical documentation
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
# Terminal 1
python backend/app.py

# Terminal 2
python main.py
```

Press **S** in the OpenCV window to start. Press **Q** to quit.

---

## Requirements

- macOS with Apple Silicon (M1/M2) — Python 3.11 required
- Amazon Echo Dot (any generation)
- Amazon Developer account (same as Echo Dot)
- Voice Monkey account — voicemonkey.io
- AWS account (API Gateway + Lambda)
- Twilio account (for emergency alerts)
- Webcam

---

## Documentation

- Full setup instructions → [`docs/SETUP.md`](docs/SETUP.md)
- Technical documentation → [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md)

---

## Tech Stack

Python 3.11 · OpenCV · MediaPipe · face_recognition · Flask · AWS Lambda · Alexa Skills Kit · Voice Monkey · Twilio
