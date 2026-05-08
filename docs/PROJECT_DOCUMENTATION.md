# Gesture-Based Alexa Control System
### Technical Documentation

**Project:** CSP67 Mini Project
**Team:** Rana Adiba, Anannya, Janet, Jersha
**Institution:** MSRIT (M.S. Ramaiah Institute of Technology)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Module Breakdown](#4-module-breakdown)
5. [Gesture Recognition Pipeline](#5-gesture-recognition-pipeline)
6. [Face Authentication](#6-face-authentication)
7. [Emergency System](#7-emergency-system)
8. [Backend API](#8-backend-api)
9. [Alexa Integration](#9-alexa-integration)
10. [DynamoDB Gesture Logging](#10-dynamodb-gesture-logging)
11. [Data Flow — End to End](#11-data-flow--end-to-end)
12. [Gesture-to-Command Mapping](#12-gesture-to-command-mapping)
13. [Environment Configuration](#13-environment-configuration)
14. [Setup & Running](#14-setup--running)
15. [Design Decisions & Justifications](#15-design-decisions--justifications)
16. [Known Limitations](#16-known-limitations)

---

## 1. Project Overview

The **Gesture-Based Alexa Control System** enables users to control an Amazon Echo Dot using hand gestures captured from a standard webcam — **no voice commands required**. The system detects gestures in real time, authenticates the user via face recognition, and delivers commands to the Echo Dot which performs the corresponding action (playing music, controlling lights, fetching weather, etc.).

### Problem Statement
Traditional Alexa interaction requires the user to speak a wake word and voice command. This is unsuitable for:
- Noisy environments
- Users with speech impairments
- Situations requiring silent control (libraries, meetings, sleeping households)

### Solution
A computer vision pipeline that recognises 6 distinct hand gestures and maps them to Alexa device actions, secured by face authentication to prevent unauthorised use.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER'S MACHINE                           │
│                                                                 │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Webcam  │───▶│  main.py         │───▶│  Flask Backend   │  │
│  └──────────┘    │  (OpenCV loop)   │    │  port 5001       │  │
│                  │                  │    │  /trigger-command│  │
│                  │ ┌──────────────┐ │    └────────┬─────────┘  │
│                  │ │ MediaPipe    │ │             │            │
│                  │ │ Hands        │ │             │            │
│                  │ └──────────────┘ │             │            │
│                  │ ┌──────────────┐ │             │            │
│                  │ │ Face Auth    │ │             │            │
│                  │ └──────────────┘ │             │            │
│                  │ ┌──────────────┐ │             │            │
│                  │ │ Emergency    │ │             │            │
│                  │ └──────────────┘ │             │            │
│                  └──────────────────┘             │            │
└──────────────────────────────────────────────────┼────────────┘
                                                   │
                          ┌────────────────────────┤
                          │                        │
                          ▼                        ▼
               ┌──────────────────┐    ┌──────────────────────┐
               │  Voice Monkey    │    │  AWS API Gateway     │
               │  API             │    │  → Lambda Function   │
               │  (Primary Path)  │    │  (Secondary Path /   │
               └────────┬─────────┘    │   CloudWatch Logs)   │
                        │              └──────────────────────┘
                        ▼
               ┌──────────────────┐
               │  Alexa Routine   │
               │  (Routine        │
               │   Trigger)       │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  Amazon Echo Dot │
               │  (Performs       │
               │   Action)        │
               └──────────────────┘
```

### Delivery Paths

**Path 1 — Voice Monkey (Primary)**
```
Flask → Voice Monkey API → Routine Trigger Device → Alexa Routine → Echo Dot Action
```
Voice Monkey is a certified Alexa skill that exposes an HTTP API to trigger Alexa Routines programmatically. This is the primary delivery path that makes the Echo Dot perform actions.

**Path 2 — AWS Lambda (Secondary)**
```
Flask → AWS API Gateway → Lambda Function → CloudWatch Logs
```
The Lambda function logs every command to CloudWatch for audit purposes and provides a fallback confirmation layer.

---

## 3. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Vision | OpenCV 4.9 | Webcam capture, frame processing, UI overlay |
| Hand Detection | MediaPipe 0.10.9 | 21-point hand landmark detection |
| Image Enhancement | CLAHE (OpenCV) | Low-light preprocessing |
| Face Recognition | face_recognition 1.3.0 (dlib) | Owner authentication |
| Backend API | Flask 3.0.2 | REST API server |
| HTTP | Requests 2.31.0 | Outbound API calls |
| Cloud | AWS API Gateway + Lambda | Serverless command relay + gesture logging |
| Database | AWS DynamoDB | Gesture command history — queryable by voice |
| Alexa SDK | ASK SDK Core | Custom Alexa Skill |
| Voice Delivery | Voice Monkey API | Proactive Echo Dot control |
| Emergency | Twilio | SMS + phone call alerts |
| Config | python-dotenv | Environment variable management |
| Language | Python 3.11 | Core runtime |

---

## 4. Module Breakdown

```
miniproj/
├── main.py                  ← Entry point, OpenCV main loop
├── gesture_recognition.py   ← MediaPipe + CLAHE + gesture classifier
├── command_mapper.py        ← Gesture → command string mapping
├── face_auth.py             ← Face enrollment + batch-voting auth
├── emergency.py             ← Hold-gesture emergency trigger + Twilio
├── enroll_face.py           ← One-time face enrollment script
├── backend/
│   ├── app.py               ← Flask app factory + CORS
│   └── routes.py            ← POST /trigger-command handler
├── alexa_skill/
│   ├── lambda_function.py   ← AWS Lambda handler (ASK SDK)
│   └── interaction_model.json ← Alexa skill intent definitions
├── .env                     ← Credentials (not committed to git)
└── requirements.txt         ← Python dependencies
```

---

## 5. Gesture Recognition Pipeline

### 5.1 CLAHE Preprocessing

Before landmark detection, each frame is enhanced using **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**:

```
BGR Frame → LAB Colorspace → Extract L channel → CLAHE → Merge → BGR
```

- Converts to LAB colorspace (separates luminance from colour)
- Applies CLAHE only to the L (luminance) channel
- Preserves colour accuracy while boosting contrast
- Detects low-light condition if mean brightness < threshold (default: 30)

### 5.2 MediaPipe Hand Landmark Detection

MediaPipe Hands detects **21 landmarks** on the hand:

```
        8           12   16   20
        |            |    |    |
        7            11   15   19
        |            |    |    |
        6   4        10   14   18
        |   |         |    |    |
        5   3         9   13   17
         \  |        /    /    /
          \ 2       /    /    /
           \|      /    /    /
            1-----/----/----/
            |
            0 (Wrist)
```

Key landmarks used:
- `0` — Wrist
- `2` — Thumb MCP (base knuckle)
- `4` — Thumb tip
- `5,9,13,17` — Index/Middle/Ring/Pinky MCP
- `6,10,14,18` — Index/Middle/Ring/Pinky PIP (middle joint)
- `8,12,16,20` — Index/Middle/Ring/Pinky fingertips

**Configuration:** `model_complexity=0` (lite model) for performance, `min_detection_confidence=0.7`, single hand tracking.

**Critical (M1 ARM64):** All `hands.process()` calls must run on the **main thread**. MediaPipe uses CoreML on Apple Silicon — offloading to threads causes crashes.

### 5.3 Rule-Based Gesture Classifier

Each gesture is classified using geometric rules on landmark coordinates (y increases downward in normalised coordinates):

```python
is_finger_extended(tip, pip) → tip.y < pip.y
```

| Gesture | Rule |
|---------|------|
| THUMBS_UP | `lm[4].y < lm[5].y` AND all 4 fingers curled |
| THUMBS_DOWN | `lm[4].y > lm[0].y` AND all 4 fingers curled |
| OPEN_PALM | All 4 fingers extended |
| CLOSED_FIST | All fingertips below MCP joints AND `lm[4].y ≤ lm[0].y` |
| PEACE | Index + Middle extended, Ring + Pinky curled |
| THREE_FINGERS | Index + Middle + Ring extended, Pinky curled |

**Classifier priority order** (CLOSED_FIST checked with explicit MCP comparison to avoid conflicts with THUMBS_DOWN):
1. THUMBS_UP → THUMBS_DOWN → OPEN_PALM → THREE_FINGERS → PEACE → CLOSED_FIST

### 5.4 Stability Buffer (Anti-Jitter)

Raw gesture output is fed through a `GestureBuffer`:

```
Frame 1:  THUMBS_UP
Frame 2:  THUMBS_UP
...
Frame 10: THUMBS_UP  ← All 10 identical → STABLE → trigger command
```

- **Buffer size:** 10 frames (configurable via `GESTURE_STABILITY_FRAMES`)
- **Rule:** All 10 frames must show the same gesture
- **Effect:** Eliminates jitter and accidental triggers from transitional hand positions
- **Command cooldown:** 3 seconds between consecutive command sends

---

## 6. Face Authentication

Ensures only the enrolled owner can trigger commands.

### 6.1 Enrollment
Run `enroll_face.py` once to register the owner:
- Opens webcam → user presses **E** to capture
- `face_recognition.face_encodings()` extracts a 128-dimensional face encoding
- Encoding saved as `enrolled_faces/owner.jpg`

### 6.2 Batch Voting Authentication

Called every frame from the main loop — designed for performance:

```
Every 10th frame:
  → Run face_recognition.compare_faces()
  → Append True/False to votes list

After 5 votes:
  → If ≥ 3 True → Authorized (3/5 majority)
  → Cache result for 25 frames (~1 second)
  → Reset votes
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| Check every N frames | 10 | face_recognition takes ~100ms; avoids blocking |
| Votes per batch | 5 | Enough for accuracy, not too slow |
| Majority threshold | 3/5 | Tolerates brief occlusions |
| Cache duration | 25 frames | ~1 second at 25fps; avoids continuous re-checking |
| Face tolerance | 0.55 | Stricter than default (0.6) for security |

**Model used:** HOG (Histogram of Oriented Gradients) — faster than CNN, sufficient for single-face comparison.

**Lazy import:** `import face_recognition` happens inside methods (not at module level) to avoid 3–5 second startup delay from dlib.

---

## 7. Emergency System

Triggered by holding **THREE_FINGERS** gesture for ~3 seconds.

### 7.1 Logic

```
THREE_FINGERS detected → hold_counter++
hold_counter ≥ 75 frames → EMERGENCY TRIGGERED
  → Twilio SMS to emergency number
  → Twilio phone call to emergency number
  → 30-second cooldown before re-triggering
```

### 7.2 UI Feedback

During hold, a **progress bar** fills on the OpenCV window showing time remaining. Text changes to red "EMERGENCY TRIGGERED" when fired.

### 7.3 Twilio Integration

- SMS: `POST /Accounts/{SID}/Messages.json`
- Call: `POST /Accounts/{SID}/Calls.json` with TwiML `<Say>` verb
- Gracefully falls back to console log if Twilio not configured

**Note:** Emergency bypasses face authentication — intentional design decision for accessibility in genuine emergencies.

---

## 8. Backend API

### 8.1 Flask Server (`backend/app.py`)

- Port: **5001** (not 5000 — macOS AirPlay Receiver conflict)
- CORS enabled for local cross-origin requests
- Blueprint-based routing

### 8.2 POST /trigger-command

**Request:**
```json
{ "command": "play_music" }
```

**Validation:**
```python
VALID_COMMANDS = {
    "play_music", "stop_music", "lights_on",
    "lights_off", "weather_report", "emergency_call"
}
```

**Response (success):**
```json
{ "status": "success", "command": "play_music", "delivery": "voice_monkey+aws" }
```

**Delivery flow:**
1. Call Voice Monkey API → trigger Alexa Routine
2. Call AWS API Gateway → Lambda → CloudWatch
3. If AWS fails: retry 2 times with 0.5s delay
4. If Voice Monkey succeeded but AWS failed: return success (voice_monkey delivery)
5. If both fail: return 502 error

### 8.3 Command → Voice Monkey Flow Mapping

```python
COMMAND_FLOW_MAP = {
    "play_music":     "playmusic",
    "stop_music":     "stopmusic",
    "lights_on":      "lightson",
    "lights_off":     "lightsoff",
    "weather_report": "weatherreport",
    "emergency_call": "emergencycall",
}
```

Each entry maps to a Voice Monkey Routine Trigger device, which fires the corresponding Alexa Routine.

---

## 9. Alexa Integration

### 9.1 Dual-Path Architecture

The system uses two complementary paths — neither is redundant:

| Path | Role | Trigger |
|------|------|---------|
| Gesture → Voice Monkey → Echo Dot | **Control** — performs actions silently | Hand gesture |
| Voice → Lambda → DynamoDB → Alexa Skill | **Monitor** — queries gesture history | Voice command |

### 9.2 Custom Alexa Skill (AWS Lambda)

Built using the **Alexa Skills Kit (ASK) SDK for Python**. Deployed on AWS Lambda.

**Intents handled:**

| Intent | Trigger phrase | Purpose |
|--------|---------------|---------|
| GestureStatusIntent | "what was the last gesture" | Reads DynamoDB — reports last gesture + time |
| GestureCountIntent | "how many gestures today" | Reads DynamoDB — reports today's count |
| PlayMusicIntent | "play music" | Voice fallback |
| StopMusicIntent | "stop music" | Voice fallback |
| TurnOnLightsIntent | "turn on the lights" | Voice fallback |
| TurnOffLightsIntent | "turn off the lights" | Voice fallback |
| WeatherIntent | "weather report" | Voice fallback |
| EmergencyIntent | "emergency" | Voice fallback |

**Invocation:** `"Alexa, ask gesture control what was the last gesture"`

### 9.2 Voice Monkey Integration

**Why Voice Monkey?**

Alexa's Proactive Events API (the official way to push speech to Echo Dot without voice trigger) requires:
- OAuth 2.0 account linking
- Published skill with proactive events interface
- Complex client credentials token refresh

Voice Monkey abstracts this as a simple HTTP GET request. It is a **certified Alexa skill** that acts as a proactive notification bridge.

**API call:**
```
GET https://api.voicemonkey.io/trigger
    ?access_token={token}
    &secret_token={secret}
    &monkey={flow_name}
```

**Flow:** Voice Monkey → Routine Trigger Device → Alexa Routine → Echo Dot Action

### 9.3 Alexa Routines

One Alexa Routine per command, configured in the Alexa app:

| Routine | Trigger Device | Action |
|---------|---------------|--------|
| Play Music | playmusic | Play music on Echo Dot |
| Stop Music | stopmusic | Stop media on Echo Dot |
| Lights On | lightson | Smart Home → Lights On |
| Lights Off | lightsoff | Smart Home → Lights Off |
| Weather | weatherreport | Alexa Says → Weather |
| Emergency | emergencycall | Alexa Says → Emergency alert |

**Justification:** Alexa Routines are the standard device configuration layer — equivalent to configuring GPIO pins on a Raspberry Pi. The gesture detection, authentication, and API triggering are entirely in code. Routines are configured once and then driven programmatically.

---

## 10. DynamoDB Gesture Logging

Every gesture command fired is logged to **AWS DynamoDB** (`gesture_commands` table) by the Lambda function.

### 10.1 Table Schema

| Field | Type | Example |
|-------|------|---------|
| `id` | String (UUID) | `"233bb479-cecf-42d6-..."` |
| `command` | String | `"play_music"` |
| `gesture` | String | `"Thumbs Up"` |
| `action` | String | `"play music"` |
| `timestamp` | ISO 8601 String | `"2026-04-07T13:32:11+00:00"` |
| `date` | String | `"2026-04-07"` |
| `time` | String | `"07:02 PM"` |

### 10.2 How It Works

```
Gesture fires → Flask → AWS API Gateway → Lambda → DynamoDB.put_item()
```

Lambda routes between two invocation types:
- **Alexa invocation** — `event["version"]` present → ASK SDK handler
- **Flask invocation** — `event["body"]` present → log to DynamoDB

### 10.3 Voice Queries (GestureStatusIntent)

```
"Alexa, ask gesture control what was the last gesture"
→ Lambda scans DynamoDB → sorts by timestamp → reads latest item
→ "The last gesture was Thumbs Up at 7:02 PM. It triggered the command to play music."

"Alexa, ask gesture control how many gestures today"
→ Lambda filters DynamoDB by today's date → counts items
→ "3 gestures detected today. Most recent was Peace Sign at 7:15 PM."
```

### 10.4 Why DynamoDB Makes Lambda Useful

Without DynamoDB, Lambda's only role was logging to CloudWatch — not queryable by voice. DynamoDB makes the voice path genuinely complementary to the gesture path:

- **Gesture path** = control (input → action)
- **Voice path** = monitoring (query → history)

---

## 11. Data Flow — End to End

```
1. CAPTURE
   Webcam → OpenCV reads frame at ~25fps → flip (mirror effect)

2. PREPROCESS
   Frame → CLAHE (LAB colorspace, L channel) → enhanced frame
   Low-light warning displayed if mean brightness < 30

3. DETECT
   Enhanced frame → MediaPipe Hands → 21 landmarks (x, y, z normalised)

4. CLASSIFY
   Landmarks → Rule-based classifier → raw gesture name (or None)

5. STABILISE
   Raw gesture → GestureBuffer (10-frame window) → stable gesture (or None)

6. AUTHENTICATE
   Frame → face_recognition (every 10th frame) → vote
   After 5 votes → 3/5 majority → authorized: True/False
   Result cached 25 frames

7. DISPATCH (if stable gesture AND authorized AND cooldown elapsed)
   stable gesture → command_mapper → command string
   command string → HTTP POST → Flask /trigger-command

8. DELIVER (Flask backend)
   Path 1: command → Voice Monkey API → Routine Trigger → Alexa Routine → Echo Dot
   Path 2: command → AWS API Gateway → Lambda → DynamoDB (gesture log)

8b. VOICE QUERY (independent, any time)
   "Alexa, ask gesture control what was the last gesture"
   → Lambda → DynamoDB scan → speech response → Echo Dot

9. EMERGENCY (parallel, bypasses auth)
   THREE_FINGERS held 75 frames → Twilio SMS + Call → emergency contact
```

---

## 11. Gesture-to-Command Mapping

| Gesture | Hand Shape | Command | Echo Dot Action |
|---------|-----------|---------|----------------|
| Thumbs Up | Thumb up, fingers curled | play_music | Plays music |
| Thumbs Down | Thumb down below wrist, fingers curled | stop_music | Stops music |
| Open Palm | All 5 fingers extended | lights_on | Turns lights on |
| Closed Fist | All fingers curled to palm | lights_off | Turns lights off |
| Peace / V Sign | Index + middle extended | weather_report | Gives weather |
| Three Fingers (hold 3s) | Index + middle + ring extended | emergency_call | Emergency alert + Twilio |

---

## 12. Environment Configuration

All credentials stored in `.env` (never committed to git):

```env
# Twilio — Emergency
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
EMERGENCY_TO_NUMBER=+91xxxxxxxxxx

# Voice Monkey
VOICE_MONKEY_ACCESS_TOKEN=your_access_token
VOICE_MONKEY_SECRET_TOKEN=your_secret_token
VOICE_MONKEY_FLOW=playmusic

# AWS
AWS_API_GATEWAY_URL=https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/prod

# Flask
FLASK_PORT=5001

# Face Auth
FACE_TOLERANCE=0.55
FACE_AUTH_ENABLED=true

# Gesture Tuning
GESTURE_STABILITY_FRAMES=10
EMERGENCY_HOLD_FRAMES=75
LOW_LIGHT_THRESHOLD=30
```

---

## 13. Setup & Running

### Prerequisites
- Python 3.11 (`/opt/homebrew/bin/python3.11` on M1 Mac)
- Homebrew + cmake (for dlib)
- Webcam

### Installation
```bash
# Create virtual environment
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate

# Install dependencies (order matters)
pip install numpy==1.26.4
brew install cmake
pip install dlib==19.24.2
pip install -r requirements.txt

# Copy and fill credentials
cp .env.template .env
# Edit .env with your credentials

# Enroll face (run once)
python enroll_face.py
```

### Running
```bash
# Terminal 1 — Backend
source venv/bin/activate
python backend/app.py

# Terminal 2 — Main system
source venv/bin/activate
python main.py
```

### Controls
- **S** — Start/pause gesture processing
- **Q** — Quit
- **E** — Capture face (during enrollment only)

---

## 14. Design Decisions & Justifications

### Why rule-based classification instead of ML model?
Rule-based classification using landmark geometry is:
- **Deterministic** — same gesture always produces same result
- **Explainable** — easy to debug and demonstrate
- **Fast** — no model inference overhead
- **Sufficient** — 6 well-separated gestures don't require ML

A trained CNN classifier would add complexity with no accuracy benefit for 6 distinct gestures.

### Why 10-frame stability buffer?
At 25fps, 10 frames = 400ms hold time. This:
- Eliminates false triggers from transitional positions
- Feels natural (short enough to be responsive, long enough to be intentional)
- Tunable via `GESTURE_STABILITY_FRAMES` env variable

### Why batch voting for face auth?
`face_recognition.compare_faces()` takes ~100ms per call on M1. Running every frame would drop FPS to ~10. Batch voting (every 10th frame, 5 votes, 25-frame cache) gives:
- Near-real-time authentication (~2 seconds for full batch)
- Minimal FPS impact
- Noise tolerance (brief occlusions don't de-authorize)

### Why Voice Monkey instead of direct Alexa Proactive Events API?
Alexa's official proactive speech API requires OAuth 2.0 account linking, a published skill, and complex token management. Voice Monkey is a certified Alexa skill that exposes a simple HTTP API — it's an integration layer equivalent to using an SDK instead of raw HTTP calls.

### Why port 5001 instead of 5000?
macOS Monterey and later runs AirPlay Receiver on port 5000. Flask defaults to 5000 which causes `Address already in use` errors.

### Why M1 main-thread constraint?
MediaPipe on Apple Silicon uses CoreML for acceleration. CoreML requires operations on the thread where it was initialized. Any call to `hands.process()` from a non-main thread causes an immediate crash.

---

## 15. Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| Glasses change face encoding | Auth fails after wearing/removing glasses | Re-enroll with current appearance |
| Voice Monkey free tier adds "Someone is at..." prefix | Cosmetic — action still executes | Premium subscription removes it |
| Single enrolled face | Only one owner can be authenticated | System architecture supports extension |
| Emergency bypasses face auth | Potential misuse | Intentional — accessibility in genuine emergencies |
| Drastic lighting changes | Reduced face match confidence | CLAHE mitigates; ensure adequate lighting |
| M1 ARM64 threading constraint | Cannot parallelise MediaPipe | All detection runs on main thread |
| Alexa Routines require one-time manual setup | 6 routines configured in Alexa app | One-time configuration, not code |

---

*Documentation generated for MSRIT CSP67 Mini Project — Gesture-Based Alexa Control System*
