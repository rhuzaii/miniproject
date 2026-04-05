# Setup Guide — Gesture-Based Alexa Control System

This guide covers everything needed to get the system running from a fresh clone.

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| macOS (Apple Silicon M1/M2) | Monterey+ | `uname -m` → arm64 |
| Python 3.11 | 3.11.x | `python3.11 --version` |
| Homebrew | Any | `brew --version` |
| Webcam | Built-in or USB | — |
| Amazon Echo Dot | Any gen | Physical device |
| Amazon account | Same on Echo Dot + Developer Console | — |

---

## Step 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd miniproj
```

---

## Step 2 — Install system dependencies

```bash
# Install cmake (required to build dlib)
brew install cmake
```

---

## Step 3 — Create Python virtual environment

> Must use Python 3.11 — dlib is incompatible with 3.12+

```bash
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
```

---

## Step 4 — Install Python dependencies

Install in this exact order:

```bash
# Step 4a — numpy first (dlib depends on it)
pip install numpy==1.26.4

# Step 4b — dlib (takes 5-10 minutes to compile)
pip install dlib==19.24.2

# Step 4c — everything else
pip install -r requirements.txt
```

---

## Step 5 — Configure environment variables

```bash
cp .env.template .env
```

Open `.env` and fill in all credentials:

```env
# ── Twilio (Emergency SMS + Call) ──────────────────────────────
# Get from: twilio.com → Console Dashboard
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # starts with AC
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1xxxxxxxxxx                         # your Twilio number
EMERGENCY_TO_NUMBER=+91xxxxxxxxxx                       # number to alert

# ── Voice Monkey (Echo Dot control) ────────────────────────────
# Get from: voicemonkey.io → Console → API Tokens
VOICE_MONKEY_ACCESS_TOKEN=your_access_token
VOICE_MONKEY_SECRET_TOKEN=your_secret_token
VOICE_MONKEY_FLOW=playmusic                             # leave as-is

# ── AWS API Gateway ─────────────────────────────────────────────
# Get from: AWS Console → API Gateway → your API → Stages
AWS_API_GATEWAY_URL=https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/prod

# ── Flask ───────────────────────────────────────────────────────
FLASK_PORT=5001                                         # do not change

# ── Face Auth ───────────────────────────────────────────────────
FACE_TOLERANCE=0.55
FACE_AUTH_ENABLED=true                                  # set false to skip during testing

# ── Gesture Tuning ──────────────────────────────────────────────
GESTURE_STABILITY_FRAMES=10
EMERGENCY_HOLD_FRAMES=75
LOW_LIGHT_THRESHOLD=30
```

---

## Step 6 — Set up Voice Monkey

Voice Monkey makes the Echo Dot perform actions when a gesture is detected.

### 6a — Create account and enable Alexa skill
1. Go to **voicemonkey.io** → Sign up / Log in
2. Click **Amazon.IN (IN)** to enable the Voice Monkey Alexa skill
3. Click **Enable to Use** → link your Amazon account
4. In Alexa app → Skills → Voice Monkey → **Manage Permissions** → enable **Notifications**

### 6b — Get API tokens
1. Voice Monkey Console → **API Tokens**
2. Copy **Access Token** and **Secret Token** → paste into `.env`
   - Note: the token shown may be `accesstoken_secrettoken` combined with `_` — split at `_` into two separate values

### 6c — Add Echo Dot as a speaker device
1. Voice Monkey Console → **Devices** → **Add a speaker**
2. Name it `echo dot` → Save
3. Click **Setup and sync**

### 6d — Create 6 Flows (one per command)

For each row below, go to Voice Monkey → **Flows** → **+ New Flow**:

| Flow name | Routine Trigger device name |
|-----------|----------------------------|
| `play-music` | `playmusic` |
| `stop-music` | `stopmusic` |
| `lights-on` | `lightson` |
| `lights-off` | `lightsoff` |
| `weather-report` | `weatherreport` |
| `emergency-call` | `emergencycall` |

Inside each flow:
- Add Action → **Routine Trigger**
- Create trigger device with the name from the table above
- Save

---

## Step 7 — Set up Alexa Routines

For each command, create one Alexa Routine in the Alexa app:

1. Alexa app → **More** → **Routines** → **+**
2. Fill in as per the table below
3. Save

| Command | When (trigger device) | Add action |
|---------|----------------------|------------|
| play_music | `playmusic` | Music & Podcasts → Amazon Music → [song/playlist] → Echo Dot |
| stop_music | `stopmusic` | Device Settings → Stop → Echo Dot |
| lights_on | `lightson` | Smart Home → [your lights] → Turn On |
| lights_off | `lightsoff` | Smart Home → [your lights] → Turn Off |
| weather_report | `weatherreport` | Alexa Says → Weather → Echo Dot |
| emergency_call | `emergencycall` | Alexa Says → Custom → "Emergency alert activated" → Echo Dot |

> **For lights:** requires smart bulbs connected to Alexa (e.g. Philips Hue, TP-Link). If no smart bulbs, use Alexa Says → Custom → "Turning lights on/off".

---

## Step 8 — Deploy AWS Lambda (Alexa Skill backend)

### 8a — Package the Lambda function
```bash
mkdir -p lambda_package
pip install ask-sdk-core -t ./lambda_package
cp alexa_skill/lambda_function.py ./lambda_package/
cd lambda_package && zip -r ../gesture_alexa_lambda.zip . && cd ..
```

### 8b — Upload to AWS Lambda
1. AWS Console → **Lambda** → **Create function**
2. Runtime: **Python 3.11**, Handler: `lambda_function.lambda_handler`
3. Upload `gesture_alexa_lambda.zip`
4. Copy the **Lambda ARN**

### 8c — Set up API Gateway
1. AWS Console → **API Gateway** → **Create API** → REST API
2. Create resource `/command` → POST method → integrate with your Lambda
3. Deploy to stage `prod`
4. Copy the invoke URL → paste into `.env` as `AWS_API_GATEWAY_URL`

### 8d — Create Alexa Skill
1. Go to **developer.amazon.com/alexa/console/ask**
2. Create Skill → Custom → Alexa-hosted (Python) or Self-hosted
3. Invocation name: `gesture control`
4. JSON Editor → paste contents of `alexa_skill/interaction_model.json`
5. Endpoint → Lambda ARN → paste your ARN
6. Build Model → Test in simulator

---

## Step 9 — Enroll your face

Run this once to register the owner's face:

```bash
source venv/bin/activate
python enroll_face.py
```

- A webcam window opens
- Position your face clearly in frame
- Press **E** to capture
- Press **Q** to quit

Face encoding saved to `enrolled_faces/owner.jpg`.

> Skip this step (or set `FACE_AUTH_ENABLED=false` in `.env`) to allow all gestures without authentication.

---

## Step 10 — Run the system

Open **two terminals**:

**Terminal 1 — Start the Flask backend:**
```bash
cd miniproj
source venv/bin/activate
python backend/app.py
```

Expected output:
```
[Flask] Starting on http://localhost:5001
[Flask] AWS Gateway: https://...
 * Running on http://127.0.0.1:5001
```

**Terminal 2 — Start the main gesture system:**
```bash
cd miniproj
source venv/bin/activate
python main.py
```

Expected output (first run takes 30–90 seconds):
```
[Main] Loading MediaPipe (first run may take 30-60s)...
[Main] MediaPipe ready.
[Main] Loading face auth...
[Main] Face auth ready.
[Main] All modules loaded. Opening webcam...
```

---

## Step 11 — Test

In the OpenCV window, press **S** to start gesture processing.

### Quick test via curl (without running main.py)
```bash
curl -s -X POST http://localhost:5001/trigger-command \
  -H "Content-Type: application/json" \
  -d '{"command": "play_music"}'
```

### All 6 commands
```bash
curl -s -X POST http://localhost:5001/trigger-command -H "Content-Type: application/json" -d '{"command": "play_music"}'
curl -s -X POST http://localhost:5001/trigger-command -H "Content-Type: application/json" -d '{"command": "stop_music"}'
curl -s -X POST http://localhost:5001/trigger-command -H "Content-Type: application/json" -d '{"command": "lights_on"}'
curl -s -X POST http://localhost:5001/trigger-command -H "Content-Type: application/json" -d '{"command": "lights_off"}'
curl -s -X POST http://localhost:5001/trigger-command -H "Content-Type: application/json" -d '{"command": "weather_report"}'
curl -s -X POST http://localhost:5001/trigger-command -H "Content-Type: application/json" -d '{"command": "emergency_call"}'
```

---

## Gesture Reference

| Gesture | Hand Shape | Command |
|---------|-----------|---------|
| Thumbs Up | Thumb pointing up, all fingers curled | play_music |
| Thumbs Down | Thumb pointing down below wrist, fingers curled | stop_music |
| Open Palm | All 5 fingers extended open | lights_on |
| Closed Fist | All fingers curled to palm | lights_off |
| Peace Sign | Index + middle extended, others curled | weather_report |
| Three Fingers (hold 3s) | Index + middle + ring extended | emergency_call |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: flask` | Requirements not installed in venv | `pip install -r requirements.txt` |
| `dlib` install fails | cmake missing | `brew install cmake` first |
| `Address already in use` on port 5001 | Another process on 5001 | `lsof -i :5001` → kill the PID |
| main.py takes 60+ seconds to start | MediaPipe CoreML compilation on first run | Wait — it caches after first run |
| Gesture not detected | Lighting too dark or hand not in frame | Ensure good lighting, hold gesture steady |
| Face auth always "Unauthorized" | Glasses changed / lighting different | Re-run `enroll_face.py` |
| Echo Dot not responding | Voice Monkey flow not set up | Verify flow exists in Voice Monkey console |
| Twilio error: invalid username | Wrong Account SID | SID must start with `AC` |
| `FACE_AUTH_ENABLED=false` not working | .env not loaded | Ensure `load_dotenv()` runs before check |

---

## File Summary

| File | Purpose |
|------|---------|
| `main.py` | Entry point — run this to start the system |
| `backend/app.py` | Flask REST API — run this first |
| `gesture_recognition.py` | MediaPipe + gesture classifier |
| `face_auth.py` | Face enrollment + authentication |
| `command_mapper.py` | Gesture → command mapping |
| `emergency.py` | Emergency hold gesture + Twilio |
| `enroll_face.py` | One-time face registration |
| `alexa_skill/lambda_function.py` | AWS Lambda Alexa skill handler |
| `alexa_skill/interaction_model.json` | Alexa skill intent definitions |
| `.env` | All credentials (never commit this) |
| `requirements.txt` | Python dependencies |

---

*MSRIT CSP67 — Gesture-Based Alexa Control System*
