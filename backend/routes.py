"""
backend/routes.py — Route handlers for the gesture command API.

POST /trigger-command
  Input:  {"command": "play_music"}
  Output: {"status": "success", "command": "play_music", "response": {...}}
          {"status": "error", "message": "..."}

Two delivery paths (checked in order):
  1. IFTTT Webhook → Alexa Routine (makes Echo Dot speak proactively)
  2. AWS API Gateway → Lambda → Alexa Skill (fallback / CloudWatch proof)

Note: Lambda alone cannot proactively push speech to an Alexa device.
IFTTT webhooks trigger Alexa Routines which DO make the device speak.
"""

import os
import time
import requests
from flask import Blueprint, request, jsonify

commands_bp = Blueprint("commands", __name__)

VALID_COMMANDS = {
    "play_music",
    "stop_music",
    "lights_on",
    "lights_off",
    "weather_report",
    "emergency_call",
}

# Maps command string → IFTTT event name (must match what you created in IFTTT)
IFTTT_EVENT_MAP = {
    "play_music":     "play_music",
    "stop_music":     "stop_music",
    "lights_on":      "lights_on",
    "lights_off":     "lights_off",
    "weather_report": "weather_report",
    "emergency_call": "emergency_call",
}

_MAX_RETRIES = 2
_RETRY_DELAY = 0.5  # seconds


COMMAND_SPEECH = {
    "play_music":     "Playing music",
    "stop_music":     "Stopping music",
    "lights_on":      "Turning the lights on",
    "lights_off":     "Turning the lights off",
    "weather_report": "Getting the weather report",
    "emergency_call": "Emergency alert activated",
}


def _get_voice_monkey_creds() -> tuple[str, str, str] | tuple[None, None, None]:
    access = os.getenv("VOICE_MONKEY_ACCESS_TOKEN", "")
    secret = os.getenv("VOICE_MONKEY_SECRET_TOKEN", "")
    flow = os.getenv("VOICE_MONKEY_FLOW", "") or os.getenv("VOICE_MONKEY_MONKEY", "")
    if not access or not secret or not flow or "your_" in access:
        return None, None, None
    return access, secret, flow


def _get_gateway_url() -> str | None:
    url = os.getenv("AWS_API_GATEWAY_URL", "")
    if not url or "xxxxxxxxxx" in url:
        return None
    return url.rstrip("/") + "/command"


COMMAND_FLOW_MAP = {
    "play_music":     "playmusic",
    "stop_music":     "stopmusic",
    "lights_on":      "lightson",
    "lights_off":     "lightsoff",
    "weather_report": "weatherreport",
    "emergency_call": "emergencycall",
}


def _trigger_voice_monkey(command: str) -> bool:
    """Call Voice Monkey API → Echo Dot speaks the announcement proactively."""
    access, secret, _ = _get_voice_monkey_creds()
    if not access:
        return False
    flow = COMMAND_FLOW_MAP.get(command)
    if not flow:
        return False
    url = (
        f"https://api.voicemonkey.io/trigger"
        f"?access_token={access}&secret_token={secret}"
        f"&monkey={flow}"
    )
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            print(f"[Routes] Voice Monkey fired: {flow}")
            return True
        print(f"[Routes] Voice Monkey failed: {resp.status_code} {resp.text[:100]}")
    except Exception as e:
        print(f"[Routes] Voice Monkey error: {e}")
    return False


@commands_bp.route("/trigger-command", methods=["POST"])
def trigger_command():
    """Receive a command from main.py, validate, deliver via Voice Monkey + AWS."""
    data = request.get_json(silent=True)

    if not data or "command" not in data:
        return jsonify({"status": "error", "message": "Missing 'command' field"}), 400

    command = data["command"]

    if command not in VALID_COMMANDS:
        return jsonify({
            "status": "error",
            "message": f"Invalid command '{command}'. Valid: {sorted(VALID_COMMANDS)}"
        }), 400

    print(f"[Routes] Received command: {command}")

    # ── Path 1: Voice Monkey → Echo Dot speaks proactively ─────────────────────
    vm_ok = _trigger_voice_monkey(command)

    # ── Path 2: AWS API Gateway → Lambda (CloudWatch logs / proof) ─────────────
    gateway_url = _get_gateway_url()
    if not gateway_url:
        if vm_ok:
            return jsonify({"status": "success", "command": command, "delivery": "voice_monkey"})
        print(f"[Routes] No delivery method configured. Simulating: {command}")
        return jsonify({
            "status": "success",
            "command": command,
            "response": {"simulated": True, "message": f"Would execute: {command}"},
        })

    payload = {"command": command}
    last_error = ""
    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            resp = requests.post(gateway_url, json=payload, timeout=10)
            if resp.status_code == 200:
                print(f"[Routes] AWS responded OK for: {command}")
                return jsonify({
                    "status": "success",
                    "command": command,
                    "delivery": "voice_monkey+aws" if vm_ok else "aws",
                    "response": resp.json() if resp.content else {},
                })
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            print(f"[Routes] AWS attempt {attempt} failed: {last_error}")
        except requests.exceptions.Timeout:
            last_error = "AWS request timed out"
        except requests.exceptions.ConnectionError:
            last_error = "Cannot reach AWS API Gateway"
        except Exception as e:
            last_error = str(e)

        if attempt <= _MAX_RETRIES:
            time.sleep(_RETRY_DELAY)

    if vm_ok:
        return jsonify({"status": "success", "command": command, "delivery": "voice_monkey"})

    return jsonify({
        "status": "error",
        "command": command,
        "message": f"All delivery paths failed: {last_error}",
    }), 502
