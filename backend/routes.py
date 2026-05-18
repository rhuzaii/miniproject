"""
backend/routes.py — Route handlers for the gesture command API.

POST /trigger-command
  Input:  {"command": "play_music", "user": "Rana", "auth_status": "Authorized"}
  Output: {"status": "success", "command": "play_music"}   ← returned IMMEDIATELY

Design: the route validates and returns 200 right away.
Voice Monkey and AWS API Gateway calls happen in a background thread so the
main.py OpenCV loop is never blocked waiting for slow external APIs.
"""

import os
import time
import threading
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

_MAX_RETRIES = 2
_RETRY_DELAY = 0.5   # seconds between AWS retries

COMMAND_FLOW_MAP = {
    "play_music":     "playmusic",
    "stop_music":     "stopmusic",
    "lights_on":      "lightson",
    "lights_off":     "lightsoff",
    "weather_report": "weatherreport",
    "emergency_call": "emergencycall",
}


def _get_voice_monkey_creds() -> tuple[str, str] | tuple[None, None]:
    access = os.getenv("VOICE_MONKEY_ACCESS_TOKEN", "")
    secret = os.getenv("VOICE_MONKEY_SECRET_TOKEN", "")
    if not access or not secret or "your_" in access:
        return None, None
    return access, secret


def _get_gateway_url() -> str | None:
    url = os.getenv("AWS_API_GATEWAY_URL", "")
    if not url or "xxxxxxxxxx" in url:
        return None
    return url.rstrip("/") + "/command"


def _trigger_voice_monkey(command: str) -> bool:
    """Call Voice Monkey → Echo Dot speaks proactively. Runs in background thread."""
    access, secret = _get_voice_monkey_creds()
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
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            print(f"[Routes] Voice Monkey fired: {flow}")
            return True
        print(f"[Routes] Voice Monkey failed: {resp.status_code} {resp.text[:120]}")
    except Exception as e:
        print(f"[Routes] Voice Monkey error: {e}")
    return False


def _log_to_aws(command: str, user: str, auth_status: str) -> bool:
    """POST to AWS API Gateway → Lambda → DynamoDB. Runs in background thread."""
    gateway_url = _get_gateway_url()
    if not gateway_url:
        return False
    payload = {"command": command, "user": user, "auth_status": auth_status}
    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            resp = requests.post(gateway_url, json=payload, timeout=12)
            if resp.status_code == 200:
                print(f"[Routes] AWS logged: {command} by {user}")
                return True
            print(f"[Routes] AWS attempt {attempt} failed: HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            print(f"[Routes] AWS attempt {attempt} timed out")
        except requests.exceptions.ConnectionError:
            print(f"[Routes] AWS attempt {attempt} — cannot reach API Gateway")
        except Exception as e:
            print(f"[Routes] AWS attempt {attempt} error: {e}")
        if attempt <= _MAX_RETRIES:
            time.sleep(_RETRY_DELAY)
    return False


def _dispatch_background(command: str, user: str, auth_status: str) -> None:
    """
    Fire Voice Monkey + AWS logging concurrently in background threads.
    Called from within the background delivery thread (not the Flask request thread).
    """
    vm_thread  = threading.Thread(target=_trigger_voice_monkey, args=(command,), daemon=True)
    aws_thread = threading.Thread(target=_log_to_aws, args=(command, user, auth_status), daemon=True)
    vm_thread.start()
    aws_thread.start()
    # Wait for both so this delivery thread lives until they finish
    vm_thread.join()
    aws_thread.join()


@commands_bp.route("/trigger-command", methods=["POST"])
def trigger_command():
    """
    Receive a gesture command from main.py.
    Validate immediately, return 200, fire API calls in background.
    """
    data = request.get_json(silent=True)

    if not data or "command" not in data:
        return jsonify({"status": "error", "message": "Missing 'command' field"}), 400

    command = data["command"]
    if command not in VALID_COMMANDS:
        return jsonify({
            "status": "error",
            "message": f"Unknown command '{command}'. Valid: {sorted(VALID_COMMANDS)}",
        }), 400

    user        = data.get("user", "Unknown")
    auth_status = data.get("auth_status", "Authorized")

    print(f"[Routes] Queuing: {command}  user: {user}")

    # ── Return 200 IMMEDIATELY — main.py never waits for external APIs ─────────
    threading.Thread(
        target=_dispatch_background,
        args=(command, user, auth_status),
        daemon=True,
    ).start()

    return jsonify({"status": "success", "command": command, "user": user})
