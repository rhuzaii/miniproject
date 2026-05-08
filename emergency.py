"""
emergency.py
Phase 4 — Three-finger hold detection + Twilio SMS & call trigger.

Spec:
  - THREE_FINGERS gesture held for >= EMERGENCY_HOLD_SECONDS real seconds (default 3.0)
    triggers emergency. Time-based so fps variations don't affect hold duration.
  - Twilio sends SMS and makes a phone call to EMERGENCY_TO_NUMBER.
  - 30-second cooldown prevents re-triggering spam.
  - Countdown shown on OpenCV frame while holding.
  - "EMERGENCY TRIGGERED" shown for 3 seconds after sending.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

EMERGENCY_HOLD_SECONDS = float(os.getenv("EMERGENCY_HOLD_SECONDS", "3.0"))
_COOLDOWN_SECONDS = 30


class EmergencySystem:
    """
    Tracks THREE_FINGERS hold duration (wall-clock seconds) and fires Twilio on threshold.
    All methods must be called from the main thread.
    """

    def __init__(self):
        self._hold_start: float | None = None   # wall time when hold began
        self._triggered: bool = False
        self._last_trigger_time: float = 0.0
        self._show_triggered_until: float = 0.0  # wall time to show "EMERGENCY TRIGGERED"

        self._twilio_available = self._check_twilio()

    def _check_twilio(self) -> bool:
        required = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                    "TWILIO_FROM_NUMBER", "EMERGENCY_TO_NUMBER"]
        missing = [k for k in required if not os.getenv(k) or "xxxx" in (os.getenv(k) or "")]
        if missing:
            print(f"[Emergency] Twilio not configured. Missing/placeholder: {missing}")
            print("[Emergency] Emergency gesture detected will print to console only.")
            return False
        return True

    def update(self, gesture: str | None) -> bool:
        """
        Call every frame with the raw (unstabilized) gesture.
        Returns True the frame the emergency is triggered.
        Uses wall-clock time so fps variations don't affect hold duration.
        """
        now = time.time()

        # Cooldown guard
        if self._triggered and (now - self._last_trigger_time) < _COOLDOWN_SECONDS:
            if gesture != "THREE_FINGERS":
                self._hold_start = None
            return False

        # Reset triggered flag after cooldown
        if self._triggered and (now - self._last_trigger_time) >= _COOLDOWN_SECONDS:
            self._triggered = False

        if gesture == "THREE_FINGERS":
            if self._hold_start is None:
                self._hold_start = now          # start the clock
            held = now - self._hold_start
            if held >= EMERGENCY_HOLD_SECONDS:
                self._triggered = True
                self._last_trigger_time = now
                self._hold_start = None
                self._show_triggered_until = now + 3.0
                self._send_emergency()
                return True
        else:
            self._hold_start = None
            return False

        return False

    def get_ui_state(self) -> tuple[str | None, float | None]:
        """
        Returns (overlay_text, progress_fraction) for the OpenCV UI.
          - overlay_text: string to display, or None
          - progress_fraction: 0.0–1.0 for progress bar, or None
        """
        now = time.time()

        # Post-trigger message
        if now < self._show_triggered_until:
            return "EMERGENCY TRIGGERED — SMS + Call sent", None

        # Cooldown period (after trigger)
        if self._triggered:
            remaining = _COOLDOWN_SECONDS - (now - self._last_trigger_time)
            return f"Emergency cooldown: {remaining:.0f}s", None

        # Counting down while holding
        if self._hold_start is not None:
            held = now - self._hold_start
            remaining = max(0.0, EMERGENCY_HOLD_SECONDS - held)
            progress = min(1.0, held / EMERGENCY_HOLD_SECONDS)
            return f"Emergency in {remaining:.1f}s...", progress

        return None, None

    def _send_emergency(self) -> None:
        """Send Twilio SMS and make a call. Logs to console in any case."""
        print("[Emergency] EMERGENCY TRIGGERED — sending SMS + call")

        if not self._twilio_available:
            print("[Emergency] (Twilio not configured — printed to console only)")
            return

        try:
            from twilio.rest import Client
            sid = os.environ["TWILIO_ACCOUNT_SID"]
            token = os.environ["TWILIO_AUTH_TOKEN"]
            from_num = os.environ["TWILIO_FROM_NUMBER"]
            to_num = os.environ["EMERGENCY_TO_NUMBER"]

            client = Client(sid, token)

            # SMS
            msg = client.messages.create(
                body="EMERGENCY ALERT: Gesture-based emergency signal triggered. Please check immediately.",
                from_=from_num,
                to=to_num,
            )
            print(f"[Emergency] SMS sent. SID: {msg.sid}")

            # Phone call with TwiML
            call = client.calls.create(
                twiml='<Response><Say>Emergency alert. This is an automated emergency call from your gesture control system. Please check on the user immediately.</Say></Response>',
                from_=from_num,
                to=to_num,
            )
            print(f"[Emergency] Call initiated. SID: {call.sid}")

        except Exception as e:
            print(f"[Emergency] Twilio error: {e}")
