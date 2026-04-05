"""
command_mapper.py
Phase 2 — Maps recognized gesture names to Alexa command strings.

Gesture → Command → Alexa Intent:
  THUMBS_UP     → play_music       → PlayMusicIntent
  THUMBS_DOWN   → stop_music       → StopMusicIntent
  OPEN_PALM     → lights_on        → TurnOnLightsIntent
  CLOSED_FIST   → lights_off       → TurnOffLightsIntent
  PEACE         → weather_report   → WeatherIntent
  THREE_FINGERS → emergency_call   → EmergencyIntent  (requires 3-second hold)
"""

GESTURE_COMMAND_MAP: dict[str, str] = {
    "THUMBS_UP": "play_music",
    "THUMBS_DOWN": "stop_music",
    "OPEN_PALM": "lights_on",
    "CLOSED_FIST": "lights_off",
    "PEACE": "weather_report",
    "THREE_FINGERS": "emergency_call",
}

VALID_COMMANDS = set(GESTURE_COMMAND_MAP.values())

# Human-readable display labels shown in the OpenCV window
GESTURE_DISPLAY_LABELS: dict[str, str] = {
    "THUMBS_UP": "Thumbs Up",
    "THUMBS_DOWN": "Thumbs Down",
    "OPEN_PALM": "Open Palm",
    "CLOSED_FIST": "Closed Fist",
    "PEACE": "Peace / V Sign",
    "THREE_FINGERS": "Three Fingers",
}

COMMAND_DISPLAY_LABELS: dict[str, str] = {
    "play_music": "Play Music",
    "stop_music": "Stop Music",
    "lights_on": "Lights On",
    "lights_off": "Lights Off",
    "weather_report": "Weather Report",
    "emergency_call": "Emergency Call",
}


def get_command(gesture_name: str) -> str | None:
    """Return the command string for a gesture, or None if not mapped."""
    return GESTURE_COMMAND_MAP.get(gesture_name)


def get_display_label(gesture_name: str) -> str:
    """Return a human-readable label for the gesture."""
    return GESTURE_DISPLAY_LABELS.get(gesture_name, gesture_name)


def get_command_label(command: str) -> str:
    """Return a human-readable label for the command."""
    return COMMAND_DISPLAY_LABELS.get(command, command)
