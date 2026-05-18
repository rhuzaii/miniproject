"""
alexa_skill/lambda_function.py — AWS Lambda handler for the GestureControl Alexa Skill.

Two roles:
  1. Receives gesture commands from Flask via API Gateway → logs to DynamoDB
  2. Handles Alexa Skill voice intents → GestureStatusIntent reads DynamoDB history

Deploy:
  1. zip this file with dependencies:
       pip install ask-sdk-core boto3 -t ./lambda_package
       cp alexa_skill/lambda_function.py ./lambda_package/
       cd lambda_package && zip -r ../gesture_alexa_lambda.zip . && cd ..
  2. Upload gesture_alexa_lambda.zip to AWS Lambda.
  3. Runtime: Python 3.11, handler: lambda_function.lambda_handler
  4. IAM Role: attach AmazonDynamoDBFullAccess policy
  5. DynamoDB table: gesture_commands (partition key: id, type: String)

Intents handled:
  GestureStatusIntent   — "what was the last gesture" / "show history"
  GestureCountIntent    — "how many gestures today"
  PlayMusicIntent       — voice fallback
  StopMusicIntent       — voice fallback
  TurnOnLightsIntent    — voice fallback
  TurnOffLightsIntent   — voice fallback
  WeatherIntent         — voice fallback
  EmergencyIntent       — voice fallback
"""

import logging
import json
import boto3
import uuid
from datetime import datetime, timezone
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sb = SkillBuilder()

TABLE_NAME = "gesture_commands"

GESTURE_LABELS = {
    "play_music":     "Thumbs Up",
    "stop_music":     "Thumbs Down",
    "lights_on":      "Open Palm",
    "lights_off":     "Closed Fist",
    "weather_report": "Peace Sign",
    "emergency_call": "Three Fingers",
}

ACTION_LABELS = {
    "play_music":     "play music",
    "stop_music":     "stop music",
    "lights_on":      "turn lights on",
    "lights_off":     "turn lights off",
    "weather_report": "get weather report",
    "emergency_call": "trigger emergency alert",
}


# ── DynamoDB helpers ──────────────────────────────────────────────────────────

def _get_table():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    return dynamodb.Table(TABLE_NAME)


def log_command(command: str, user: str = "Unknown", auth_status: str = "Authorized"):
    """Write a gesture command entry to DynamoDB, including who triggered it."""
    try:
        table = _get_table()
        now = datetime.now(timezone.utc)
        table.put_item(Item={
            "id":          str(uuid.uuid4()),
            "command":     command,
            "gesture":     GESTURE_LABELS.get(command, command),
            "action":      ACTION_LABELS.get(command, command),
            "user":        user,
            "auth_status": auth_status,
            "timestamp":   now.isoformat(),
            "date":        now.strftime("%Y-%m-%d"),
            "time":        now.strftime("%I:%M %p"),
        })
        logger.info(f"[DynamoDB] Logged: {command} by {user} ({auth_status}) at {now.isoformat()}")
    except Exception as e:
        logger.error(f"[DynamoDB] Failed to log command: {e}")


def get_last_command():
    """Return the most recent command entry from DynamoDB."""
    try:
        table = _get_table()
        result = table.scan()
        items = result.get("Items", [])
        if not items:
            return None
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return items[0]
    except Exception as e:
        logger.error(f"[DynamoDB] Failed to read: {e}")
        return None


def get_today_commands():
    """Return all commands logged today."""
    try:
        table = _get_table()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr("date").eq(today)
        )
        items = result.get("Items", [])
        items.sort(key=lambda x: x.get("timestamp", ""))
        return items
    except Exception as e:
        logger.error(f"[DynamoDB] Failed to read today: {e}")
        return []


# ── Direct invocation from Flask (API Gateway) ────────────────────────────────

def _handle_direct_invocation(event):
    """
    Called when Flask POSTs to API Gateway → Lambda directly (not via Alexa).
    Logs the command, user identity, and auth status to DynamoDB.
    """
    body = {}
    if event.get("body"):
        try:
            body = json.loads(event["body"])
        except Exception:
            pass

    command     = body.get("command", "unknown")
    user        = body.get("user", "Unknown")
    auth_status = body.get("auth_status", "Authorized")
    log_command(command, user=user, auth_status=auth_status)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"status": "logged", "command": command, "user": user}),
    }


# ── Alexa Skill Handlers ──────────────────────────────────────────────────────

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speech = (
            "Gesture Control is active. "
            "You can ask: what was the last gesture, "
            "or how many gestures today."
        )
        return (
            handler_input.response_builder
            .speak(speech)
            .set_should_end_session(False)
            .response
        )


class GestureStatusIntentHandler(AbstractRequestHandler):
    """'Alexa, ask gesture control what was the last gesture'"""
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("GestureStatusIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        last = get_last_command()
        if not last:
            speech = "No gestures have been detected yet. Start the gesture system and try again."
        else:
            user = last.get("user", "Unknown")
            by_whom = f"by {user} " if user and user != "Unknown" else ""
            speech = (
                f"The last gesture was {last['gesture']}, triggered {by_whom}at {last['time']}. "
                f"It triggered the command to {last['action']}."
            )
        return handler_input.response_builder.speak(speech).response


class GestureCountIntentHandler(AbstractRequestHandler):
    """'Alexa, ask gesture control how many gestures today'"""
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("GestureCountIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        items = get_today_commands()
        count = len(items)
        if count == 0:
            speech = "No gestures have been detected today."
        elif count == 1:
            user = items[0].get("user", "Unknown")
            by_whom = f" by {user}" if user and user != "Unknown" else ""
            speech = f"One gesture has been detected today{by_whom}. It was {items[0]['gesture']}."
        else:
            last = items[-1]
            user = last.get("user", "Unknown")
            by_whom = f" by {user}" if user and user != "Unknown" else ""
            speech = (
                f"{count} gestures have been detected today. "
                f"The most recent was {last['gesture']} at {last['time']}{by_whom}."
            )
        return handler_input.response_builder.speak(speech).response


class PlayMusicIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("PlayMusicIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        log_command("play_music")
        return handler_input.response_builder.speak("Playing music.").response


class StopMusicIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("StopMusicIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        log_command("stop_music")
        return handler_input.response_builder.speak("Stopping music.").response


class TurnOnLightsIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("TurnOnLightsIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        log_command("lights_on")
        return handler_input.response_builder.speak("Turning the lights on.").response


class TurnOffLightsIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("TurnOffLightsIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        log_command("lights_off")
        return handler_input.response_builder.speak("Turning the lights off.").response


class WeatherIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("WeatherIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        log_command("weather_report")
        return handler_input.response_builder.speak("Getting the weather report.").response


class EmergencyIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("EmergencyIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        log_command("emergency_call")
        return handler_input.response_builder.speak("Emergency alert activated.").response


class CancelAndStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return (
            is_intent_name("AMAZON.CancelIntent")(handler_input)
            or is_intent_name("AMAZON.StopIntent")(handler_input)
        )

    def handle(self, handler_input: HandlerInput) -> Response:
        return handler_input.response_builder.speak("Goodbye from Gesture Control.").response


class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        return handler_input.response_builder.response


class FallbackIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        return (
            handler_input.response_builder
            .speak("Try asking: what was the last gesture, or how many gestures today.")
            .set_should_end_session(False)
            .response
        )


sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GestureStatusIntentHandler())
sb.add_request_handler(GestureCountIntentHandler())
sb.add_request_handler(PlayMusicIntentHandler())
sb.add_request_handler(StopMusicIntentHandler())
sb.add_request_handler(TurnOnLightsIntentHandler())
sb.add_request_handler(TurnOffLightsIntentHandler())
sb.add_request_handler(WeatherIntentHandler())
sb.add_request_handler(EmergencyIntentHandler())
sb.add_request_handler(CancelAndStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(FallbackIntentHandler())

_alexa_handler = sb.lambda_handler()


# ── Entry point ───────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Routes between direct Flask invocation and Alexa Skill invocation.
    Alexa calls Lambda directly — 'version' is at top level of event.
    Flask calls via API Gateway — request body is in event['body'].
    """
    if "version" in event:
        # Alexa Skill invocation
        return _alexa_handler(event, context)
    else:
        # Direct Flask → API Gateway invocation
        return _handle_direct_invocation(event)
