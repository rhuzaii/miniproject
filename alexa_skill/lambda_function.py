"""
alexa_skill/lambda_function.py — AWS Lambda handler for the GestureControl Alexa Skill.

Deploy:
  1. zip this file with ask-sdk-core installed:
       pip install ask-sdk-core -t ./package
       cp lambda_function.py ./package/
       cd package && zip -r ../gesture_alexa_lambda.zip .
  2. Upload gesture_alexa_lambda.zip to AWS Lambda.
  3. Set runtime: Python 3.11, handler: lambda_function.lambda_handler
  4. Connect Lambda ARN as endpoint in Alexa Developer Console.

Intents handled (6 total):
  PlayMusicIntent       — play_music
  StopMusicIntent       — stop_music
  TurnOnLightsIntent    — lights_on
  TurnOffLightsIntent   — lights_off
  WeatherIntent         — weather_report
  EmergencyIntent       — emergency_call
"""

import logging
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sb = SkillBuilder()


class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speech = "Gesture Control is ready. I am listening for gesture commands."
        return (
            handler_input.response_builder
            .speak(speech)
            .set_should_end_session(False)
            .response
        )


class PlayMusicIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("PlayMusicIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("PlayMusicIntent triggered by gesture")
        return handler_input.response_builder.speak("Playing music.").response


class StopMusicIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("StopMusicIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("StopMusicIntent triggered by gesture")
        return handler_input.response_builder.speak("Stopping music.").response


class TurnOnLightsIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("TurnOnLightsIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("TurnOnLightsIntent triggered by gesture")
        return handler_input.response_builder.speak("Turning the lights on.").response


class TurnOffLightsIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("TurnOffLightsIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("TurnOffLightsIntent triggered by gesture")
        return handler_input.response_builder.speak("Turning the lights off.").response


class WeatherIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("WeatherIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("WeatherIntent triggered by gesture")
        return handler_input.response_builder.speak("Getting the weather report.").response


class EmergencyIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("EmergencyIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("EmergencyIntent triggered by gesture")
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
            .speak("Gesture Control did not recognise that command.")
            .set_should_end_session(False)
            .response
        )


sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(PlayMusicIntentHandler())
sb.add_request_handler(StopMusicIntentHandler())
sb.add_request_handler(TurnOnLightsIntentHandler())
sb.add_request_handler(TurnOffLightsIntentHandler())
sb.add_request_handler(WeatherIntentHandler())
sb.add_request_handler(EmergencyIntentHandler())
sb.add_request_handler(CancelAndStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(FallbackIntentHandler())

lambda_handler = sb.lambda_handler()
