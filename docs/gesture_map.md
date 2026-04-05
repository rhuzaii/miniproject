# Gesture → Command → Alexa Intent Map

| Gesture | Hand Shape | Command String | Alexa Intent | Hold? |
|---|---|---|---|---|
| Thumbs Up | Thumb up, 4 fingers curled | `play_music` | `PlayMusicIntent` | No |
| Thumbs Down | Thumb down, 4 fingers curled | `stop_music` | `StopMusicIntent` | No |
| Open Palm | All 5 fingers extended | `lights_on` | `TurnOnLightsIntent` | No |
| Closed Fist | All fingers curled | `lights_off` | `TurnOffLightsIntent` | No |
| Peace / V Sign | Index + middle extended | `weather_report` | `WeatherIntent` | No |
| Three Fingers | Index + middle + ring extended | `emergency_call` | `EmergencyIntent` | YES — 3 seconds |

## Gesture Classification Logic (MediaPipe Landmarks)

- `is_finger_extended(tip, pip)` → `lm[tip].y < lm[pip].y`
- **Thumbs Up**: `lm[4].y < lm[3].y` AND index/middle/ring/pinky curled
- **Thumbs Down**: `lm[4].y > lm[3].y` AND index/middle/ring/pinky curled
- **Open Palm**: all of index(8,6), middle(12,10), ring(16,14), pinky(20,18) extended
- **Three Fingers**: index + middle + ring extended, pinky curled
- **Peace**: index + middle extended, ring + pinky curled
- **Closed Fist**: all four curled + thumb curled (`lm[4].x > lm[3].x`)

## Stability Buffer

All gestures require **10 consecutive identical frames** before triggering.
Emergency gesture uses the **raw (unstabilized)** gesture for the hold counter.
