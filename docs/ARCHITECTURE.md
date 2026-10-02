# Smart Irrigation Control — Architecture

## Runtime Path

~~~text
Sensors → conditioning → decision engine → controller → actuators
~~~

## Decision Modes

1. ML-assisted decision when a valid model is available.
2. Threshold-based fallback when the model path is unavailable.

## Design Goal

Keep the runtime control loop understandable and testable even when ML artifacts are absent.
