<div align="center">

# Smart Irrigation Control System

### Sensors · ML-Assisted Decisions · Actuator Control

</div>

---

## Overview

Compact Python irrigation-control prototype that combines sensor acquisition, signal processing, machine-learning-assisted decisions and pump/valve control.

A broader related implementation with cloud synchronization, cloud-oriented training and synthetic-data tooling is available in [ControladorSistemaRiego](https://github.com/KevinT31/ControladorSistemaRiego).

## Control Flow

~~~mermaid
flowchart LR
    Sensors --> Conditioning[Signal Conditioning]
    Conditioning --> Decision[Decision Engine]
    Decision --> Controller
    Controller --> Actuators

    Model[Serialized ML Model] --> Decision
    Thresholds[Threshold Fallback] --> Decision
~~~

## Main Components

- sensor abstraction
- signal conditioning
- central controller
- ML decision engine
- threshold-based fallback
- pump/valve actuator logic
- model-training utilities
- GUI
- tests

## Tech Stack

Python · pandas · NumPy · scikit-learn · Flask · Google Cloud libraries · Adafruit libraries · Modbus · Matplotlib

## Repository Structure

~~~text
.
├── config/
├── data/
├── docs/
├── lib/
├── scripts/
├── src/
│   ├── main.py
│   ├── controller.py
│   ├── sensors.py
│   ├── signal_conditioning.py
│   ├── decision_engine.py
│   ├── actuators.py
│   ├── model_training.py
│   └── gui.py
├── tests/
├── modelo_actualizado.pkl
└── requirements.txt
~~~

## Run Locally

~~~bash
git clone https://github.com/KevinT31/Siemens.git
cd Siemens

python -m venv .venv
# activate the virtual environment

pip install -r requirements.txt
python src/main.py
~~~

## Degraded / Fallback Behavior

If a trained model is unavailable, the decision engine can continue using predefined thresholds. This keeps the control path testable and avoids making the entire prototype depend on one serialized artifact.

## Scope

This is a compact automation/IoT prototype. It is useful for demonstrating system integration, but it should not be treated as a production-certified irrigation controller.

---

### What this project demonstrates

**Sensor-oriented Python · ML fallback design · actuator control · automation architecture**
