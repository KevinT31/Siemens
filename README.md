# Smart Irrigation Control System

Intelligent irrigation prototype that combines sensor acquisition, actuator control, machine-learning-assisted decisions and operational monitoring.

## Overview

This project implements a modular automatic irrigation system in Python. The application reads environmental and hydraulic variables, conditions sensor signals, evaluates irrigation needs and controls actuators through a central controller.

A machine learning decision engine can be loaded from a serialized model. If the model is unavailable, the system falls back to threshold-based decision logic so the control flow can continue in simulation or degraded mode.

## Main Components

- Sensor acquisition and abstraction
- Signal conditioning
- Central irrigation controller
- Machine learning decision engine
- Threshold-based fallback logic
- Pump and valve actuator management
- Model training utilities
- Graphical interface
- Automated tests
- Cloud-related integration dependencies

## Tech Stack

- Python
- pandas / NumPy
- scikit-learn
- Flask
- Google Cloud libraries
- Adafruit sensor libraries
- Modbus
- Matplotlib

## Project Structure

```text
.
├── config/                 # Configuration files
├── data/                   # Local data used by the project
├── docs/                   # Supporting documentation
├── lib/                    # Additional project libraries
├── scripts/                # Utility scripts
├── src/
│   ├── main.py             # Main application entry point
│   ├── controller.py       # Irrigation system orchestration
│   ├── sensors.py          # Sensor acquisition
│   ├── signal_conditioning.py
│   ├── decision_engine.py  # ML and rule-based decisions
│   ├── actuators.py        # Pump/valve control
│   ├── model_training.py   # Model training utilities
│   └── gui.py              # User interface
├── tests/                  # Test suite
├── modelo_actualizado.pkl  # Serialized model
└── requirements.txt
```

## Getting Started

### Requirements

- Python 3
- A virtual environment is recommended

### Installation

```bash
git clone https://github.com/KevinT31/Siemens.git
cd Siemens
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the application from the project root:

```bash
python src/main.py
```

## Decision Logic

The decision engine evaluates sensor values such as humidity, temperature, water level and flow rate. When a trained model is available, it is used to support irrigation decisions. When it is not available, the application applies predefined thresholds as a fallback.

## Portfolio Notes

This repository demonstrates integration between software control, sensor-oriented programming and machine learning for an automation/IoT use case.
