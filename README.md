# Human Following Robot

A differential drive robot that follows a person using YOLOv8n object detection and ODrive motor controllers.

## Features

- Real-time person detection and tracking (YOLOv8n + BotSort)
- Differential drive control for smooth following
- Distance-based following (maintains target distance)
- Safety stops and emergency braking

## Requirements

- Python 3.8+
- ODrive motor controllers (2x)
- USB camera
- YOLOv8n model (`yolov8n.pt` - auto-downloaded on first run)

## Installation

```bash
pip install -r dependencies.txt
```

## Usage

### Main Following Script
```bash
python human_follower.py
```

**Controls:**
- `Q` - Quit
- `R` - Reset target (lock onto new person)
- `S` - Toggle following on/off

### Test Scripts
```bash
python motor_test.py      # Test motor control
python human_detection.py # Test detection only
```

### Wheel Calibration (Wheels Off Ground)
```bash
# Run all calibration tests with analysis
python wheel_calibration_test.py --test all --analyze

# Individual tests
python wheel_calibration_test.py --test step --analyze    # Step response
python wheel_calibration_test.py --test asymmetry         # L/R symmetry
python wheel_calibration_test.py --test sine              # Frequency response

# Visualize results
python calibration_visualizer.py calibration_logs/*.csv
```

### Vision-Motor Integration Testing
```bash
# Test position response (person moves left/right)
python vision_motor_test.py --test position --duration 30 --analyze

# Test distance response (person moves closer/farther)
python vision_motor_test.py --test distance --duration 30 --analyze

# Full integration test
python vision_motor_test.py --test combined --duration 60 --analyze

# Safe mode: log commands without sending to motors
python vision_motor_test.py --test combined --no-motors --analyze

# Visualize results
python integration_visualizer.py integration_logs/*.csv
```

See [CALIBRATION_README.md](CALIBRATION_README.md) for detailed documentation on:
- Motor calibration tests and metrics (latency, symmetry, tracking error)
- Vision-motor integration tests (position/distance response mapping)
- How to interpret results and tune parameters

## Configuration

Edit `human_follower.py` to adjust:

**Motor Direction:**
```python
LEFT_MOTOR_DIRECTION = -1   # -1 to reverse, 1 for normal
RIGHT_MOTOR_DIRECTION = -1
```

**Distance Calibration:**
1. Run script and stand at desired distance
2. Note the "Height: X.XX" value on screen
3. Set `TARGET_BBOX_HEIGHT_RATIO` to that value

**Speed/Tuning:**
- `MAX_VELOCITY` - Maximum wheel speed (rev/sec)
- `TURN_GAIN` - Steering sensitivity
- `SPEED_GAIN` - Forward/backward speed

## Hardware Setup

- **Left Motor:** ODrive serial `325735623133`
- **Right Motor:** ODrive serial `306F388B3533`
- **Camera:** USB camera (index 0)

## License

MIT

