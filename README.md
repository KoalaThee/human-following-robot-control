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

