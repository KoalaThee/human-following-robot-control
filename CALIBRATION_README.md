# Wheel Calibration & Metrics Testing System

A comprehensive system for testing and calibrating robot wheel systems with wheels off the ground. This allows you to evaluate motor response, symmetry, and control characteristics before on-ground testing.

## Overview

This system provides:
1. **Data logging** at 100Hz with all relevant motor/sensor signals
2. **Automated test patterns** (steps, ramps, sine waves, etc.)
3. **Metrics computation** (tracking error, latency, symmetry, etc.)
4. **Visualization** tools for analysis

## Quick Start

```bash
# Install additional dependencies
pip install scipy matplotlib

# Run all tests with analysis
python wheel_calibration_test.py --test all --analyze

# Run individual test
python wheel_calibration_test.py --test step --analyze

# Analyze existing data
python wheel_calibration_test.py --analyze-only calibration_logs/step_test_20241201_143022.csv

# Generate plots
python calibration_visualizer.py calibration_logs/*.csv
```

## Test Patterns

### 1. Idle Test (`--test idle`)
Records 30s at command = 0. Measures baseline noise and drift.

### 2. Step Tests (`--test step`)
Steps from 0 → X at multiple levels (20%, 40%, 60%, 80%, 100% of max velocity).
- Measures rise time, settling time, overshoot
- Multiple repetitions for repeatability analysis

### 3. Single Wheel Test (`--test single`)
Commands left wheel only, then right wheel only.
- Measures cross-coupling/isolation between wheels

### 4. Ramp Test (`--test ramp`)
Slow linear ramp from 0 to max and back.
- Reveals hysteresis and lag

### 5. Sine Test (`--test sine`)
Oscillates at frequencies 0.1-2 Hz.
- Measures frequency response/bandwidth
- Shows how fast the control loop can track

### 6. Asymmetry Test (`--test asymmetry`)
Same command to both wheels simultaneously.
- Measures left/right mismatch
- Critical for straight-line driving

### 7. Long Hold Test (`--test long`)
Steady command for 120s.
- Monitors thermal drift and battery sag

### All Tests (`--test all`)
Runs all tests except long hold (takes ~10 minutes).

## Metrics Computed

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| **A. Tracking Error (RMSE/MAE)** | How well motor follows commands | <5% of command |
| **B. Linearity (k, b, R²)** | Linear relationship cmd→speed | k≈1, b≈0, R²>0.99 |
| **C. Latency** | Delay from command to response | <50ms |
| **D. Rise Time** | 10%→90% response time | Depends on application |
| **D. Settling Time** | Time to reach ±5% of final | Depends on application |
| **D. Overshoot** | Peak above target | <20% |
| **F. Symmetry** | Left vs Right difference | Mean diff <0.05 rev/s |
| **G. Cross-coupling** | Inactive wheel response | <5% |
| **H. Noise/Jitter** | Steady-state variation | STD <0.02 rev/s |
| **I. Repeatability** | Run-to-run consistency | STD <0.02 rev/s |
| **J. Battery Sag** | Voltage drop under load | <2V |

## Log File Format (CSV)

```csv
time,left_cmd,right_cmd,left_vel,right_vel,left_rpm,right_rpm,left_pulse_count,right_pulse_count,battery_volt,left_current,right_current,left_temp,right_temp,offset_x
0.000,0.00,0.00,-0.001,0.002,-0.06,0.12,0,0,24.1,0.1,-0.05,28.5,29.1,0
0.010,0.50,0.50,0.15,0.18,9.0,10.8,122,147,24.0,2.3,2.1,28.6,29.2,0
...
```

### Field Descriptions
- `time` - Seconds since test start (float)
- `left_cmd/right_cmd` - Commanded velocity (rev/s)
- `left_vel/right_vel` - Measured velocity (rev/s)
- `left_rpm/right_rpm` - Measured velocity (RPM)
- `left_pulse_count/right_pulse_count` - Encoder pulse count
- `battery_volt` - Bus voltage (V)
- `left_current/right_current` - Motor current (A)
- `left_temp/right_temp` - FET temperature (°C)
- `offset_x` - Reserved for tracking offset

## Interpreting Results

### Tracking Error
```
  Left wheel:  RMSE = 0.0234 rev/s, MAE = 0.0189 rev/s
  Right wheel: RMSE = 0.0256 rev/s, MAE = 0.0201 rev/s
  Average error: 2.1% of command
  ✓ PASS: Error < 5%
```
**Action:** If >10%, tune PID gains on ODrive.

### Linearity
```
  Left wheel: k = 0.987, b = 0.012, R² = 0.9987
    ✓ Good linearity
```
**Action:** If k ≠ 1, apply software scaling: `cmd_scaled = cmd / k`

### Latency
```
  Left wheel: Latency = 32.1 ms (3 samples)
    ✓ Low latency (<50ms)
```
**Action:** If >200ms, reduce controller gains or add feedforward.

### Step Response
```
  Left wheel:
    Rise time (10%→90%): 45.2 ms
    Settling time (±5%): 123.4 ms
    Overshoot: 8.2%
```
**Action:** If overshoot >20%, reduce proportional gain.

### Symmetry
```
  Mean difference (L - R): 0.023 rev/s
  Speed ratio (L/R): 1.018
  ⚠ Left wheel faster - will drift right on ground
  → Suggested calibration: scale right by 1.018
```
**Action:** Apply per-wheel scaling in `set_motor_velocities()`.

### Cross-coupling
```
  Left→Right coupling: 0.8%
  Right→Left coupling: 1.2%
  ✓ Good isolation (<5%)
```
**Action:** If >10%, check electrical isolation or mechanical coupling.

## Applying Calibration Results

After running tests, update `human_follower.py`:

### 1. Symmetry Correction
```python
# Add scaling factors based on symmetry test
LEFT_WHEEL_SCALE = 1.0       # Adjust based on symmetry ratio
RIGHT_WHEEL_SCALE = 1.018    # Scale slower wheel up

def set_motor_velocities(self, left_vel, right_vel):
    # Apply calibration
    left_vel *= LEFT_WHEEL_SCALE
    right_vel *= RIGHT_WHEEL_SCALE
    # ... rest of function
```

### 2. Latency Compensation (if needed)
```python
# If latency is high, reduce gain or add feedforward
TURN_GAIN = 1.5  # Reduce from 2.0 if oscillating
SPEED_GAIN = 3.0  # Reduce from 4.0 if overshooting
```

### 3. Noise Filtering
If noise is high, increase smoothing:
```python
SMOOTH_FACTOR = 0.2  # Lower = smoother (was 0.3)
```

## Generated Files

After running tests:
```
calibration_logs/
├── idle_test_20241201_143022.csv
├── idle_test_20241201_143022_report.md
├── step_test_20241201_143125.csv
├── step_test_20241201_143125_report.md
├── step_test_20241201_143125_overview.png
├── step_test_20241201_143125_linearity.png
├── step_test_20241201_143125_step_response.png
├── step_test_20241201_143125_symmetry.png
├── step_test_20241201_143125_noise.png
├── step_test_20241201_143125_power.png
└── ...
```

## Hardware Configuration

Update these in `wheel_calibration_test.py` to match your setup:

```python
# Motor direction (match human_follower.py)
LEFT_MOTOR_DIRECTION = -1
RIGHT_MOTOR_DIRECTION = -1

# ODrive serial numbers
LEFT_ODRIVE_SERIAL = "325735623133"
RIGHT_ODRIVE_SERIAL = "306F388B3533"

# Encoder PPR (for pulse counting)
PPR = 8192  # Adjust for your encoder

# Test parameters
MAX_CMD = 4.0  # Match your MAX_VELOCITY
```

## Safety Notes

⚠️ **ALWAYS ensure wheels are off the ground before running tests!**

- Secure the robot on a stand or jack
- Keep hands clear of wheels during tests
- Have emergency stop ready (unplug if needed)
- Start with low-speed tests before high-speed

---

# Vision-Motor Integration Testing

Test how the vision/tracking algorithm and motors work together.

## Quick Start

```bash
# Test position response (person moves left/right)
python vision_motor_test.py --test position --duration 30 --analyze

# Test distance response (person moves closer/farther)
python vision_motor_test.py --test distance --duration 30 --analyze

# Full integration test
python vision_motor_test.py --test combined --duration 60 --analyze

# Safe mode: log only, no motor output
python vision_motor_test.py --test combined --no-motors --analyze

# Visualize results
python integration_visualizer.py integration_logs/*.csv
```

## Test Types

| Test | What to Do | What It Measures |
|------|------------|------------------|
| `position` | Move LEFT and RIGHT | Steering response mapping |
| `distance` | Move CLOSER and FARTHER | Speed response mapping |
| `combined` | Move naturally | Full integration |
| `latency` | Make QUICK step movements | Response delay |

## Logged Data Fields

```csv
time, frame_time, detection_time, control_time, total_latency,
detection_valid, track_id,
center_x, center_y, height_ratio, width_ratio, bbox_area,
steering_error, distance_error, position_zone, distance_zone,
left_cmd, right_cmd, left_cmd_smoothed, right_cmd_smoothed,
left_vel_actual, right_vel_actual, left_tracking_error, right_tracking_error,
bus_voltage, left_current, right_current, fps
```

## Integration Metrics

### 1. Timing & Latency
- **Frame capture time**: Camera → buffer
- **Detection time**: YOLO inference + tracking
- **Control time**: Command calculation
- **Total latency**: Vision → motor command (target: <50ms)

### 2. Position Response Mapping
Maps `steering_error` → `turn_differential (L-R)`

| Person Position | Steering Error | Expected Turn Diff |
|-----------------|----------------|-------------------|
| LEFT | negative | negative (turn left) |
| CENTER | ~0 | ~0 |
| RIGHT | positive | positive (turn right) |

**Key metric**: Steering gain = slope of (turn_diff vs steering_error)

### 3. Distance Response Mapping
Maps `distance_error` → `forward_speed`

| Person Distance | Distance Error | Expected Speed |
|-----------------|----------------|----------------|
| FAR | positive | positive (move forward) |
| GOOD | ~0 | ~0 |
| CLOSE | negative | negative (back up) |

**Key metric**: Speed gain = slope of (forward_speed vs distance_error)

### 4. Vision-Motor Synchronization
- **Cross-correlation lag**: Time offset between vision input change and motor response
- **Response delay**: How quickly motors react to step changes in position

## Interpreting Results

### Timing Example
```
  Frame capture:    8.23 ms
  Detection (YOLO): 28.45 ms
  Control calc:     1.12 ms
  ─────────────────────────────
  Total latency:    37.80 ms (±5.23)
  Max latency:      52.10 ms
  Average FPS:      26.4
  ✓ Excellent latency (<50ms)
```

### Position Response Example
```
  Steering Gain (turn_diff / error): 3.892
  Offset bias: 0.0023
  R² fit: 0.9876
  Person LEFT  → Turn diff: -0.823 (should be negative) ✓
  Person RIGHT → Turn diff: +0.891 (should be positive) ✓
  Person CENTER → Turn diff: +0.012 (should be ~0) ✓
```

### Synchronization Example
```
  Vision→Motor sync lag: 23.4 ms
  Cross-correlation peak: 0.892
  ✓ Good synchronization
```

## Generated Plots

| Plot | Shows |
|------|-------|
| `*_overview.png` | Full timeline: vision → errors → commands → actual |
| `*_timing.png` | Latency breakdown and FPS |
| `*_position.png` | Steering error → motor response mapping |
| `*_distance.png` | Distance error → speed mapping |
| `*_motor_tracking.png` | Command vs actual velocities |
| `*_sync.png` | Cross-correlation and response delay |

## Tuning Based on Results

### If latency is too high (>100ms)
1. Reduce YOLO model size or input resolution
2. Increase `SMOOTH_FACTOR` (less smoothing = faster response)
3. Consider prediction/feedforward

### If position response is weak
```python
TURN_GAIN = 2.5  # Increase from 2.0
```

### If distance response is weak
```python
SPEED_GAIN = 5.0  # Increase from 4.0
```

### If motors oscillate
```python
SMOOTH_FACTOR = 0.2  # Decrease from 0.3 (more smoothing)
CENTER_DEADBAND = 0.10  # Increase from 0.08
```

### If sync lag is high
- The smoothing filter adds delay
- Reduce `SMOOTH_FACTOR` or use feedforward compensation

---

## Troubleshooting

### "ODrive not found"
- Check USB connections
- Verify serial numbers match your ODrives
- Try `odrivetool` to confirm ODrives are working

### "No step transitions found"
- Ensure test was run (not just idle data)
- Check that commands are being sent (verify motor wiring)

### High noise in measurements
- Check encoder connections
- Verify encoder resolution (PPR setting)
- May need hardware debouncing

### Large left/right asymmetry
- Check wheel alignment/mounting
- Verify both motors have same configuration
- Could indicate different motor or gearbox characteristics
