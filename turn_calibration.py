#!/usr/bin/env python3
"""
Turn calibration script for the human-following robot.
Turns the robot by a chosen angle (30°, 45°, 90°, 180°) and stops, with tunable
turn speed and braking. Use this to tune STEER_INNER / STEER_RAMP_END and turn
behavior for human_follower.py.

Usage:
  python turn_calibration.py              # interactive menu
  python turn_calibration.py 90           # turn 90° right
  python turn_calibration.py 45 left      # turn 45° left
"""

import odrive
import time
import sys

# ODrive: use numeric values so script works regardless of enum names
AXIS_STATE_CLOSED_LOOP_CONTROL = 8
AXIS_STATE_IDLE = 1
CONTROL_MODE_VELOCITY_CONTROL = 2
INPUT_MODE_PASSTHROUGH = 1

# =====================
# TUNABLE PARAMETERS
# =====================

# ODrive (match human_follower.py / motor_test.py)
ODRV0_SERIAL = "325735623133"   # left
ODRV1_SERIAL = "306F388B3533"   # right
LEFT_MOTOR_DIRECTION = 1
RIGHT_MOTOR_DIRECTION = -1

# Turn speed: wheel velocity in rev/s (increase if 45° or 30° barely moves)
TURN_VELOCITY = 1.8

# Time-based mode: seconds for 90°; short turns use at least MIN_TURN_SEC so motors overcome friction
SECONDS_PER_90_DEG = 1.8
MIN_TURN_SEC = 1.0   # Minimum turn time so 30°/45° actually spin

# Braking: seconds before target to start ramping turn down (longer = brake earlier)
BRAKE_DURATION_SEC = 0.35

# Ramp shape: 0 = linear ramp, >0 = more aggressive start to brake (exponent)
BRAKE_RAMP_EXPONENT = 1.2

# Encoder-based mode (optional): set to True and set DEG_PER_ENCODER_TURN after one 90° run
USE_ENCODER_STOP = False
# After one 90° turn, print (right_pos - left_pos) and set: angle_deg = diff * DEG_PER_ENCODER_TURN
DEG_PER_ENCODER_TURN = 90.0   # e.g. if 90° gave diff 1.5, set 90/1.5 = 60
# Start ramping down when this many degrees from target (encoder mode)
BRAKE_START_DEGREES = 25.0

# ODrive config: 1 = direct (immediate), 2 = vel_ramp
INPUT_MODE_DIRECT = 1


def connect():
    print("Finding ODrives...")
    odrv0 = odrive.find_any(serial_number=ODRV0_SERIAL)
    odrv1 = odrive.find_any(serial_number=ODRV1_SERIAL)
    print("  Left (odrv0) and Right (odrv1) found.")
    time.sleep(0.5)
    odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
    odrv1.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
    time.sleep(0.5)
    odrv0.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
    odrv0.axis0.controller.config.input_mode = INPUT_MODE_DIRECT
    odrv1.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
    odrv1.axis0.controller.config.input_mode = INPUT_MODE_DIRECT
    odrv0.axis0.controller.input_vel = 0
    odrv1.axis0.controller.input_vel = 0
    return odrv0, odrv1


def stop(odrv0, odrv1):
    odrv0.axis0.controller.input_vel = 0
    odrv1.axis0.controller.input_vel = 0


def get_angle_turned(odrv0, odrv1):
    """Approximate heading change (deg) from encoder position difference. Right - left = CCW positive."""
    left_pos = odrv0.axis0.encoder.pos_estimate
    right_pos = odrv1.axis0.encoder.pos_estimate
    diff = (right_pos - left_pos) * RIGHT_MOTOR_DIRECTION * LEFT_MOTOR_DIRECTION
    return diff * DEG_PER_ENCODER_TURN


def run_turn(odrv0, odrv1, target_deg, direction="right"):
    """
    Turn by target_deg. direction in ("right", "left").
    Uses time-based by default; if USE_ENCODER_STOP and encoders available, uses encoder to stop.
    """
    duration_sec = max((target_deg / 90.0) * SECONDS_PER_90_DEG, MIN_TURN_SEC)
    # Turn: same sign to both motors = turn in place. (Opposite signs = forward on your robot.)
    # If the robot turns the wrong way, swap the signs in the two branches below.
    if direction == "left":
        left_vel = -TURN_VELOCITY
        right_vel = -TURN_VELOCITY
        sign = -1
    else:
        left_vel = TURN_VELOCITY
        right_vel = TURN_VELOCITY
        sign = 1

    start_time = time.monotonic()
    start_left = odrv0.axis0.encoder.pos_estimate
    start_right = odrv1.axis0.encoder.pos_estimate

    print("Turning {} {}° (velocity={}, brake_sec={})...".format(
          direction, target_deg, TURN_VELOCITY, BRAKE_DURATION_SEC))
    print("  Motor commands: left={:.2f} right={:.2f}".format(left_vel, right_vel))

    # Send turn command immediately so motors start
    odrv0.axis0.controller.input_vel = left_vel
    odrv1.axis0.controller.input_vel = right_vel
    time.sleep(0.05)

    try:
        while True:
            t = time.monotonic() - start_time

            if USE_ENCODER_STOP:
                angle = sign * get_angle_turned(odrv0, odrv1)
                if angle >= target_deg:
                    break
                remaining = target_deg - angle
                if remaining <= BRAKE_START_DEGREES:
                    # Ramp down: 0 at target, 1 at BRAKE_START_DEGREES
                    ramp = max(0, remaining / BRAKE_START_DEGREES) ** BRAKE_RAMP_EXPONENT
                    odrv0.axis0.controller.input_vel = left_vel * ramp
                    odrv1.axis0.controller.input_vel = right_vel * ramp
                else:
                    odrv0.axis0.controller.input_vel = left_vel
                    odrv1.axis0.controller.input_vel = right_vel
            else:
                if t >= duration_sec:
                    break
                remaining_sec = duration_sec - t
                if remaining_sec <= BRAKE_DURATION_SEC:
                    # Ramp down over last BRAKE_DURATION_SEC
                    ramp = (remaining_sec / BRAKE_DURATION_SEC) ** BRAKE_RAMP_EXPONENT
                    ramp = max(0, min(1, ramp))
                    odrv0.axis0.controller.input_vel = left_vel * ramp
                    odrv1.axis0.controller.input_vel = right_vel * ramp
                else:
                    odrv0.axis0.controller.input_vel = left_vel
                    odrv1.axis0.controller.input_vel = right_vel

            time.sleep(0.02)
    finally:
        stop(odrv0, odrv1)

    elapsed = time.monotonic() - start_time
    if USE_ENCODER_STOP:
        angle = sign * get_angle_turned(odrv0, odrv1)
        print("  Stopped. Angle ~{:.1f}° in {:.2f}s".format(angle, elapsed))
    else:
        print("  Stopped after {:.2f}s (target ~{}° time-based).".format(elapsed, target_deg))


def main():
    print("=" * 50)
    print("  TURN CALIBRATION")
    print("  Tunable: TURN_VELOCITY, SECONDS_PER_90_DEG, BRAKE_DURATION_SEC")
    print("=" * 50)

    odrv0, odrv1 = connect()

    # Parse args: turn_calibration.py [degrees] [left|right]
    if len(sys.argv) >= 2:
        try:
            target = int(sys.argv[1])
            direction = (sys.argv[2].lower() if len(sys.argv) >= 3 else "right")
            if direction not in ("left", "right"):
                direction = "right"
            if target not in (30, 45, 90, 180):
                print("Use 30, 45, 90, or 180 degrees.")
                target = 90
            run_turn(odrv0, odrv1, target, direction)
            stop(odrv0, odrv1)
            time.sleep(0.3)
            odrv0.axis0.requested_state = AXIS_STATE_IDLE
            odrv1.axis0.requested_state = AXIS_STATE_IDLE
            return
        except ValueError:
            pass

    # Interactive menu
    choices = [
        (30, "right"), (45, "right"), (90, "right"), (180, "right"),
        (30, "left"), (45, "left"), (90, "left"), (180, "left"),
    ]
    while True:
        print()
        print("Turn:  1=30°R  2=45°R  3=90°R  4=180°R  5=30°L  6=45°L  7=90°L  8=180°L  q=quit")
        try:
            key = input("Choice: ").strip().lower()
        except EOFError:
            break
        if key == "q":
            break
        if key in "12345678":
            idx = int(key) - 1
            target, direction = choices[idx]
            run_turn(odrv0, odrv1, target, direction)
        else:
            print("Invalid.")

    print("Idling motors.")
    stop(odrv0, odrv1)
    time.sleep(0.3)
    odrv0.axis0.requested_state = AXIS_STATE_IDLE
    odrv1.axis0.requested_state = AXIS_STATE_IDLE


if __name__ == "__main__":
    main()
