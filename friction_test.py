#!/usr/bin/env python3
"""
Friction threshold test — find the minimum commanded velocity that
actually makes each wheel move (defeats static friction).

Ramps input_vel from 0 upward in small steps, reads back the encoder
velocity, and reports the threshold where motion begins.

Results are printed at the end and can be used to tune
MotorConfig.min_velocity in human_follower.py.
"""

import odrive
from odrive.enums import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_IDLE,
    CONTROL_MODE_VELOCITY_CONTROL,
    INPUT_MODE_PASSTHROUGH,
)
import time

ODRV0_SERIAL = "325735623133"   # left
ODRV1_SERIAL = "306F388B3533"   # right
LEFT_DIR = 1
RIGHT_DIR = -1

VEL_START = 0.0
VEL_END = 0.6
VEL_STEP = 0.01
HOLD_TIME = 0.4       # seconds to hold each step before reading
MOVING_THRESH = 0.02  # encoder rev/s to consider "moving"


def setup_motor(odrv):
    odrv.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
    time.sleep(0.3)
    odrv.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
    odrv.axis0.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
    odrv.axis0.controller.input_vel = 0


def read_actual_vel(odrv) -> float:
    return abs(odrv.axis0.encoder.vel_estimate)


def find_friction_threshold(odrv, direction: int, label: str) -> float | None:
    """Ramp velocity up and return the commanded value where the wheel starts moving."""
    print(f"\n--- {label} (direction={direction:+d}) ---")
    odrv.axis0.controller.input_vel = 0
    time.sleep(0.5)

    vel = VEL_START
    threshold = None
    while vel <= VEL_END:
        cmd = vel * direction
        odrv.axis0.controller.input_vel = cmd
        time.sleep(HOLD_TIME)
        actual = read_actual_vel(odrv)
        bar = "#" * int(actual * 80)
        print(f"  cmd={vel:5.2f}  actual={actual:5.3f}  {bar}")
        if actual >= MOVING_THRESH and threshold is None:
            threshold = vel
            print(f"  >>> MOVING at cmd={vel:.2f} (actual={actual:.3f}) <<<")
        vel = round(vel + VEL_STEP, 4)

    odrv.axis0.controller.input_vel = 0
    time.sleep(0.3)
    return threshold


def main():
    print("Connecting to ODrives...")
    odrv0 = odrive.find_any(serial_number=ODRV0_SERIAL)
    odrv1 = odrive.find_any(serial_number=ODRV1_SERIAL)
    print("✓ Connected\n")

    time.sleep(1)
    setup_motor(odrv0)
    setup_motor(odrv1)
    time.sleep(0.5)

    results = {}

    try:
        results["left_fwd"] = find_friction_threshold(odrv0, LEFT_DIR, "LEFT wheel forward")
        results["left_rev"] = find_friction_threshold(odrv0, -LEFT_DIR, "LEFT wheel reverse")
        results["right_fwd"] = find_friction_threshold(odrv1, RIGHT_DIR, "RIGHT wheel forward")
        results["right_rev"] = find_friction_threshold(odrv1, -RIGHT_DIR, "RIGHT wheel reverse")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        odrv0.axis0.controller.input_vel = 0
        odrv1.axis0.controller.input_vel = 0
        time.sleep(0.3)
        odrv0.axis0.requested_state = AXIS_STATE_IDLE
        odrv1.axis0.requested_state = AXIS_STATE_IDLE

    print("\n" + "=" * 50)
    print("  FRICTION TEST RESULTS")
    print("=" * 50)
    thresholds = []
    for name, val in results.items():
        s = f"{val:.2f} rev/s" if val is not None else "NOT FOUND (didn't move)"
        print(f"  {name:12s}: {s}")
        if val is not None:
            thresholds.append(val)

    if thresholds:
        worst = max(thresholds)
        print(f"\n  Recommended min_velocity: {worst:.2f}")
        print(f"  (worst-case across all wheels/directions)")
    print("=" * 50)


if __name__ == "__main__":
    main()
