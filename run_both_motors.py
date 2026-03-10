#!/usr/bin/env python3
"""
Run both motors after calibration. Uses same serials as odrive_calibrate.py so
odrv0 = left, odrv1 = right every time (no USB order confusion).

Usage: python run_both_motors.py
       python run_both_motors.py --duration 10  # spin for 10 seconds
"""

import odrive
from odrive.enums import *
import time
import argparse

# Same as odrive_calibrate.py: odrv0 = left, odrv1 = right
ODRV0_SERIAL = "325735623133"
ODRV1_SERIAL = "306F388B3533"
# Expected board-reported serials (for verification)
ODRV0_BOARD_SERIAL = "55350139171123"  # left
ODRV1_BOARD_SERIAL = "53254248150323"  # right

AXIS_STATE_IDLE = getattr(odrive.enums, "AXIS_STATE_IDLE", 1)
AXIS_STATE_CLOSED_LOOP = getattr(odrive.enums, "AXIS_STATE_CLOSED_LOOP_CONTROL", 8)
CONTROL_MODE_VEL = getattr(odrive.enums, "CONTROL_MODE_VELOCITY_CONTROL", 2)
INPUT_MODE_RAMP = getattr(odrive.enums, "INPUT_MODE_VEL_RAMP", 2)

DEFAULT_VELOCITY = 1.0   # rev/s
DEFAULT_DURATION = 5.0   # seconds


def get_board_serial(odrv):
    try:
        return str(getattr(odrv, "serial_number", None) or getattr(odrv, "serial_str", "") or "?")
    except Exception:
        return "?"


def clear_errors(odrv):
    try:
        if hasattr(odrv, "clear_errors"):
            odrv.clear_errors()
            return
    except Exception:
        pass
    try:
        odrv.axis0.error = 0
        odrv.axis0.motor.error = 0
        odrv.axis0.encoder.error = 0
        odrv.axis0.controller.error = 0
    except Exception:
        pass


def dump_errors(odrv, name):
    """Print error/calibration state for the failing axis."""
    print(f"\n  --- Error dump: {name} ---")
    try:
        print(f"    axis0.error         = {odrv.axis0.error} (0x{odrv.axis0.error:x})")
        print(f"    motor.error         = {odrv.axis0.motor.error}")
        print(f"    encoder.error       = {odrv.axis0.encoder.error}")
        print(f"    current_state       = {odrv.axis0.current_state}")
        mc = getattr(odrv.axis0.motor, "is_calibrated", None)
        er = getattr(odrv.axis0.encoder, "is_ready", None)
        if mc is not None or er is not None:
            print(f"    motor.is_calibrated = {mc}")
            print(f"    encoder.is_ready   = {er}")
    except Exception as e:
        print(f"    (read failed: {e})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run both motors (odrv0=left, odrv1=right by serial)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help=f"Spin duration in seconds (default {DEFAULT_DURATION})")
    parser.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY, help=f"Velocity in rev/s (default {DEFAULT_VELOCITY})")
    args = parser.parse_args()

    print("Run both motors (same serials as odrive_calibrate: odrv0=left, odrv1=right)")
    print()

    print("Finding odrv0 (left)...")
    try:
        odrv0 = odrive.find_any(serial_number=ODRV0_SERIAL, timeout=10)
    except Exception as e:
        print(f"FAIL: Could not find odrv0 (left) serial {ODRV0_SERIAL}: {e}")
        return 1
    print(f"Connected to ODrive {ODRV0_SERIAL} as odrv0")
    print("Finding odrv1 (right)...")
    try:
        odrv1 = odrive.find_any(serial_number=ODRV1_SERIAL, timeout=10)
    except Exception as e:
        print(f"FAIL: Could not find odrv1 (right) serial {ODRV1_SERIAL}: {e}")
        return 1
    print(f"Connected to ODrive {ODRV1_SERIAL} as odrv1")

    s0 = get_board_serial(odrv0)
    s1 = get_board_serial(odrv1)
    print(f"  odrv0 (left)  board serial: {s0}")
    print(f"  odrv1 (right) board serial: {s1}")
    if s0 and s1 and s0 == s1:
        print("  ERROR: Both serials returned the same board. Check USB/serials.")
        return 1
    if s0 == ODRV0_BOARD_SERIAL and s1 == ODRV1_BOARD_SERIAL:
        print("  ✓ odrv0/odrv1 match expected (left/right correct)")
    elif s0 == ODRV1_BOARD_SERIAL and s1 == ODRV0_BOARD_SERIAL:
        print("  ⚠ Board serials are swapped: odrv0 is right, odrv1 is left. Swap ODRV0_SERIAL and ODRV1_SERIAL in script if needed.")
    print()

    print("Clearing errors on both...")
    clear_errors(odrv0)
    clear_errors(odrv1)
    time.sleep(0.5)

    print("Configuring velocity control...")
    for odrv in (odrv0, odrv1):
        odrv.axis0.controller.config.control_mode = CONTROL_MODE_VEL
        odrv.axis0.controller.config.input_mode = INPUT_MODE_RAMP
        odrv.axis0.controller.input_vel = 0
    time.sleep(0.2)

    print("Requesting closed loop...")
    odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP
    odrv1.axis0.requested_state = AXIS_STATE_CLOSED_LOOP
    time.sleep(1.0)

    if odrv0.axis0.current_state != AXIS_STATE_CLOSED_LOOP or odrv1.axis0.current_state != AXIS_STATE_CLOSED_LOOP:
        print(f"  odrv0 state={odrv0.axis0.current_state}, odrv1 state={odrv1.axis0.current_state}")
        if odrv0.axis0.current_state != AXIS_STATE_CLOSED_LOOP:
            dump_errors(odrv0, "odrv0 (left)")
        if odrv1.axis0.current_state != AXIS_STATE_CLOSED_LOOP:
            dump_errors(odrv1, "odrv1 (right)")
        print("  One or both did not enter closed loop.")
        if odrv1.axis0.current_state == AXIS_STATE_IDLE:
            print("  Right motor (odrv1) is not calibrated or calibration did not persist.")
            print("  Try: python calibrate_and_test_one_motor.py --target right")
            print("  Then power off the right ODrive completely and power back on. See ODRIVE_CALIBRATION_ERROR.md")
        else:
            print("  Run odrive_calibrate.py --calibrate or calibrate_and_test_one_motor.py --target left/right")
        return 1

    print(f"Spinning both at {args.velocity} rev/s for {args.duration}s...")
    odrv0.axis0.controller.input_vel = args.velocity
    odrv1.axis0.controller.input_vel = args.velocity
    time.sleep(args.duration)

    print("Stopping...")
    odrv0.axis0.controller.input_vel = 0
    odrv1.axis0.controller.input_vel = 0
    time.sleep(0.5)
    odrv0.axis0.requested_state = AXIS_STATE_IDLE
    odrv1.axis0.requested_state = AXIS_STATE_IDLE
    print("Done. Both motors idle.")


if __name__ == "__main__":
    exit(main() or 0)
