#!/usr/bin/env python3
"""
Calibrate and test a SINGLE (broken) motor only.
Uses odrv0 = left, odrv1 = right; finds both by serial so assignment is stable.
Verifies the two boards are different (avoids same board used for both).
Runs motor calibration only (no Hall/encoder calibration).

Usage:
  # With BOTH ODrives plugged in, calibrate/test the right (broken) motor:
  python calibrate_and_test_one_motor.py

  # Calibrate/test the left motor instead:
  python calibrate_and_test_one_motor.py --target left
"""

import odrive
from odrive.enums import *
import time
import argparse

# Left and right USB serials (stable: odrv0=left, odrv1=right every time)
ODRV0_SERIAL = "325735623133"   # left
ODRV1_SERIAL = "306F388B3533"   # right (broken)

AXIS_STATE_IDLE = getattr(odrive.enums, "AXIS_STATE_IDLE", 1)
AXIS_STATE_MOTOR_CALIBRATION = getattr(odrive.enums, "AXIS_STATE_MOTOR_CALIBRATION", 4)
AXIS_STATE_CLOSED_LOOP = getattr(odrive.enums, "AXIS_STATE_CLOSED_LOOP_CONTROL", 8)
CONTROL_MODE_VEL = getattr(odrive.enums, "CONTROL_MODE_VELOCITY_CONTROL", 2)
INPUT_MODE_RAMP = getattr(odrive.enums, "INPUT_MODE_VEL_RAMP", 2)

TEST_VELOCITY = 1.0
TEST_DURATION = 3.0


def get_board_serial(odrv):
    try:
        return str(getattr(odrv, "serial_number", None) or getattr(odrv, "serial_str", "") or "?")
    except Exception:
        return "?"


def apply_config(odrv, is_odrv1=False):
    """Apply same config as odrive_calibrate.py. is_odrv1=True => requested_current_range=1 (right)."""
    print("\n--- Applying config ---")
    odrv.axis0.motor.config.current_lim = 12
    odrv.axis0.controller.config.vel_limit = 15
    odrv.config.brake_resistance = 1.0
    odrv.config.dc_max_negative_current = -0.01
    try:
        odrv.config.enable_brake_resistor = True
    except AttributeError:
        pass
    odrv.axis0.motor.config.pole_pairs = 15
    odrv.axis0.motor.config.torque_constant = 8.27 / 37.5
    odrv.axis0.motor.config.motor_type = MOTOR_TYPE_HIGH_CURRENT
    odrv.axis0.motor.config.calibration_current = 5
    odrv.axis0.motor.config.requested_current_range = 10
    odrv.axis0.motor.config.resistance_calib_max_voltage = 2
    odrv.axis0.encoder.config.mode = ENCODER_MODE_HALL
    odrv.axis0.encoder.config.cpr = 90
    print("  ✓ Config applied")


def motor_cal(odrv):
    print("\n--- Motor calibration ---")
    odrv.axis0.requested_state = AXIS_STATE_MOTOR_CALIBRATION
    while odrv.axis0.current_state != AXIS_STATE_IDLE:
        time.sleep(0.1)
    err = odrv.axis0.motor.error
    if err:
        print(f"  ⚠ Motor error: {err} (0x{err:x})")
        if err == 1:
            print("  → MOTOR_ERROR_PHASE_RESISTANCE_OUT_OF_RANGE: measured resistance outside valid range.")
            print("    Check: motor phase wiring (A/B/C), loose connections, correct motor type and pole_pairs.")
            print("    If motor is fine: try odrive_calibrate.py --config with resistance_calib_max_voltage = 4 for this axis.")
        elif err == 2:
            print("  → MOTOR_ERROR_PHASE_INDUCTANCE_OUT_OF_RANGE.")
        return False
    print("  ✓ Motor calibration done")
    return True


def dump_errors(odrv, name="ODrive"):
    print(f"\n{'='*50}")
    print(f"  ERROR DUMP: {name}")
    print("="*50)
    for attr, label in [
        ("error", "odrv.error"),
        ("axis0.error", "axis0.error"),
        ("axis0.motor.error", "motor.error"),
        ("axis0.encoder.error", "encoder.error"),
        ("axis0.current_state", "current_state"),
    ]:
        try:
            obj = odrv
            for part in attr.split("."):
                obj = getattr(obj, part)
            print(f"  {label:20} = {obj}")
        except Exception as e:
            print(f"  {label:20} = (read failed: {e})")
    try:
        print(f"  {'motor.is_calibrated':20} = {odrv.axis0.motor.is_calibrated}")
        print(f"  {'encoder.is_ready':20} = {odrv.axis0.encoder.is_ready}")
    except Exception:
        pass
    print()


def clear_errors(odrv):
    try:
        if hasattr(odrv, "clear_errors"):
            odrv.clear_errors()
            print("  Cleared errors")
            return
    except Exception:
        pass
    try:
        odrv.axis0.error = 0
        odrv.axis0.motor.error = 0
        odrv.axis0.encoder.error = 0
        odrv.axis0.controller.error = 0
        print("  Cleared errors (manual)")
    except Exception as e:
        print(f"  Clear failed: {e}")


def test_spin(odrv):
    print("\n--- Test spin ---")
    clear_errors(odrv)
    time.sleep(0.3)
    odrv.axis0.requested_state = AXIS_STATE_IDLE
    time.sleep(0.2)
    clear_errors(odrv)
    time.sleep(0.2)
    odrv.axis0.controller.config.control_mode = CONTROL_MODE_VEL
    odrv.axis0.controller.config.input_mode = INPUT_MODE_RAMP
    odrv.axis0.controller.input_vel = 0
    odrv.axis0.requested_state = AXIS_STATE_CLOSED_LOOP
    time.sleep(1.0)
    if odrv.axis0.current_state != AXIS_STATE_CLOSED_LOOP:
        print(f"  FAIL: Did not enter closed loop (state={odrv.axis0.current_state})")
        dump_errors(odrv, "after failed closed loop")
        return False
    print(f"  Spinning at {TEST_VELOCITY} rev/s for {TEST_DURATION}s...")
    odrv.axis0.controller.input_vel = TEST_VELOCITY
    time.sleep(TEST_DURATION)
    odrv.axis0.controller.input_vel = 0
    time.sleep(0.5)
    odrv.axis0.requested_state = AXIS_STATE_IDLE
    time.sleep(0.3)
    dump_errors(odrv, "after test")
    ok = (
        getattr(odrv.axis0.motor, "error", 0) == 0
        and getattr(odrv.axis0.encoder, "error", 0) == 0
        and getattr(odrv.axis0, "error", 0) == 0
    )
    return ok


def main():
    parser = argparse.ArgumentParser(description="Calibrate and test one (broken) motor only; uses odrv0=left, odrv1=right by serial")
    parser.add_argument(
        "--target",
        choices=["left", "right", "odrv0", "odrv1"],
        default="right",
        help="Which motor to calibrate and test (default: right)",
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip applying config (only calibrate + test)",
    )
    parser.add_argument(
        "--skip-reboot",
        action="store_true",
        help="Do not reboot after save (test immediately without reconnect)",
    )
    args = parser.parse_args()

    target = "odrv1" if args.target in ("right", "odrv1") else "odrv0"
    target_serial = ODRV1_SERIAL if target == "odrv1" else ODRV0_SERIAL

    print("Calibrate and test ONE motor (odrv0=left, odrv1=right by serial)")
    print(f"  Target: {target} ({args.target})")
    print()

    # Find BOTH ODrives by serial so odrv0/odrv1 are always left/right
    print("Finding odrv0 (left)...")
    try:
        odrv0 = odrive.find_any(serial_number=ODRV0_SERIAL, timeout=10)
    except Exception as e:
        print(f"FAIL: Could not find odrv0 (left) serial {ODRV0_SERIAL}: {e}")
        return 1
    print("Finding odrv1 (right)...")
    try:
        odrv1 = odrive.find_any(serial_number=ODRV1_SERIAL, timeout=10)
    except Exception as e:
        print(f"FAIL: Could not find odrv1 (right) serial {ODRV1_SERIAL}: {e}")
        return 1

    s0 = get_board_serial(odrv0)
    s1 = get_board_serial(odrv1)
    print(f"  odrv0 (left)  board serial: {s0}")
    print(f"  odrv1 (right) board serial: {s1}")

    if s0 and s1 and s0 == s1:
        print()
        print("  ERROR: Both serials returned the SAME board. odrv0 and odrv1 would be the same device.")
        print("  Unplug one ODrive and run again, or check that ODRV0_SERIAL and ODRV1_SERIAL are correct.")
        print()
        return 1

    odrv = odrv1 if target == "odrv1" else odrv0
    target_name = f"{target} ({'right' if target == 'odrv1' else 'left'})"

    print("\n--- Clearing all errors (both controllers) ---")
    clear_errors(odrv0)
    clear_errors(odrv1)
    time.sleep(0.5)

    print(f"\n--- Operating on {target_name} only ---")
    clear_errors(odrv)
    time.sleep(0.3)

    if not args.skip_config:
        apply_config(odrv, is_odrv1=(target == "odrv1"))
        odrv.save_configuration()
        print("  Config saved.")
        time.sleep(0.5)

    if not motor_cal(odrv):
        return 1

    odrv.save_configuration()
    print("\n  Calibration saved.")

    if args.skip_reboot:
        print("  (Skip reboot: testing same connection)")
        ok = test_spin(odrv)
        print("\n  Result:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    print("  Rebooting target...")
    try:
        odrv.reboot()
    except Exception:
        pass
    time.sleep(3)
    print("  Reconnecting to both...")
    try:
        odrv0 = odrive.find_any(serial_number=ODRV0_SERIAL, timeout=15)
        odrv1 = odrive.find_any(serial_number=ODRV1_SERIAL, timeout=15)
    except Exception as e:
        print(f"  Reconnect failed: {e}. Run diagnostic or this script again to test.")
        return 1
    odrv = odrv1 if target == "odrv1" else odrv0

    ok = test_spin(odrv)
    print("\n  Result:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
