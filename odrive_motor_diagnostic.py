#!/usr/bin/env python3
"""
ODrive Mini dual-motor diagnostic: test each motor separately, dump errors, then test both.
Use this when only one wheel spins to find which controller/axis has errors.

Usage: python odrive_motor_diagnostic.py
"""

import odrive
from odrive.enums import *
import time

ODRV0_SERIAL = "325735623133"  # left
ODRV1_SERIAL = "306F388B3533"  # right

# Numeric fallbacks if enums not in this firmware
AXIS_STATE_CLOSED_LOOP = getattr(odrive.enums, "AXIS_STATE_CLOSED_LOOP_CONTROL", 8)
AXIS_STATE_IDLE = getattr(odrive.enums, "AXIS_STATE_IDLE", 1)
CONTROL_MODE_VEL = getattr(odrive.enums, "CONTROL_MODE_VELOCITY_CONTROL", 2)
INPUT_MODE_RAMP = getattr(odrive.enums, "INPUT_MODE_VEL_RAMP", 2)

TEST_VELOCITY = 1.0   # rev/s
TEST_DURATION = 3.0   # seconds per test


def dump_errors(odrv, name="ODrive"):
    """Print all error and state fields for this ODrive."""
    print(f"\n{'='*60}")
    print(f"  ERROR DUMP: {name}")
    print("="*60)
    try:
        e = odrv.error
        print(f"  odrv.error              = {e} (0x{e:x})")
    except AttributeError:
        print("  odrv.error              = (not available)")
    except Exception as ex:
        print(f"  odrv.error              = (read failed: {ex})")
    try:
        a = odrv.axis0.error
        print(f"  axis0.error             = {a} (0x{a:x})")
    except Exception as ex:
        print(f"  axis0.error             = (read failed: {ex})")
    try:
        m = odrv.axis0.motor.error
        print(f"  axis0.motor.error       = {m} (0x{m:x})")
    except Exception as ex:
        print(f"  axis0.motor.error       = (read failed: {ex})")
    try:
        enc = odrv.axis0.encoder.error
        print(f"  axis0.encoder.error    = {enc} (0x{enc:x})")
    except Exception as ex:
        print(f"  axis0.encoder.error    = (read failed: {ex})")
    try:
        c = odrv.axis0.controller.error
        print(f"  axis0.controller.error = {c} (0x{c:x})")
    except Exception as ex:
        print(f"  axis0.controller.error = (read failed: {ex})")
    try:
        state = odrv.axis0.current_state
        print(f"  axis0.current_state     = {state}")
    except Exception as ex:
        print(f"  axis0.current_state     = (read failed: {ex})")
    # Calibration readiness (required for closed loop)
    try:
        mc = getattr(odrv.axis0.motor, "is_calibrated", None)
        er = getattr(odrv.axis0.encoder, "is_ready", None)
        if mc is not None or er is not None:
            print(f"  motor.is_calibrated     = {mc}")
            print(f"  encoder.is_ready        = {er}")
    except Exception:
        pass
    try:
        from odrive.utils import dump_errors as odrv_dump
        odrv_dump(odrv)
    except ImportError:
        pass
    except Exception as ex:
        print(f"  (odrive.utils.dump_errors failed: {ex})")
    print()


def clear_errors(odrv):
    """Clear errors so we can try closed loop again."""
    try:
        if hasattr(odrv, "clear_errors"):
            odrv.clear_errors()
            print("  Cleared errors via odrv.clear_errors()")
            return
    except Exception as ex:
        print(f"  clear_errors() failed: {ex}")
    try:
        odrv.axis0.error = 0
        odrv.axis0.motor.error = 0
        odrv.axis0.encoder.error = 0
        odrv.axis0.controller.error = 0
        print("  Cleared errors by setting axis/motor/encoder/controller.error = 0")
    except Exception as ex:
        print(f"  Manual clear failed: {ex}")


def test_single_motor(serial, name):
    """
    Connect to one ODrive, dump errors, clear, enter closed loop, spin, stop, dump again.
    Returns True if axis reached closed loop and ran without new errors.
    """
    print(f"\n{'#'*60}")
    print(f"  TESTING SINGLE MOTOR: {name} (serial {serial})")
    print("#"*60)
    try:
        odrv = odrive.find_any(serial_number=serial, timeout=10)
    except Exception as e:
        print(f"  FAIL: Could not find ODrive: {e}")
        return False
    try:
        actual_serial = getattr(odrv, "serial_number", None) or getattr(odrv, "serial_str", None)
        if actual_serial is not None:
            print(f"  Found {name}. Board serial: {actual_serial}")
        else:
            print(f"  Found {name}.")
    except Exception:
        print(f"  Found {name}.")
    dump_errors(odrv, name)
    clear_errors(odrv)
    time.sleep(0.5)
    # Ensure idle before requesting closed loop
    try:
        odrv.axis0.requested_state = AXIS_STATE_IDLE
        time.sleep(0.3)
    except Exception as e:
        print(f"  Set IDLE failed: {e}")
    clear_errors(odrv)
    time.sleep(0.2)
    # Request closed loop
    try:
        odrv.axis0.controller.config.control_mode = CONTROL_MODE_VEL
        odrv.axis0.controller.config.input_mode = INPUT_MODE_RAMP
        odrv.axis0.controller.input_vel = 0
        odrv.axis0.requested_state = AXIS_STATE_CLOSED_LOOP
        time.sleep(1.0)
    except Exception as e:
        print(f"  Request closed loop failed: {e}")
        dump_errors(odrv, name + " (after fail)")
        return False
    state = odrv.axis0.current_state
    if state != AXIS_STATE_CLOSED_LOOP:
        print(f"  Axis did not enter closed loop. current_state = {state}")
        if odrv.axis0.error == 1:  # AXIS_ERROR_INVALID_STATE
            print()
            print("  >>> AXIS_ERROR_INVALID_STATE: this axis is NOT CALIBRATED.")
            print("  >>> The right ODrive (odrv1) must be calibrated before closed loop.")
            print("  >>> With BOTH ODrives plugged in, run:")
            print("  >>>   python odrive_calibrate.py --calibrate")
            print("  >>> Then re-run this diagnostic.")
            print()
        dump_errors(odrv, name + " (after request)")
        return False
    print(f"  Closed loop OK. Spinning at {TEST_VELOCITY} rev/s for {TEST_DURATION}s...")
    try:
        odrv.axis0.controller.input_vel = TEST_VELOCITY
        time.sleep(TEST_DURATION)
        odrv.axis0.controller.input_vel = 0
        time.sleep(0.5)
    except Exception as e:
        print(f"  Velocity command failed: {e}")
    odrv.axis0.requested_state = AXIS_STATE_IDLE
    time.sleep(0.3)
    dump_errors(odrv, name + " (after test)")
    motor_ok = (odrv.axis0.motor.error == 0 and odrv.axis0.encoder.error == 0 and odrv.axis0.error == 0)
    if motor_ok:
        print(f"  OK: {name} completed test with no errors.")
    else:
        print(f"  FAIL: {name} has errors after test (see dump above).")
    return motor_ok


def test_both_motors():
    """Connect to both ODrives and run both motors together."""
    print(f"\n{'#'*60}")
    print("  TESTING BOTH MOTORS TOGETHER")
    print("#"*60)
    try:
        odrv0 = odrive.find_any(serial_number=ODRV0_SERIAL, timeout=10)
        odrv1 = odrive.find_any(serial_number=ODRV1_SERIAL, timeout=10)
    except Exception as e:
        print(f"  FAIL: Could not find both ODrives: {e}")
        return False
    print("  Both ODrives found.")
    for odrv, n in [(odrv0, "odrv0"), (odrv1, "odrv1")]:
        dump_errors(odrv, n)
        clear_errors(odrv)
    time.sleep(0.5)
    for odrv in (odrv0, odrv1):
        try:
            odrv.axis0.requested_state = AXIS_STATE_IDLE
            odrv.axis0.controller.config.control_mode = CONTROL_MODE_VEL
            odrv.axis0.controller.config.input_mode = INPUT_MODE_RAMP
            odrv.axis0.controller.input_vel = 0
        except Exception as e:
            print(f"  Config failed: {e}")
            return False
    for odrv in (odrv0, odrv1):
        clear_errors(odrv)
    time.sleep(0.2)
    try:
        odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP
        odrv1.axis0.requested_state = AXIS_STATE_CLOSED_LOOP
        time.sleep(1.0)
    except Exception as e:
        print(f"  Request closed loop failed: {e}")
        return False
    s0 = odrv0.axis0.current_state
    s1 = odrv1.axis0.current_state
    print(f"  odrv0.current_state = {s0},  odrv1.current_state = {s1}")
    if s0 != AXIS_STATE_CLOSED_LOOP or s1 != AXIS_STATE_CLOSED_LOOP:
        print("  One or both axes did not enter closed loop.")
        dump_errors(odrv0, "odrv0 (both test)")
        dump_errors(odrv1, "odrv1 (both test)")
        for odrv in (odrv0, odrv1):
            try:
                odrv.axis0.requested_state = AXIS_STATE_IDLE
            except Exception:
                pass
        return False
    print(f"  Both in closed loop. Spinning both at {TEST_VELOCITY} rev/s for {TEST_DURATION}s...")
    try:
        odrv0.axis0.controller.input_vel = TEST_VELOCITY
        odrv1.axis0.controller.input_vel = TEST_VELOCITY
        time.sleep(TEST_DURATION)
        odrv0.axis0.controller.input_vel = 0
        odrv1.axis0.controller.input_vel = 0
        time.sleep(0.5)
    except Exception as e:
        print(f"  Velocity command failed: {e}")
    for odrv in (odrv0, odrv1):
        odrv.axis0.requested_state = AXIS_STATE_IDLE
    time.sleep(0.3)
    dump_errors(odrv0, "odrv0 (after both test)")
    dump_errors(odrv1, "odrv1 (after both test)")
    ok0 = odrv0.axis0.motor.error == 0 and odrv0.axis0.encoder.error == 0 and odrv0.axis0.error == 0
    ok1 = odrv1.axis0.motor.error == 0 and odrv1.axis0.encoder.error == 0 and odrv1.axis0.error == 0
    if ok0 and ok1:
        print("  OK: Both motors completed with no errors.")
    else:
        if not ok0:
            print("  odrv0 has errors (see dump above).")
        if not ok1:
            print("  odrv1 has errors (see dump above).")
    return ok0 and ok1


def main():
    print("ODrive Mini dual-motor diagnostic")
    print("  odrv0 (left)  serial:", ODRV0_SERIAL)
    print("  odrv1 (right) serial:", ODRV1_SERIAL)
    ok0 = test_single_motor(ODRV0_SERIAL, "odrv0 (left)")
    ok1 = test_single_motor(ODRV1_SERIAL, "odrv1 (right)")
    print("\n" + "="*60)
    if ok0 and ok1:
        print("  Single-motor tests passed. Running both together...")
        both_ok = test_both_motors()
        print("\n  SUMMARY: Single odrv0 OK, Single odrv1 OK, Both together:", "OK" if both_ok else "FAIL")
    else:
        print("  Single-motor tests: odrv0=", "OK" if ok0 else "FAIL", ", odrv1=", "OK" if ok1 else "FAIL")
        if not ok1 and ok0:
            print()
            print("  FIX: Right motor (odrv1) needs calibration. Plug in BOTH ODrives, then run:")
            print("       python odrive_calibrate.py --calibrate")
            print("  Then run this diagnostic again.")
            print()
        print("  Fix errors above before testing both. Re-run script after fixes.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
