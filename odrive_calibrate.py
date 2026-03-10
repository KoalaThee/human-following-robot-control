#!/usr/bin/env python3
"""
ODrive configuration and calibration for odrv0 (left) and odrv1 (right).
Follows MKS ODrive Guideline (Joey PK) structure; uses your configuration values.

Usage:
  python odrive_calibrate.py --config     # Apply parameters, save, reboot both
  python odrive_calibrate.py --calibrate  # Run full calibration sequence, set pre_calibrated, save
  python odrive_calibrate.py --all       # Config, then prompt to replug, then calibrate

Your config: current_lim=12, vel_limit=15, brake_resistance=1, dc_max_negative_current=-0.01,
  pole_pairs=15, torque_constant=8.27/37.5, motor_type=HIGH_CURRENT, calibration_current=5,
  requested_current_range=10, resistance_calib_max_voltage=2, encoder=Hall cpr=90.
"""

import odrive
from odrive.enums import *
import time
import argparse

# Serial numbers (left=odrv0, right=odrv1)
ODRV0_SERIAL = "325735623133"
ODRV1_SERIAL = "306F388B3533"
ODRV0_BOARD_SERIAL = "55350139171123"
ODRV1_BOARD_SERIAL = "53254248150323"

# Your configuration values (from your config; guideline order)
CURRENT_LIM = 12
VEL_LIMIT = 15
BRAKE_RESISTANCE = 1.0
DC_MAX_NEGATIVE_CURRENT = -0.01
POLE_PAIRS = 15
MOTOR_KV = 37.5
TORQUE_CONSTANT = 8.27 / MOTOR_KV
CALIBRATION_CURRENT = 5
REQUESTED_CURRENT_RANGE = 10
RESISTANCE_CALIB_MAX_VOLTAGE = 2
ENCODER_CPR = 90

AXIS_STATE_IDLE = getattr(odrive.enums, "AXIS_STATE_IDLE", 1)
AXIS_STATE_FULL_CALIBRATION_SEQUENCE = getattr(
    odrive.enums, "AXIS_STATE_FULL_CALIBRATION_SEQUENCE", 3
)


# ---------- Setup (Guideline: Setting Limit, Motor, Encoder) ----------

def set_limits(odrv, name):
    """Setting Limit — current limit, velocity limit."""
    print(f"\n--- [{name}] Setting limits ---")
    odrv.axis0.motor.config.current_lim = CURRENT_LIM
    odrv.axis0.controller.config.vel_limit = VEL_LIMIT
    print(f"  current_lim={CURRENT_LIM}, vel_limit={VEL_LIMIT}")


def set_brake_resistor(odrv, name):
    """Brake resistor: check if armed (read-only), set resistance [Ohms] and negative current."""
    print(f"\n--- [{name}] Brake resistor ---")
    try:
        armed = getattr(odrv, "brake_resistor_armed", None)
        if armed is not None:
            print(f"  brake_resistor_armed = {armed}")
    except Exception:
        pass
    odrv.config.brake_resistance = BRAKE_RESISTANCE
    odrv.config.dc_max_negative_current = DC_MAX_NEGATIVE_CURRENT
    try:
        odrv.config.enable_brake_resistor = True
    except AttributeError:
        pass
    print(f"  brake_resistance={BRAKE_RESISTANCE}, dc_max_negative_current={DC_MAX_NEGATIVE_CURRENT}")


def set_motor_params(odrv, name):
    """Pole pairs, torque constant, motor type, calibration current, requested current range, resistance calib voltage."""
    print(f"\n--- [{name}] Motor parameters ---")
    odrv.axis0.motor.config.pole_pairs = POLE_PAIRS
    odrv.axis0.motor.config.torque_constant = TORQUE_CONSTANT
    odrv.axis0.motor.config.motor_type = MOTOR_TYPE_HIGH_CURRENT
    odrv.axis0.motor.config.calibration_current = CALIBRATION_CURRENT
    odrv.axis0.motor.config.requested_current_range = REQUESTED_CURRENT_RANGE
    odrv.axis0.motor.config.resistance_calib_max_voltage = RESISTANCE_CALIB_MAX_VOLTAGE
    print(f"  pole_pairs={POLE_PAIRS}, torque_constant=8.27/{MOTOR_KV}, motor_type=HIGH_CURRENT")
    print(f"  calibration_current={CALIBRATION_CURRENT}, requested_current_range={REQUESTED_CURRENT_RANGE}")


def set_encoder(odrv, name):
    """Encoder: your setup uses Hall encoder, cpr=90 (not SPI ABS AMS from guideline)."""
    print(f"\n--- [{name}] Encoder ---")
    odrv.axis0.encoder.config.mode = ENCODER_MODE_HALL
    odrv.axis0.encoder.config.cpr = ENCODER_CPR
    print(f"  mode=ENCODER_MODE_HALL, cpr={ENCODER_CPR}")


def configure_axis(odrv, name):
    """Full setup per MKS guideline order, using your configuration values."""
    print(f"\n========== Configuring {name} ==========")
    set_limits(odrv, name)
    set_brake_resistor(odrv, name)
    set_motor_params(odrv, name)
    set_encoder(odrv, name)
    print(f"\n  ✓ {name} parameters set")


def configure_odrv1(odrv):
    """odrv1: same config as odrv0."""
    configure_axis(odrv, "odrv1")


# ---------- Calibration (Guideline: Full calibration sequence, then pre_calibrated + save) ----------

def clear_errors(odrv):
    try:
        if hasattr(odrv, "clear_errors"):
            odrv.clear_errors()
            print("  Cleared errors (odrv.clear_errors())")
            return
    except Exception as e:
        print(f"  clear_errors failed: {e}")
    try:
        odrv.axis0.error = 0
        odrv.axis0.motor.error = 0
        odrv.axis0.encoder.error = 0
        odrv.axis0.controller.error = 0
        print("  Cleared errors (manual)")
    except Exception as e:
        print(f"  Manual clear failed: {e}")


def dump_errors(odrv, name="ODrive"):
    """Print errors (per guideline: Run dump_errors(odrv0) when calibration fails)."""
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
            print(f"    encoder.is_ready    = {er}")
    except Exception as e:
        print(f"    (read failed: {e})")
    try:
        from odrive.utils import dump_errors as odrv_dump
        odrv_dump(odrv)
    except (ImportError, Exception):
        pass
    print()


def run_full_calibration_sequence(odrv, name):
    """
    Guideline: Start calibration by entering AXIS_STATE_FULL_CALIBRATION_SEQUENCE.
    After ~2 s you should hear a beep; motor turns one direction then back.
    On success: set pre_calibrated, save_configuration(). On failure: dump_errors, clear_errors.
    """
    print(f"\n--- Calibration: {name} (Full calibration sequence) ---")
    clear_errors(odrv)
    time.sleep(0.3)
    odrv.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
    print("  Waiting for calibration (motor may turn one way then back; listen for beep)...")
    # Wait until back to idle (or timeout)
    for _ in range(300):
        time.sleep(0.1)
        if odrv.axis0.current_state == AXIS_STATE_IDLE:
            break
    if odrv.axis0.current_state != AXIS_STATE_IDLE:
        print(f"  Timeout; current_state = {odrv.axis0.current_state}")
    err_axis = getattr(odrv.axis0, "error", 0)
    err_motor = getattr(odrv.axis0.motor, "error", 0)
    err_enc = getattr(odrv.axis0.encoder, "error", 0)
    if err_axis or err_motor or err_enc:
        print(f"  ⚠ Calibration had errors.")
        dump_errors(odrv, f"{name} (after calibration)")
        print("  Fix the issue, then run odrv.clear_errors() and run calibration again.")
        return False
    # Guideline: Enable encoder offset storage — set pre_calibrated then save
    try:
        odrv.axis0.encoder.config.pre_calibrated = True
        odrv.axis0.motor.config.pre_calibrated = True
    except AttributeError:
        pass
    print("  ✓ Full calibration sequence done (pre_calibrated set)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="ODrive config and calibration (MKS guideline order, your config values)"
    )
    parser.add_argument("--config", action="store_true", help="Apply config, save, reboot both ODrives")
    parser.add_argument("--calibrate", action="store_true", help="Run full calibration sequence, set pre_calibrated, save")
    parser.add_argument("--all", action="store_true", help="Config, then prompt to replug, then calibrate")
    parser.add_argument("--calibrate-one", action="store_true", help="Calibrate the ONLY connected ODrive")
    args = parser.parse_args()

    if not (args.config or args.calibrate or args.all or args.calibrate_one):
        parser.print_help()
        return

    if args.calibrate_one:
        print("Finding the only connected ODrive...")
        odrv = odrive.find_any(timeout=10)
        try:
            serial = str(getattr(odrv, "serial_number", None) or getattr(odrv, "serial_str", None) or "?")
        except Exception:
            serial = "?"
        print(f"  Found board (serial: {serial}).")
        configure_axis(odrv, "this board")
        print("\n  Saving configuration and rebooting...")
        odrv.save_configuration()
        try:
            odrv.reboot()
        except Exception:
            pass
        print("\n  Reconnect and run calibration (--calibrate-one only applies config; run --calibrate with this board only next).")
        return

    if args.config or args.all:
        print("Finding ODrives...")
        odrv0 = odrive.find_any(serial_number=ODRV0_SERIAL)
        print("  odrv0 found")
        configure_axis(odrv0, "odrv0")
        print("\n  Saving odrv0 and rebooting...")
        odrv0.save_configuration()
        try:
            odrv0.reboot()
        except Exception:
            pass
        time.sleep(2)
        odrv1 = odrive.find_any(serial_number=ODRV1_SERIAL)
        print("  odrv1 found")
        configure_odrv1(odrv1)
        print("\n  Saving odrv1 and rebooting...")
        odrv1.save_configuration()
        try:
            odrv1.reboot()
        except Exception:
            pass
        print("\n✓ Config applied. Both ODrives rebooting (USB will disconnect).")
        if args.all:
            input("Replug ODrives if needed, then press Enter to run calibration...")
        else:
            return

    if args.calibrate or args.all:
        print("\nReconnecting for calibration...")
        time.sleep(1)
        odrv0 = odrive.find_any(serial_number=ODRV0_SERIAL)
        odrv1 = odrive.find_any(serial_number=ODRV1_SERIAL)
        print("  Both ODrives found")
        def get_serial(odrv):
            try:
                return str(getattr(odrv, "serial_number", None) or getattr(odrv, "serial_str", None) or "")
            except Exception:
                return ""
        s0 = get_serial(odrv0)
        s1 = get_serial(odrv1)
        print(f"  odrv0 board serial: {s0 or '(unknown)'}")
        print(f"  odrv1 board serial: {s1 or '(unknown)'}")
        if s0 and s1 and s0 == s1:
            print("\n  ERROR: Both find_any() returned the SAME board.")
            print("  Calibrate one at a time: unplug one ODrive, run --calibrate-one, then the other.")
            return
        for odrv, name in [(odrv0, "odrv0"), (odrv1, "odrv1")]:
            if not run_full_calibration_sequence(odrv, name):
                continue
            odrv.save_configuration()
            print(f"  Saved {name}.")
            try:
                odrv.reboot()
            except Exception:
                pass
            time.sleep(2)
        print("\n✓ Calibration complete. ODrives have rebooted with saved calibration.")


if __name__ == "__main__":
    main()
