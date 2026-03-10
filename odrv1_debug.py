#!/usr/bin/env python3
"""
odrv1 diagnostic - runs all 4 debug steps:

  Step 1: Confirm odrv1 serial and bus voltage
  Step 2: Detect which axis (axis0 or axis1) the Hall encoder is wired to
  Step 3: Force spin test on the detected axis
  Step 4: Print hardware mapping summary

Usage:
  Unplug odrv0. Leave ONLY odrv1 connected. Then run:
    python odrv1_debug.py
"""

import odrive
from odrive.enums import *
import time

ODRV1_SERIAL = "306F388B3533"   # right board USB serial

AXIS_STATE_IDLE         = getattr(odrive.enums, "AXIS_STATE_IDLE", 1)
AXIS_STATE_CLOSED_LOOP  = getattr(odrive.enums, "AXIS_STATE_CLOSED_LOOP_CONTROL", 8)
CONTROL_MODE_VEL        = getattr(odrive.enums, "CONTROL_MODE_VELOCITY_CONTROL", 2)
INPUT_MODE_PASSTHROUGH  = getattr(odrive.enums, "INPUT_MODE_PASSTHROUGH", 1)

SPIN_VEL      = 2.0   # rev/s for the spin test
SPIN_DURATION = 2.0   # seconds


def sep(title=""):
    print()
    print("=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


# ─── Step 1: Connect and confirm board identity ───────────────────────────────
sep("STEP 1 — Confirm odrv1 identity")
print("Connecting (odrv1 only should be plugged in)...")
try:
    odrv = odrive.find_any(timeout=10)
except Exception as e:
    print(f"FAIL: Could not find any ODrive: {e}")
    exit(1)

try:
    sn = str(getattr(odrv, "serial_number", None) or getattr(odrv, "serial_str", "") or "?")
except Exception:
    sn = "?"

try:
    vbus = odrv.vbus_voltage
except Exception:
    vbus = "?"

print(f"  serial_number : {sn}")
print(f"  vbus_voltage  : {vbus}")

if sn != "?" and sn not in (ODRV1_SERIAL, "53254248150323"):
    print(f"  ⚠ WARNING: This board serial ({sn}) doesn't match known odrv1 serials.")
    print(f"    Expected USB serial {ODRV1_SERIAL} or board serial 53254248150323.")
    print("    Make sure you unplugged odrv0 and only odrv1 is connected.")
else:
    print("  ✓ Board identity looks correct for odrv1")

try:
    print(f"  axis0 state={odrv.axis0.current_state}  motor.is_calibrated={odrv.axis0.motor.is_calibrated}")
except Exception as e:
    print(f"  axis0: (read failed: {e})")
try:
    print(f"  axis1 state={odrv.axis1.current_state}  motor.is_calibrated={odrv.axis1.motor.is_calibrated}")
except Exception:
    print("  axis1: not available (single-axis board)")


# ─── Step 2: Detect which axis the Hall encoder is on ─────────────────────────
sep("STEP 2 — Detect Hall encoder axis (rotate wheel slowly now)")
print("Sampling Hall state for 5 seconds — SLOWLY ROTATE THE WHEEL by hand...")

samples = 50
delay   = 0.1
a0_states = set()
a1_states = set()

for i in range(samples):
    try:
        h0 = odrv.axis0.encoder.hall_state
        a0_states.add(h0)
    except Exception:
        h0 = "N/A"
    try:
        h1 = odrv.axis1.encoder.hall_state
        a1_states.add(h1)
    except Exception:
        h1 = "N/A"
    print(f"  [{i+1:02d}/{samples}]  axis0.hall_state={h0}  axis1.hall_state={h1}", end="\r")
    time.sleep(delay)

print()  # newline after \r loop

a0_changed = len(a0_states) > 1
a1_changed = len(a1_states) > 1

print(f"\n  axis0 Hall states seen: {sorted(a0_states)}  → {'CHANGED (encoder here)' if a0_changed else 'no change'}")
print(f"  axis1 Hall states seen: {sorted(a1_states)}  → {'CHANGED (encoder here)' if a1_changed else 'no change'}")

if a0_changed and not a1_changed:
    detected_axis = 0
    print("\n  → Motor encoder wired to AXIS0. Use odrv.axis0 to command this motor.")
elif a1_changed and not a0_changed:
    detected_axis = 1
    print("\n  → Motor encoder wired to AXIS1. Use odrv.axis1 to command this motor.")
elif a0_changed and a1_changed:
    detected_axis = 0
    print("\n  ⚠ Both axes show Hall changes — can't auto-detect. Defaulting to axis0.")
    print("    (This may indicate wiring cross-talk; check physical connections.)")
else:
    detected_axis = 0
    print("\n  ⚠ No Hall state changes detected on either axis.")
    print("    Either the wheel wasn't rotated, or encoder is not connected to this board.")
    print("    Defaulting to axis0 for the spin test.")


# ─── Step 3: Force spin test on detected axis ─────────────────────────────────
sep(f"STEP 3 — Force spin test on axis{detected_axis}")
ax = odrv.axis0 if detected_axis == 0 else odrv.axis1

print("Clearing errors...")
try:
    odrv.clear_errors()
except Exception:
    try:
        ax.error = 0
        ax.motor.error = 0
        ax.encoder.error = 0
        ax.controller.error = 0
    except Exception:
        pass
time.sleep(0.3)

print(f"  Pre-spin errors:")
try:
    print(f"    axis.error    = {ax.error} (0x{ax.error:x})")
    print(f"    motor.error   = {ax.motor.error}")
    print(f"    encoder.error = {ax.encoder.error}")
    print(f"    is_calibrated = {ax.motor.is_calibrated}")
    print(f"    encoder.ready = {ax.encoder.is_ready}")
except Exception as e:
    print(f"    (read failed: {e})")

print(f"\nRequesting CLOSED_LOOP on axis{detected_axis}...")
ax.requested_state = AXIS_STATE_CLOSED_LOOP
time.sleep(0.5)

print(f"  axis{detected_axis}.current_state = {ax.current_state}")
if ax.current_state != AXIS_STATE_CLOSED_LOOP:
    print("  FAIL: Did not enter closed loop — see error dump below.")
else:
    print("  ✓ Closed loop entered. Setting velocity control + commanding spin...")
    ax.controller.config.control_mode = CONTROL_MODE_VEL
    ax.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
    ax.controller.input_vel = SPIN_VEL
    print(f"  Commanding {SPIN_VEL} rev/s for {SPIN_DURATION}s — does the wheel spin?")
    time.sleep(SPIN_DURATION)
    ax.controller.input_vel = 0.0
    time.sleep(0.3)

ax.requested_state = AXIS_STATE_IDLE
time.sleep(0.2)

print(f"\n  Post-spin errors:")
try:
    print(f"    axis.error    = {ax.error} (0x{ax.error:x})")
    print(f"    motor.error   = {ax.motor.error}")
    print(f"    encoder.error = {ax.encoder.error}")
except Exception as e:
    print(f"    (read failed: {e})")


# ─── Step 4: Hardware mapping summary ─────────────────────────────────────────
sep("STEP 4 — Hardware mapping summary")
print(f"  Board:          odrv1  (serial {sn})")
print(f"  Detected axis:  axis{detected_axis}")
print(f"  Hall encoder →  A{detected_axis} connector on ODrive")
print(f"  Motor phases →  M{detected_axis} connector on ODrive")
print()
print("  If the wheel spun in Step 3:")
print(f"    → Update all scripts to use odrv1.axis{detected_axis} (not the other axis).")
print()
print("  If the wheel did NOT spin and no errors:")
print(f"    → Motor phases may not be connected to M{detected_axis}.")
print("    → Check physical wiring: Hall=A{0}, Motor=M{0} must match.".format(detected_axis, detected_axis))
print()
print("  If errors appeared after closed loop:")
print("    → Paste the error numbers here for next steps.")

if detected_axis != 0:
    sep("ACTION NEEDED")
    print(f"  ⚠ Encoder detected on axis{detected_axis}, but your scripts use axis0 for odrv1.")
    print(f"  Update calibrate_and_test_one_motor.py and run_both_motors.py to use axis{detected_axis}.")
    print(f"  e.g. change all:  odrv1.axis0  →  odrv1.axis{detected_axis}")
