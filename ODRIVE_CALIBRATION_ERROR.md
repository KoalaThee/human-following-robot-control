# ODrive Right-Motor Calibration Error

This document explains the **AXIS_ERROR_INVALID_STATE** / “axis not calibrated” issue when only one wheel (the right) spins, and what to do about it.

---

## The Error

When running the motor diagnostic or your application:

- **Left motor (odrv0)** – Works: `motor.is_calibrated = True`, `encoder.is_ready = True`, enters closed loop and spins.
- **Right motor (odrv1)** – Fails: `motor.is_calibrated = False`, `encoder.is_ready = False`. Requesting closed loop sets `axis0.error = 1` (**AXIS_ERROR_INVALID_STATE**) and the axis stays in IDLE.

So the right ODrive refuses to enter closed loop because it does not consider itself calibrated.

---

## What It Means

- **AXIS_ERROR_INVALID_STATE (0x1)** means: “The requested state transition is not allowed.”
- For **closed-loop control**, the ODrive requires:
  1. **Motor calibration** (phase resistance/inductance) completed and valid.
  2. **Encoder calibration** (e.g. Hall phase) completed and valid.
- If either is missing or not stored, the axis will not go to closed loop and reports this error.

So in practice: **the right board is either never calibrated, or its calibration is not present after reboot** (e.g. not saved or not loaded).

---

## Your Setup

- **Two ODrive Minis** (one per wheel).
- **Left**  – USB serial `325735623133` → board serial **55350139171123** → calibrated, works.
- **Right** – USB serial `306F388B3533` → board serial **53254248150323** → reports not calibrated, fails.

---

## What Was Happening When Both Were Plugged In

With **both** ODrives connected, the calibration script does:

1. `find_any(serial_number=ODRV0_SERIAL)` → get first board.
2. `find_any(serial_number=ODRV1_SERIAL)` → get second board.

Depending on USB enumeration, **both calls can return the same device** (often the one that appears first). In that case:

- The script calibrates that **same** board twice (as “odrv0” and “odrv1”).
- The **other** board is never calibrated.
- Result: one wheel works, one reports “not calibrated” and gives AXIS_ERROR_INVALID_STATE.

So the error on the right was explained by: **only one physical board was being calibrated when both were plugged in.**

---

## What You Did: Calibrate One at a Time

You ran:

1. Unplug left → `python odrive_calibrate.py --calibrate-one`  
   - Only board **53254248150323** (right) was connected.  
   - Motor and encoder calibration ran and completed.  
   - Script saved configuration and rebooted.

2. Unplug right, plug left → `python odrive_calibrate.py --calibrate-one`  
   - Only board **55350139171123** (left) was connected.  
   - Motor and encoder calibration ran and completed.  
   - Script saved configuration and rebooted.

3. Plug both in → `python odrive_motor_diagnostic.py`  
   - Left (55350139171123): still calibrated, works.  
   - Right (53254248150323): **still** reports `motor.is_calibrated = False`, `encoder.is_ready = False` and fails with AXIS_ERROR_INVALID_STATE.

So even after a successful calibration and “Saved; rebooting…” on the right board, **after reboot that board no longer shows as calibrated.**

---

## Why the Right Might Still Show “Not Calibrated” After Reboot

Possible reasons:

1. **Save didn’t persist**  
   - `save_configuration()` might have failed or not finished (e.g. USB dropped during write).  
   - Try running `--calibrate-one` on the right again and wait a few seconds after “Saved; rebooting…” before unplugging or power-cycling.

2. **Different firmware or save behavior**  
   - Some ODrive firmware or board variants may not store calibration in the same way, or may require a full power cycle (unplug power, not only USB) after save.

3. **Power cycle vs. reboot**  
   - After “Saved; rebooting…”, try **fully powering off** the right ODrive (main power and USB), then powering it back on, and run the diagnostic again.

4. **Right board needs config first**  
   - If the right board was never updated with your motor/encoder parameters (e.g. from `odrive_calibrate.py --config`), it might be in a state where calibration doesn’t stick or isn’t valid.  
   - Run `--config` with **only the right** ODrive connected so that board gets the same config as the left, then run `--calibrate-one` again.

5. **Hardware / flash**  
   - Rarely, a bad flash or hardware issue on the right board could prevent configuration (including calibration) from persisting across reboots.

---

## Recommended Next Steps

1. **Apply config to the right board only**  
   - Unplug the left ODrive.  
   - Run: `python odrive_calibrate.py --config`  
   - (This will find the only connected board = right, apply config, save, reboot.)

2. **Calibrate the right board again**  
   - With still only the right connected:  
     `python odrive_calibrate.py --calibrate-one`  
   - Wait until the script says “Saved; rebooting…” and give it a few seconds.

3. **Full power cycle**  
   - Power off the right ODrive completely (motor power and USB).  
   - Power it back on, plug both ODrives in, then run:  
     `python odrive_motor_diagnostic.py`

4. **If it still fails**  
   - Run the diagnostic and note the exact error dump for the right board.  
   - Try the same `--config` and `--calibrate-one` sequence on the **left** board only; if the left keeps working after reboot but the right never does, the issue is likely specific to the right board (firmware, save, or hardware).

---

## Quick Reference

| Symptom | Meaning |
|--------|--------|
| `axis0.error = 1` | AXIS_ERROR_INVALID_STATE |
| `motor.is_calibrated = False` | Motor calibration missing or not loaded |
| `encoder.is_ready = False` | Encoder calibration missing or not loaded |
| Only one wheel spins | One ODrive is calibrated and one is not (or one not persisting after reboot) |

**Commands:**

- Calibrate the only connected ODrive:  
  `python odrive_calibrate.py --calibrate-one`
- Apply config (then calibrate if needed):  
  `python odrive_calibrate.py --config`
- Test both motors:  
  `python odrive_motor_diagnostic.py`
