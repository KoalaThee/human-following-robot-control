import time
import odrive
from odrive.enums import *

odrv = odrive.find_any()
sn = odrv.serial_number
print("Serial:", sn)

odrv.axis0.clear_errors()

# Calibrate
odrv.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
while odrv.axis0.current_state != AXIS_STATE_IDLE:
    time.sleep(0.05)

print("After calibration:",
      "motor.is_calibrated =", odrv.axis0.motor.is_calibrated,
      "| encoder.is_ready =", odrv.axis0.encoder.is_ready,
      "| axis.error =", odrv.axis0.error)

# Save
print("Saving...")
odrv.save_configuration()
time.sleep(2)  # give flash write time

print("\nNOW DO A FULL POWER CYCLE:")
print("1) Unplug motor power (DC) AND USB")
print("2) Wait 5 seconds")
print("3) Plug motor power back in, then USB")
input("\nPress Enter AFTER you power-cycled...")

odrv2 = odrive.find_any(serial_number=sn)
print("After power cycle:",
      "motor.is_calibrated =", odrv2.axis0.motor.is_calibrated,
      "| encoder.is_ready =", odrv2.axis0.encoder.is_ready,
      "| axis.error =", odrv2.axis0.error)