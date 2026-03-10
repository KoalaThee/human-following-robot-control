import time
import odrive
from odrive.enums import *

odrv = odrive.find_any()
print("Serial:", odrv.serial_number)

# 1) Clear errors
odrv.axis0.clear_errors()

# 2) Calibrate
odrv.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
while odrv.axis0.current_state != AXIS_STATE_IDLE:
    time.sleep(0.05)

print("After calibration:",
      "motor.is_calibrated =", odrv.axis0.motor.is_calibrated,
      "| encoder.is_ready =", odrv.axis0.encoder.is_ready,
      "| axis.error =", odrv.axis0.error)

# 3) Save (IMPORTANT: print result)
ok = odrv.save_configuration()
print("save_configuration() returned:", ok)

# 4) Reboot
print("Rebooting...")
odrv.reboot()
time.sleep(5)

# 5) Reconnect and verify persistence
odrv2 = odrive.find_any(serial_number=odrv.serial_number)
print("After reboot:",
      "motor.is_calibrated =", odrv2.axis0.motor.is_calibrated,
      "| encoder.is_ready =", odrv2.axis0.encoder.is_ready,
      "| axis.error =", odrv2.axis0.error)