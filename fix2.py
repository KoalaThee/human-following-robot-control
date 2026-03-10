import odrive
from odrive.enums import *

odrv1 = odrive.find_any()
print("Serial:", odrv1.serial_number)

odrv1.axis0.clear_errors()
odrv1.axis1.clear_errors()

odrv1.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE

# Wait until it returns to IDLE
while odrv1.axis0.current_state != AXIS_STATE_IDLE:
    pass

print("After cal:")
print("axis0 axis.error:", odrv1.axis0.error)
print("axis0 motor.error:", odrv1.axis0.motor.error)
print("axis0 encoder.error:", odrv1.axis0.encoder.error)
print("axis0 motor.is_calibrated:", odrv1.axis0.motor.is_calibrated)
print("axis0 encoder.is_ready:", odrv1.axis0.encoder.is_ready)