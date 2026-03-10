import odrive
from odrive.enums import *

odrv1 = odrive.find_any()  # right board should be the only one connected
print("Serial:", odrv1.serial_number)

# Print a config fingerprint BEFORE changes
print("pole_pairs:", odrv1.axis0.motor.config.pole_pairs)
print("current_lim:", odrv1.axis0.motor.config.current_lim)
print("encoder mode:", odrv1.axis0.encoder.config.mode)
print("encoder cpr:", odrv1.axis0.encoder.config.cpr)

# Also check BOTH axes to avoid axis mismatch
for ax in [odrv1.axis0, odrv1.axis1]:
    print("AXIS", 0 if ax is odrv1.axis0 else 1,
          "motor.is_calibrated:", ax.motor.is_calibrated,
          "encoder.is_ready:", ax.encoder.is_ready,
          "axis.error:", ax.error)