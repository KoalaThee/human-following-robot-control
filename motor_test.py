import odrive
from odrive.enums import *
import time

# Serial numbers: odrv0 = left, odrv1 = right (match run_both_motors.py / odrive_calibrate.py)
ODRV0_SERIAL = "325735623133"
ODRV1_SERIAL = "306F388B3533"

# Motor direction: 1 = normal, -1 = reverse. If one wheel spins opposite, set it to -1.
LEFT_MOTOR_DIRECTION = 1
RIGHT_MOTOR_DIRECTION = -1   # reverse right so both wheels "forward" together

# -------------------------
# 1️⃣ Connect to ODrive
# -------------------------
print("Finding ODrive...")
odrv0 = odrive.find_any(serial_number=ODRV0_SERIAL)
odrv1 = odrive.find_any(serial_number=ODRV1_SERIAL)
print("ODrive found!")


# Wait a moment after reboot
time.sleep(2)

# -------------------------
# 2️⃣ Motor ON (closed loop)
# -------------------------
odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
odrv1.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
time.sleep(1)

odrv0.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
odrv0.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP

odrv1.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
odrv1.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
# -------------------------
# 3️⃣ Velocity control test
# -------------------------

print("Spinning motor slowly...")
speed = 1  # rev/sec
odrv0.axis0.controller.input_vel = speed * LEFT_MOTOR_DIRECTION
odrv1.axis0.controller.input_vel = speed * RIGHT_MOTOR_DIRECTION
time.sleep(15)
odrv0.axis0.controller.input_vel = 0
odrv1.axis0.controller.input_vel = 0
print("Velocity test complete.")

# -------------------------
# 4️⃣ Turn motor OFF
# -------------------------
odrv0.axis0.requested_state = AXIS_STATE_IDLE
odrv1.axis0.requested_state = AXIS_STATE_IDLE