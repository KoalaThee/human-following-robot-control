import odrive
from odrive.enums import *
import time

# -------------------------
# 1️⃣ Connect to ODrive
# -------------------------
print("Finding ODrive...")
odrv0 = odrive.find_any(serial_number="325735623133")
odrv1 = odrive.find_any(serial_number="306F388B3533")
print("ODrive found!")


# Wait a moment after reboot
time.sleep(2)

# -------------------------
# 3️⃣ Motor ON (closed loop)
# -------------------------
odrv0.axis0.requested_state = 8
odrv1.axis0.requested_state = 8
time.sleep(1)

odrv0.axis0.controller.config.control_mode = 2
odrv0.axis0.controller.config.input_mode = 2

odrv1.axis0.controller.config.control_mode = 2
odrv1.axis0.controller.config.input_mode = 2
# -------------------------
# 4️⃣ Velocity control test
# -------------------------

print("Spinning motor slowly...")
odrv0.axis0.controller.input_vel = 3
odrv1.axis0.controller.input_vel = 3 # rev/sec (very slow)
time.sleep(5)
odrv0.axis0.controller.input_vel = 1
odrv1.axis0.controller.input_vel = 1
time.sleep(5)   
odrv0.axis0.controller.input_vel = 2
odrv1.axis0.controller.input_vel = 2
time.sleep(5)                        # run for 5 seconds
odrv0.axis0.controller.input_vel = 0 
odrv1.axis0.controller.input_vel = 0 # stop motor
print("Velocity test complete.")

# -------------------------
# 6️⃣ Turn motor OFF
# -------------------------
odrv0.axis0.requested_state = AXIS_STATE_IDLE
odrv1.axis0.requested_state = AXIS_STATE_IDLE