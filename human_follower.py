#!/usr/bin/env python3
"""
Human Following Robot using YOLOv8n + BotSort tracking and ODrive motors
Follows the first detected person using differential drive

Controls:
- 'q' to quit
- 'r' to reset target (lock onto new person)
- 's' to toggle following on/off (safety stop)
"""

import cv2
from ultralytics import YOLO
import odrive
from odrive.enums import *
import time
import numpy as np

# =====================
# Configuration
# =====================

# ===========================================
# MOTOR DIRECTION - Change sign to reverse
# ===========================================
# Set to -1 to reverse motor direction, 1 for normal
LEFT_MOTOR_DIRECTION = -1   # Reverse left motor
RIGHT_MOTOR_DIRECTION = -1  # Reverse right motor

# Motor settings
MAX_VELOCITY = 4.0      # Maximum wheel velocity (rev/sec)
MIN_VELOCITY = 0.3      # Minimum velocity to overcome static friction
TURN_GAIN = 2.0         # Steering sensitivity (higher = more aggressive turns)
SPEED_GAIN = 4.0        # Forward/backward speed gain

# ===========================================
# DISTANCE CALIBRATION
# ===========================================
# Distance is estimated by how much of the frame the person fills (bounding box height)
# - Larger bbox height ratio = person is CLOSER
# - Smaller bbox height ratio = person is FARTHER
#
# HOW TO CALIBRATE:
# 1. Run the script and stand at your desired following distance
# 2. Look at the on-screen "Height: X.XX" value 
# 3. Set TARGET_BBOX_HEIGHT_RATIO to that value
#
# Example distances (approximate, depends on camera FOV):
#   0.20 = person ~3-4 meters away (far)
#   0.35 = person ~1.5-2 meters away (medium)
#   0.50 = person ~1 meter away (close)
#   0.70 = person very close (<0.5m)
#
TARGET_BBOX_HEIGHT_RATIO = 0.7  # Target: person takes up 35% of frame height
DISTANCE_DEADBAND = 0.05         # ±5% tolerance before adjusting distance
TOO_CLOSE_RATIO = 0.6            # Emergency stop if person fills >60% of frame

# Steering control (based on horizontal center position)
CENTER_DEADBAND = 0.08  # ±8% from center before steering (reduces jitter)

# Tracking
LOST_TRACK_TIMEOUT = 1.5  # Seconds without detection before stopping
SMOOTH_FACTOR = 0.3       # Velocity smoothing (0-1, lower = smoother)

# Camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class HumanFollower:
    def __init__(self):
        self.odrv0 = None  # Left motor
        self.odrv1 = None  # Right motor
        self.model = None
        self.cap = None
        self.target_id = None
        self.last_detection_time = time.time()
        self.following_enabled = True
        
        # Velocity smoothing
        self.current_left_vel = 0.0
        self.current_right_vel = 0.0
        
    def connect_motors(self):
        """Connect to ODrive motor controllers"""
        print("Finding ODrive controllers...")
        print("  Looking for left motor (odrv0)...")
        self.odrv0 = odrive.find_any(serial_number="325735623133")
        print("  Looking for right motor (odrv1)...")
        self.odrv1 = odrive.find_any(serial_number="306F388B3533")
        print("✓ ODrive controllers found!")
        
        time.sleep(1)
        
        # Set to closed loop control (state 8 = AXIS_STATE_CLOSED_LOOP_CONTROL)
        print("Configuring motors for velocity control...")
        self.odrv0.axis0.requested_state = 8
        self.odrv1.axis0.requested_state = 8
        time.sleep(0.5)
        
        # Configure for velocity control (mode 2 = VELOCITY_CONTROL)
        self.odrv0.axis0.controller.config.control_mode = 2
        self.odrv0.axis0.controller.config.input_mode = 2  # VEL_RAMP for smooth accel
        
        self.odrv1.axis0.controller.config.control_mode = 2
        self.odrv1.axis0.controller.config.input_mode = 2
        
        # Initialize velocities to zero
        self.odrv0.axis0.controller.input_vel = 0
        self.odrv1.axis0.controller.input_vel = 0
        
        print("✓ Motors configured")
        
    def stop_motors(self):
        """Stop both motors immediately"""
        if self.odrv0 and self.odrv1:
            self.odrv0.axis0.controller.input_vel = 0
            self.odrv1.axis0.controller.input_vel = 0
            self.current_left_vel = 0
            self.current_right_vel = 0
            
    def set_motor_velocities(self, left_vel, right_vel):
        """Set motor velocities with smoothing"""
        # Clamp velocities to safe range
        left_vel = np.clip(left_vel, -MAX_VELOCITY, MAX_VELOCITY)
        right_vel = np.clip(right_vel, -MAX_VELOCITY, MAX_VELOCITY)
        
        # Apply smoothing (exponential moving average)
        self.current_left_vel += SMOOTH_FACTOR * (left_vel - self.current_left_vel)
        self.current_right_vel += SMOOTH_FACTOR * (right_vel - self.current_right_vel)
        
        # Apply minimum velocity threshold to avoid motor stalling
        final_left = self.current_left_vel
        final_right = self.current_right_vel
        
        if 0 < abs(final_left) < MIN_VELOCITY:
            final_left = MIN_VELOCITY * np.sign(final_left)
        if 0 < abs(final_right) < MIN_VELOCITY:
            final_right = MIN_VELOCITY * np.sign(final_right)
            
        # Small values should be zero (deadband)
        if abs(final_left) < 0.1:
            final_left = 0
        if abs(final_right) < 0.1:
            final_right = 0
        
        # Send to motors (apply direction multipliers)
        self.odrv0.axis0.controller.input_vel = final_left * LEFT_MOTOR_DIRECTION
        self.odrv1.axis0.controller.input_vel = final_right * RIGHT_MOTOR_DIRECTION
        
        return final_left, final_right
        
    def shutdown_motors(self):
        """Safely shutdown motors to idle state"""
        if self.odrv0 and self.odrv1:
            self.stop_motors()
            time.sleep(0.3)
            self.odrv0.axis0.requested_state = AXIS_STATE_IDLE
            self.odrv1.axis0.requested_state = AXIS_STATE_IDLE
            print("✓ Motors set to idle")
            
    def initialize_camera(self):
        """Initialize camera and YOLO model"""
        print("Loading YOLOv8n model...")
        self.model = YOLO('yolov8n.pt')
        
        print("Opening camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        print("✓ Camera initialized")
        
    def find_target_person(self, results):
        """
        Find the target person in detection results.
        Locks onto first detected person and tracks them by ID.
        
        Note: YOLO is already filtered to only detect persons (classes=[0])
        
        Returns: (center_x_normalized, height_ratio, track_id) or None
        """
        if not results or len(results) == 0:
            return None
            
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
            
        # If we have a target ID, try to find that specific person
        if self.target_id is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            for i in range(len(boxes)):
                if int(ids[i]) == self.target_id:
                    return self._extract_box_info(boxes[i], self.target_id)
                    
        # Target not found or no target set - pick first tracked person
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            track_id = int(ids[0])
            self.target_id = track_id
            print(f"🎯 Locked onto person ID: {self.target_id}")
            return self._extract_box_info(boxes[0], self.target_id)
                    
        # Fallback: use first detection without tracking ID
        return self._extract_box_info(boxes[0], None)
        
    def _extract_box_info(self, box, track_id):
        """Extract normalized position and size from bounding box"""
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        
        # Calculate center X (normalized 0-1, where 0.5 is center)
        center_x = ((x1 + x2) / 2) / FRAME_WIDTH
        
        # Calculate height ratio (proxy for distance - larger = closer)
        bbox_height = y2 - y1
        height_ratio = bbox_height / FRAME_HEIGHT
        
        return (center_x, height_ratio, track_id)
        
    def calculate_motor_commands(self, center_x, height_ratio):
        """
        Calculate differential drive motor velocities.
        
        Differential drive principles:
        - Both wheels same speed forward = move forward
        - Both wheels same speed backward = move backward  
        - Left faster than right = turn right
        - Right faster than left = turn left
        
        Args:
            center_x: 0.0 (left edge) to 1.0 (right edge), 0.5 is center
            height_ratio: bbox height / frame height (larger = person closer)
            
        Returns: (left_velocity, right_velocity)
        """
        # ===== STEERING CALCULATION =====
        # Error: positive = person on right side, negative = person on left
        steering_error = center_x - 0.5
        
        # Apply deadband to reduce jitter when person is roughly centered
        if abs(steering_error) < CENTER_DEADBAND:
            steering_error = 0
            
        # ===== DISTANCE/SPEED CALCULATION =====
        # Error: positive = person too far, negative = person too close
        distance_error = TARGET_BBOX_HEIGHT_RATIO - height_ratio
        
        # Safety: emergency stop if person is too close
        if height_ratio > TOO_CLOSE_RATIO:
            return 0, 0
            
        # Apply deadband for distance
        if abs(distance_error) < DISTANCE_DEADBAND:
            distance_error = 0
            
        # Calculate forward speed based on distance
        # Positive error (too far) = move forward, negative (too close) = back up
        forward_speed = distance_error * SPEED_GAIN
        
        # Limit backward speed more than forward (safety)
        if forward_speed < 0:
            forward_speed = max(forward_speed, -MAX_VELOCITY * 0.4)
        else:
            forward_speed = min(forward_speed, MAX_VELOCITY)
            
        # ===== COMBINE INTO DIFFERENTIAL DRIVE =====
        # Turn differential: added to one side, subtracted from other
        turn_diff = steering_error * TURN_GAIN
        
        # If person is on right (positive error), we need to turn right
        # Turn right = left wheel faster, right wheel slower
        left_vel = forward_speed + turn_diff
        right_vel = forward_speed - turn_diff
        
        # Allow spinning in place if person is to the side but at correct distance
        if abs(forward_speed) < 0.1 and abs(turn_diff) > 0.1:
            left_vel = turn_diff * 0.8
            right_vel = -turn_diff * 0.8
            
        return left_vel, right_vel
        
    def draw_ui(self, frame, results, status_info):
        """Draw UI overlays on frame"""
        # Get annotated frame with detections
        if results and len(results) > 0:
            annotated = results[0].plot()
        else:
            annotated = frame.copy()
            
        # Status bar background
        cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 90), (0, 0, 0), -1)
        
        # Status text
        status_color = (0, 255, 0) if self.following_enabled else (0, 0, 255)
        mode_text = "FOLLOWING" if self.following_enabled else "STOPPED"
        cv2.putText(annotated, mode_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                   
        # Target info
        if status_info.get('tracking'):
            target_text = f"Target ID: {status_info.get('track_id', '?')}"
            cv2.putText(annotated, target_text, (200, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                       
        # Motor velocities
        left_vel = status_info.get('left_vel', 0)
        right_vel = status_info.get('right_vel', 0)
        vel_text = f"Motors L:{left_vel:+.2f} R:{right_vel:+.2f}"
        cv2.putText(annotated, vel_text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
        # Distance status
        dist_status = status_info.get('distance_status', '')
        height_ratio = status_info.get('height_ratio', 0)
        dist_color = (0, 255, 255)
        if dist_status == 'TOO CLOSE':
            dist_color = (0, 0, 255)
        elif dist_status == 'TOO FAR':
            dist_color = (0, 165, 255)
        cv2.putText(annotated, f"Distance: {dist_status}  Height: {height_ratio:.2f}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)
                   
        # Draw center guideline
        center_x = FRAME_WIDTH // 2
        cv2.line(annotated, (center_x, 90), (center_x, FRAME_HEIGHT),
                (100, 100, 100), 1)
                
        # Draw target zone (where person should be)
        zone_left = int(FRAME_WIDTH * (0.5 - CENTER_DEADBAND))
        zone_right = int(FRAME_WIDTH * (0.5 + CENTER_DEADBAND))
        cv2.rectangle(annotated, (zone_left, 90), (zone_right, FRAME_HEIGHT),
                     (0, 255, 0), 2)
                     
        # Controls help
        cv2.putText(annotated, "Q:quit R:reset S:stop", (FRAME_WIDTH - 200, FRAME_HEIGHT - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                   
        return annotated
        
    def run(self):
        """Main control loop"""
        try:
            self.connect_motors()
            self.initialize_camera()
            
            print("\n" + "=" * 50)
            print("  HUMAN FOLLOWING ROBOT")
            print("=" * 50)
            print("Controls:")
            print("  Q - Quit")
            print("  R - Reset target (lock onto new person)")
            print("  S - Toggle following on/off")
            print("=" * 50 + "\n")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠ Failed to read frame")
                    continue
                    
                # Run detection with tracking (only persons)
                results = self.model.track(
                    frame,
                    tracker="botsort.yaml",
                    persist=True,
                    verbose=False,
                    classes=[0]  # Only detect person class
                )
                
                # Prepare status info for UI
                status_info = {
                    'tracking': False,
                    'left_vel': 0,
                    'right_vel': 0,
                    'distance_status': 'NO TARGET',
                    'height_ratio': 0
                }
                
                # Find target person
                target_info = self.find_target_person(results)
                
                if target_info and self.following_enabled:
                    center_x, height_ratio, track_id = target_info
                    self.last_detection_time = time.time()
                    
                    # Determine distance status
                    if height_ratio > TOO_CLOSE_RATIO:
                        status_info['distance_status'] = 'TOO CLOSE'
                    elif height_ratio > TARGET_BBOX_HEIGHT_RATIO + DISTANCE_DEADBAND:
                        status_info['distance_status'] = 'CLOSE'
                    elif height_ratio < TARGET_BBOX_HEIGHT_RATIO - DISTANCE_DEADBAND:
                        status_info['distance_status'] = 'TOO FAR'
                    else:
                        status_info['distance_status'] = 'GOOD'
                    
                    # Calculate and apply motor commands
                    left_vel, right_vel = self.calculate_motor_commands(center_x, height_ratio)
                    actual_left, actual_right = self.set_motor_velocities(left_vel, right_vel)
                    
                    status_info['tracking'] = True
                    status_info['track_id'] = track_id
                    status_info['left_vel'] = actual_left
                    status_info['right_vel'] = actual_right
                    status_info['height_ratio'] = height_ratio
                    
                else:
                    # No target detected
                    time_since_last = time.time() - self.last_detection_time
                    
                    if time_since_last > LOST_TRACK_TIMEOUT:
                        # Lost target for too long - stop and reset
                        self.stop_motors()
                        if self.target_id is not None:
                            print(f"⚠ Lost track of person {self.target_id}")
                            self.target_id = None
                        status_info['distance_status'] = 'LOST'
                    else:
                        # Brief loss - coast (keep current velocity)
                        status_info['distance_status'] = 'SEARCHING'
                        status_info['left_vel'] = self.current_left_vel
                        status_info['right_vel'] = self.current_right_vel
                        
                # If following disabled, ensure motors are stopped
                if not self.following_enabled:
                    self.stop_motors()
                    status_info['left_vel'] = 0
                    status_info['right_vel'] = 0
                    
                # Draw UI and display
                display_frame = self.draw_ui(frame, results, status_info)
                cv2.imshow('Human Following Robot', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    self.target_id = None
                    print("🔄 Target reset - will lock onto next person")
                elif key == ord('s'):
                    self.following_enabled = not self.following_enabled
                    if not self.following_enabled:
                        self.stop_motors()
                    status = "ENABLED" if self.following_enabled else "DISABLED"
                    print(f"Following {status}")
                    
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
        finally:
            print("\nShutting down safely...")
            self.stop_motors()
            time.sleep(0.3)
            self.shutdown_motors()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✓ Shutdown complete")


def main():
    """Entry point"""
    print("=" * 50)
    print("  Human Following Robot v1.0")
    print("  YOLOv8n + BotSort + ODrive Differential Drive")
    print("=" * 50 + "\n")
    
    follower = HumanFollower()
    follower.run()


if __name__ == "__main__":
    main()

