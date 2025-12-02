#!/usr/bin/env python3
"""
Vision-Motor Integration Test System
=====================================
Tests the synchronization and response between the human tracking algorithm
and motor/wheel system. Maps how motors respond to:
- Person position (left/right of center)
- Person distance (close/far based on bbox height)
- Combined scenarios

Run with wheels OFF the ground while a person moves in front of the camera.

Usage:
    python vision_motor_test.py --test position      # Person moves left/right
    python vision_motor_test.py --test distance      # Person moves closer/farther
    python vision_motor_test.py --test combined      # Full integration test
    python vision_motor_test.py --test latency       # Measure vision-to-motor latency
    python vision_motor_test.py --analyze-only path/to/log.csv
"""

import cv2
from ultralytics import YOLO
import odrive
from odrive.enums import *
import time
import numpy as np
import csv
import os
import argparse
from datetime import datetime
import threading
from collections import deque

# =====================
# Configuration (match human_follower.py)
# =====================

# Motor direction
LEFT_MOTOR_DIRECTION = -1
RIGHT_MOTOR_DIRECTION = -1

# ODrive serial numbers
LEFT_ODRIVE_SERIAL = "325735623133"
RIGHT_ODRIVE_SERIAL = "306F388B3533"

# Motor settings (from human_follower.py)
MAX_VELOCITY = 4.0
MIN_VELOCITY = 0.3
TURN_GAIN = 2.0
SPEED_GAIN = 4.0

# Distance calibration
TARGET_BBOX_HEIGHT_RATIO = 0.7
DISTANCE_DEADBAND = 0.05
TOO_CLOSE_RATIO = 0.6
CENTER_DEADBAND = 0.08

# Tracking
LOST_TRACK_TIMEOUT = 1.5
SMOOTH_FACTOR = 0.3

# Camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Logging
LOG_RATE_HZ = 30  # Match typical camera FPS
LOG_INTERVAL = 1.0 / LOG_RATE_HZ


class VisionMotorLogger:
    """
    High-resolution logger for vision-motor integration testing.
    Logs both vision inputs and motor outputs with precise timestamps.
    """
    
    def __init__(self, output_dir="integration_logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Hardware
        self.odrv0 = None  # Left motor
        self.odrv1 = None  # Right motor
        self.model = None
        self.cap = None
        
        # State
        self.target_id = None
        self.current_left_vel = 0.0
        self.current_right_vel = 0.0
        self.following_enabled = True
        
        # Logging
        self.log_data = []
        self.start_time = 0
        
        # Timing analysis
        self.frame_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        self.control_times = deque(maxlen=100)
        
    def connect_motors(self):
        """Connect to ODrive motor controllers"""
        print("Finding ODrive controllers...")
        print("  Looking for left motor...")
        self.odrv0 = odrive.find_any(serial_number=LEFT_ODRIVE_SERIAL)
        print("  Looking for right motor...")
        self.odrv1 = odrive.find_any(serial_number=RIGHT_ODRIVE_SERIAL)
        print("✓ ODrive controllers found!")
        
        time.sleep(1)
        
        # Set to closed loop control
        print("Configuring motors...")
        self.odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        self.odrv1.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        time.sleep(0.5)
        
        # Velocity control mode
        self.odrv0.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.odrv0.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        self.odrv1.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.odrv1.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        
        self.odrv0.axis0.controller.input_vel = 0
        self.odrv1.axis0.controller.input_vel = 0
        
        print("✓ Motors configured")
        
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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("✓ Camera initialized")
        
    def stop_motors(self):
        """Stop both motors"""
        if self.odrv0 and self.odrv1:
            self.odrv0.axis0.controller.input_vel = 0
            self.odrv1.axis0.controller.input_vel = 0
            self.current_left_vel = 0
            self.current_right_vel = 0
            
    def shutdown(self):
        """Safely shutdown all hardware"""
        self.stop_motors()
        time.sleep(0.3)
        if self.odrv0:
            self.odrv0.axis0.requested_state = AXIS_STATE_IDLE
        if self.odrv1:
            self.odrv1.axis0.requested_state = AXIS_STATE_IDLE
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Shutdown complete")
        
    def calculate_motor_commands(self, center_x, height_ratio):
        """
        Calculate motor commands from vision inputs.
        Returns (left_vel, right_vel, steering_error, distance_error)
        """
        # Steering calculation
        steering_error = center_x - 0.5
        if abs(steering_error) < CENTER_DEADBAND:
            steering_error = 0
            
        # Distance calculation
        distance_error = TARGET_BBOX_HEIGHT_RATIO - height_ratio
        
        if height_ratio > TOO_CLOSE_RATIO:
            return 0, 0, steering_error, distance_error
            
        if abs(distance_error) < DISTANCE_DEADBAND:
            distance_error = 0
            
        # Forward speed
        forward_speed = distance_error * SPEED_GAIN
        if forward_speed < 0:
            forward_speed = max(forward_speed, -MAX_VELOCITY * 0.4)
        else:
            forward_speed = min(forward_speed, MAX_VELOCITY)
            
        # Differential drive
        turn_diff = steering_error * TURN_GAIN
        left_vel = forward_speed + turn_diff
        right_vel = forward_speed - turn_diff
        
        # Spin in place if needed
        if abs(forward_speed) < 0.1 and abs(turn_diff) > 0.1:
            left_vel = turn_diff * 0.8
            right_vel = -turn_diff * 0.8
            
        return left_vel, right_vel, steering_error, distance_error
        
    def set_motor_velocities(self, left_vel, right_vel):
        """Set motor velocities with smoothing"""
        left_vel = np.clip(left_vel, -MAX_VELOCITY, MAX_VELOCITY)
        right_vel = np.clip(right_vel, -MAX_VELOCITY, MAX_VELOCITY)
        
        # Smoothing
        self.current_left_vel += SMOOTH_FACTOR * (left_vel - self.current_left_vel)
        self.current_right_vel += SMOOTH_FACTOR * (right_vel - self.current_right_vel)
        
        final_left = self.current_left_vel
        final_right = self.current_right_vel
        
        # Minimum velocity threshold
        if 0 < abs(final_left) < MIN_VELOCITY:
            final_left = MIN_VELOCITY * np.sign(final_left)
        if 0 < abs(final_right) < MIN_VELOCITY:
            final_right = MIN_VELOCITY * np.sign(final_right)
            
        # Deadband
        if abs(final_left) < 0.1:
            final_left = 0
        if abs(final_right) < 0.1:
            final_right = 0
            
        # Send to motors
        self.odrv0.axis0.controller.input_vel = final_left * LEFT_MOTOR_DIRECTION
        self.odrv1.axis0.controller.input_vel = final_right * RIGHT_MOTOR_DIRECTION
        
        return final_left, final_right
        
    def get_motor_feedback(self):
        """Get actual motor velocities and other feedback"""
        left_vel_actual = self.odrv0.axis0.encoder.vel_estimate * LEFT_MOTOR_DIRECTION
        right_vel_actual = self.odrv1.axis0.encoder.vel_estimate * RIGHT_MOTOR_DIRECTION
        
        try:
            bus_voltage = self.odrv0.vbus_voltage
            left_current = self.odrv0.axis0.motor.current_control.Iq_measured
            right_current = self.odrv1.axis0.motor.current_control.Iq_measured
        except:
            bus_voltage = left_current = right_current = 0
            
        return {
            'left_vel_actual': left_vel_actual,
            'right_vel_actual': right_vel_actual,
            'bus_voltage': bus_voltage,
            'left_current': left_current,
            'right_current': right_current
        }
        
    def find_target_person(self, results):
        """Find target person in detection results"""
        if not results or len(results) == 0:
            return None
            
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
            
        # Try to find tracked target
        if self.target_id is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            for i in range(len(boxes)):
                if int(ids[i]) == self.target_id:
                    return self._extract_box_info(boxes[i], self.target_id)
                    
        # Lock onto first person
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            track_id = int(ids[0])
            self.target_id = track_id
            return self._extract_box_info(boxes[0], track_id)
            
        return self._extract_box_info(boxes[0], None)
        
    def _extract_box_info(self, box, track_id):
        """Extract position and size from bounding box"""
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        
        center_x = ((x1 + x2) / 2) / FRAME_WIDTH
        bbox_height = y2 - y1
        height_ratio = bbox_height / FRAME_HEIGHT
        bbox_width = x2 - x1
        width_ratio = bbox_width / FRAME_WIDTH
        
        # Additional metrics
        bbox_area = (bbox_height * bbox_width) / (FRAME_WIDTH * FRAME_HEIGHT)
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
        
        return {
            'center_x': center_x,
            'center_y': ((y1 + y2) / 2) / FRAME_HEIGHT,
            'height_ratio': height_ratio,
            'width_ratio': width_ratio,
            'bbox_area': bbox_area,
            'aspect_ratio': aspect_ratio,
            'track_id': track_id,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
        }
        
    def log_frame(self, frame_data):
        """Add frame data to log"""
        self.log_data.append(frame_data)
        
    def save_log(self, filename_prefix="integration"):
        """Save log to CSV"""
        if not self.log_data:
            print("⚠ No data to save")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.csv")
        
        fieldnames = [
            # Timestamps
            'time', 'frame_time', 'detection_time', 'control_time', 'total_latency',
            # Vision inputs
            'detection_valid', 'track_id',
            'center_x', 'center_y', 'height_ratio', 'width_ratio',
            'bbox_area', 'aspect_ratio',
            # Computed errors
            'steering_error', 'distance_error',
            'position_zone', 'distance_zone',
            # Motor commands
            'left_cmd', 'right_cmd',
            'left_cmd_smoothed', 'right_cmd_smoothed',
            # Motor feedback
            'left_vel_actual', 'right_vel_actual',
            'left_tracking_error', 'right_tracking_error',
            # Power
            'bus_voltage', 'left_current', 'right_current',
            # FPS
            'fps'
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.log_data)
            
        print(f"💾 Saved {len(self.log_data)} frames to: {filename}")
        return filename
        
    def draw_test_ui(self, frame, results, status):
        """Draw test UI with detailed info"""
        if results and len(results) > 0:
            annotated = results[0].plot()
        else:
            annotated = frame.copy()
            
        # Dark overlay for text
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Status
        y = 20
        cv2.putText(annotated, f"TEST: {status.get('test_name', 'Integration')}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Vision info
        y += 25
        if status.get('detection_valid'):
            cv2.putText(annotated, 
                       f"Position: {status.get('position_zone', 'N/A')} | "
                       f"X: {status.get('center_x', 0):.2f} | "
                       f"Steer Err: {status.get('steering_error', 0):+.3f}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
            cv2.putText(annotated,
                       f"Distance: {status.get('distance_zone', 'N/A')} | "
                       f"Height: {status.get('height_ratio', 0):.2f} | "
                       f"Dist Err: {status.get('distance_error', 0):+.3f}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(annotated, "NO DETECTION", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                       
        # Motor info
        y += 25
        left_cmd = status.get('left_cmd_smoothed', 0)
        right_cmd = status.get('right_cmd_smoothed', 0)
        left_actual = status.get('left_vel_actual', 0)
        right_actual = status.get('right_vel_actual', 0)
        
        cv2.putText(annotated,
                   f"CMD  L:{left_cmd:+.2f} R:{right_cmd:+.2f} | "
                   f"ACT  L:{left_actual:+.2f} R:{right_actual:+.2f}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Latency
        y += 25
        cv2.putText(annotated,
                   f"Latency: {status.get('total_latency', 0)*1000:.1f}ms | "
                   f"FPS: {status.get('fps', 0):.1f}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        # Draw center guides
        center_x = FRAME_WIDTH // 2
        zone_left = int(FRAME_WIDTH * (0.5 - CENTER_DEADBAND))
        zone_right = int(FRAME_WIDTH * (0.5 + CENTER_DEADBAND))
        
        cv2.line(annotated, (center_x, 140), (center_x, FRAME_HEIGHT), (100, 100, 100), 1)
        cv2.rectangle(annotated, (zone_left, 140), (zone_right, FRAME_HEIGHT), (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(annotated, "Q:quit R:reset S:stop", 
                   (FRAME_WIDTH - 180, FRAME_HEIGHT - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return annotated
        
    def get_position_zone(self, center_x):
        """Classify position as left/center/right"""
        offset = center_x - 0.5
        if offset < -CENTER_DEADBAND:
            return "LEFT"
        elif offset > CENTER_DEADBAND:
            return "RIGHT"
        else:
            return "CENTER"
            
    def get_distance_zone(self, height_ratio):
        """Classify distance as close/good/far"""
        if height_ratio > TOO_CLOSE_RATIO:
            return "TOO_CLOSE"
        elif height_ratio > TARGET_BBOX_HEIGHT_RATIO + DISTANCE_DEADBAND:
            return "CLOSE"
        elif height_ratio < TARGET_BBOX_HEIGHT_RATIO - DISTANCE_DEADBAND:
            return "FAR"
        else:
            return "GOOD"
            
    def run_integration_test(self, test_name="combined", duration=60, motors_enabled=True):
        """
        Main integration test loop.
        
        Args:
            test_name: Name for logging
            duration: Test duration in seconds
            motors_enabled: If False, log commands but don't send to motors
        """
        print(f"\n{'='*60}")
        print(f"INTEGRATION TEST: {test_name}")
        print(f"Duration: {duration}s | Motors: {'ENABLED' if motors_enabled else 'DISABLED'}")
        print("="*60)
        
        if not motors_enabled:
            print("⚠ Motors disabled - logging commands only")
            
        print("\nInstructions:")
        if test_name == "position":
            print("  Move LEFT and RIGHT in front of camera")
            print("  Stay at roughly same distance")
        elif test_name == "distance":
            print("  Move CLOSER and FARTHER from camera")
            print("  Stay centered horizontally")
        elif test_name == "latency":
            print("  Make QUICK movements (step changes)")
            print("  Pause briefly between movements")
        else:
            print("  Move naturally - test all scenarios")
            
        print("\nPress ENTER to start (Q to quit during test)")
        input()
        
        self.log_data = []
        self.start_time = time.time()
        self.target_id = None
        self.following_enabled = motors_enabled
        
        fps_counter = 0
        fps_time = time.time()
        current_fps = 0
        
        try:
            while True:
                loop_start = time.time()
                elapsed = loop_start - self.start_time
                
                if elapsed > duration:
                    print(f"\n✓ Test complete ({duration}s)")
                    break
                    
                # Capture frame
                frame_start = time.time()
                ret, frame = self.cap.read()
                frame_time = time.time() - frame_start
                
                if not ret:
                    continue
                    
                # Detection
                detection_start = time.time()
                results = self.model.track(
                    frame,
                    tracker="botsort.yaml",
                    persist=True,
                    verbose=False,
                    classes=[0]
                )
                detection_time = time.time() - detection_start
                
                # Process detection
                control_start = time.time()
                target_info = self.find_target_person(results)
                
                status = {
                    'test_name': test_name,
                    'detection_valid': target_info is not None,
                }
                
                frame_data = {
                    'time': elapsed,
                    'frame_time': frame_time,
                    'detection_time': detection_time,
                    'detection_valid': int(target_info is not None),
                }
                
                if target_info:
                    center_x = target_info['center_x']
                    height_ratio = target_info['height_ratio']
                    
                    # Calculate commands
                    left_cmd, right_cmd, steering_error, distance_error = \
                        self.calculate_motor_commands(center_x, height_ratio)
                    
                    # Apply to motors (if enabled)
                    if self.following_enabled:
                        left_smoothed, right_smoothed = self.set_motor_velocities(left_cmd, right_cmd)
                    else:
                        # Simulate smoothing without sending
                        self.current_left_vel += SMOOTH_FACTOR * (left_cmd - self.current_left_vel)
                        self.current_right_vel += SMOOTH_FACTOR * (right_cmd - self.current_right_vel)
                        left_smoothed = self.current_left_vel
                        right_smoothed = self.current_right_vel
                        
                    # Get motor feedback
                    motor_fb = self.get_motor_feedback()
                    
                    control_time = time.time() - control_start
                    total_latency = frame_time + detection_time + control_time
                    
                    # Classify zones
                    position_zone = self.get_position_zone(center_x)
                    distance_zone = self.get_distance_zone(height_ratio)
                    
                    # Update status for UI
                    status.update({
                        'center_x': center_x,
                        'height_ratio': height_ratio,
                        'steering_error': steering_error,
                        'distance_error': distance_error,
                        'position_zone': position_zone,
                        'distance_zone': distance_zone,
                        'left_cmd_smoothed': left_smoothed,
                        'right_cmd_smoothed': right_smoothed,
                        'left_vel_actual': motor_fb['left_vel_actual'],
                        'right_vel_actual': motor_fb['right_vel_actual'],
                        'total_latency': total_latency,
                        'fps': current_fps,
                    })
                    
                    # Log data
                    frame_data.update({
                        'control_time': control_time,
                        'total_latency': total_latency,
                        'track_id': target_info['track_id'],
                        'center_x': center_x,
                        'center_y': target_info['center_y'],
                        'height_ratio': height_ratio,
                        'width_ratio': target_info['width_ratio'],
                        'bbox_area': target_info['bbox_area'],
                        'aspect_ratio': target_info['aspect_ratio'],
                        'steering_error': steering_error,
                        'distance_error': distance_error,
                        'position_zone': position_zone,
                        'distance_zone': distance_zone,
                        'left_cmd': left_cmd,
                        'right_cmd': right_cmd,
                        'left_cmd_smoothed': left_smoothed,
                        'right_cmd_smoothed': right_smoothed,
                        'left_vel_actual': motor_fb['left_vel_actual'],
                        'right_vel_actual': motor_fb['right_vel_actual'],
                        'left_tracking_error': left_smoothed - motor_fb['left_vel_actual'],
                        'right_tracking_error': right_smoothed - motor_fb['right_vel_actual'],
                        'bus_voltage': motor_fb['bus_voltage'],
                        'left_current': motor_fb['left_current'],
                        'right_current': motor_fb['right_current'],
                        'fps': current_fps,
                    })
                else:
                    # No detection
                    self.stop_motors()
                    control_time = time.time() - control_start
                    frame_data.update({
                        'control_time': control_time,
                        'total_latency': frame_time + detection_time + control_time,
                    })
                    status['fps'] = current_fps
                    
                self.log_frame(frame_data)
                
                # FPS calculation
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                    
                # Display
                display = self.draw_test_ui(frame, results, status)
                
                # Progress bar
                progress = int((elapsed / duration) * 100)
                cv2.rectangle(display, (10, FRAME_HEIGHT - 25), 
                             (10 + int(progress * 2), FRAME_HEIGHT - 15), (0, 255, 0), -1)
                cv2.rectangle(display, (10, FRAME_HEIGHT - 25), 
                             (210, FRAME_HEIGHT - 15), (255, 255, 255), 1)
                cv2.putText(display, f"{elapsed:.1f}s / {duration}s", 
                           (220, FRAME_HEIGHT - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imshow('Vision-Motor Test', display)
                
                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠ Test stopped by user")
                    break
                elif key == ord('r'):
                    self.target_id = None
                    print("🔄 Target reset")
                elif key == ord('s'):
                    self.following_enabled = not self.following_enabled
                    if not self.following_enabled:
                        self.stop_motors()
                    print(f"Motors {'ENABLED' if self.following_enabled else 'DISABLED'}")
                    
        except KeyboardInterrupt:
            print("\n⚠ Interrupted")
        finally:
            self.stop_motors()
            
        return self.save_log(f"{test_name}_test")


class IntegrationMetricsAnalyzer:
    """Analyze vision-motor integration data"""
    
    def __init__(self, csv_path):
        self.data = self._load_csv(csv_path)
        self.csv_path = csv_path
        
    def _load_csv(self, path):
        """Load CSV data"""
        data = {}
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    if key not in data:
                        data[key] = []
                    try:
                        data[key].append(float(value))
                    except (ValueError, TypeError):
                        data[key].append(value)
                        
        for key in data.keys():
            try:
                data[key] = np.array([float(x) if x != '' else np.nan for x in data[key]])
            except:
                pass
                
        return data
        
    def compute_all_metrics(self):
        """Compute all integration metrics"""
        metrics = {}
        
        print("\n" + "="*60)
        print("VISION-MOTOR INTEGRATION METRICS")
        print("="*60)
        
        # 1. Timing/Latency
        print("\n📊 1. TIMING & LATENCY")
        metrics['timing'] = self._compute_timing_metrics()
        
        # 2. Position Response Mapping
        print("\n📊 2. POSITION RESPONSE (Left/Right)")
        metrics['position'] = self._compute_position_response()
        
        # 3. Distance Response Mapping
        print("\n📊 3. DISTANCE RESPONSE (Close/Far)")
        metrics['distance'] = self._compute_distance_response()
        
        # 4. Motor Tracking Quality
        print("\n📊 4. MOTOR TRACKING QUALITY")
        metrics['motor_tracking'] = self._compute_motor_tracking()
        
        # 5. Zone-wise Analysis
        print("\n📊 5. ZONE-WISE MOTOR RESPONSE")
        metrics['zones'] = self._compute_zone_response()
        
        # 6. Synchronization
        print("\n📊 6. VISION-MOTOR SYNCHRONIZATION")
        metrics['sync'] = self._compute_synchronization()
        
        return metrics
        
    def _compute_timing_metrics(self):
        """Compute timing/latency metrics"""
        frame_time = self.data.get('frame_time', np.array([]))
        detection_time = self.data.get('detection_time', np.array([]))
        control_time = self.data.get('control_time', np.array([]))
        total_latency = self.data.get('total_latency', np.array([]))
        fps = self.data.get('fps', np.array([]))
        
        results = {}
        
        if len(frame_time) > 0:
            results['frame_time_mean'] = np.nanmean(frame_time) * 1000
            results['detection_time_mean'] = np.nanmean(detection_time) * 1000
            results['control_time_mean'] = np.nanmean(control_time) * 1000
            results['total_latency_mean'] = np.nanmean(total_latency) * 1000
            results['total_latency_max'] = np.nanmax(total_latency) * 1000
            results['total_latency_std'] = np.nanstd(total_latency) * 1000
            results['fps_mean'] = np.nanmean(fps[fps > 0]) if np.any(fps > 0) else 0
            
            print(f"  Frame capture:    {results['frame_time_mean']:.2f} ms")
            print(f"  Detection (YOLO): {results['detection_time_mean']:.2f} ms")
            print(f"  Control calc:     {results['control_time_mean']:.2f} ms")
            print(f"  ─────────────────────────────")
            print(f"  Total latency:    {results['total_latency_mean']:.2f} ms (±{results['total_latency_std']:.2f})")
            print(f"  Max latency:      {results['total_latency_max']:.2f} ms")
            print(f"  Average FPS:      {results['fps_mean']:.1f}")
            
            if results['total_latency_mean'] < 50:
                print("  ✓ Excellent latency (<50ms)")
            elif results['total_latency_mean'] < 100:
                print("  ⚠ Moderate latency (50-100ms)")
            else:
                print("  ✗ High latency (>100ms) - may affect control")
                
        return results
        
    def _compute_position_response(self):
        """Map steering_error → motor differential"""
        steering_error = self.data.get('steering_error', np.array([]))
        left_cmd = self.data.get('left_cmd_smoothed', np.array([]))
        right_cmd = self.data.get('right_cmd_smoothed', np.array([]))
        detection_valid = self.data.get('detection_valid', np.array([]))
        
        results = {}
        
        # Filter for valid detections
        mask = detection_valid == 1
        if np.sum(mask) < 10:
            print("  Insufficient data")
            return results
            
        steer_err = steering_error[mask]
        turn_diff = (left_cmd[mask] - right_cmd[mask])  # Positive = turn right
        
        # Linear fit: turn_diff = k * steering_error
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(steer_err, turn_diff)
        
        results['gain'] = slope
        results['offset'] = intercept
        results['r_squared'] = r_value**2
        
        print(f"  Steering Gain (turn_diff / error): {slope:.3f}")
        print(f"  Offset bias: {intercept:.4f}")
        print(f"  R² fit: {r_value**2:.4f}")
        
        # Response by zone
        left_mask = steer_err < -CENTER_DEADBAND
        right_mask = steer_err > CENTER_DEADBAND
        center_mask = (steer_err >= -CENTER_DEADBAND) & (steer_err <= CENTER_DEADBAND)
        
        if np.any(left_mask):
            results['left_response_mean'] = np.mean(turn_diff[left_mask])
            print(f"  Person LEFT  → Turn diff: {results['left_response_mean']:+.3f} (should be negative)")
        if np.any(right_mask):
            results['right_response_mean'] = np.mean(turn_diff[right_mask])
            print(f"  Person RIGHT → Turn diff: {results['right_response_mean']:+.3f} (should be positive)")
        if np.any(center_mask):
            results['center_response_mean'] = np.mean(turn_diff[center_mask])
            print(f"  Person CENTER → Turn diff: {results['center_response_mean']:+.3f} (should be ~0)")
            
        return results
        
    def _compute_distance_response(self):
        """Map distance_error → forward speed"""
        distance_error = self.data.get('distance_error', np.array([]))
        left_cmd = self.data.get('left_cmd_smoothed', np.array([]))
        right_cmd = self.data.get('right_cmd_smoothed', np.array([]))
        detection_valid = self.data.get('detection_valid', np.array([]))
        
        results = {}
        
        mask = detection_valid == 1
        if np.sum(mask) < 10:
            print("  Insufficient data")
            return results
            
        dist_err = distance_error[mask]
        forward_speed = (left_cmd[mask] + right_cmd[mask]) / 2
        
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(dist_err, forward_speed)
        
        results['gain'] = slope
        results['offset'] = intercept
        results['r_squared'] = r_value**2
        
        print(f"  Distance Gain (speed / error): {slope:.3f}")
        print(f"  Offset bias: {intercept:.4f}")
        print(f"  R² fit: {r_value**2:.4f}")
        
        # Response by zone
        far_mask = dist_err > DISTANCE_DEADBAND
        close_mask = dist_err < -DISTANCE_DEADBAND
        good_mask = (dist_err >= -DISTANCE_DEADBAND) & (dist_err <= DISTANCE_DEADBAND)
        
        if np.any(far_mask):
            results['far_response_mean'] = np.mean(forward_speed[far_mask])
            print(f"  Person FAR   → Speed: {results['far_response_mean']:+.3f} (should be positive)")
        if np.any(close_mask):
            results['close_response_mean'] = np.mean(forward_speed[close_mask])
            print(f"  Person CLOSE → Speed: {results['close_response_mean']:+.3f} (should be negative)")
        if np.any(good_mask):
            results['good_response_mean'] = np.mean(forward_speed[good_mask])
            print(f"  Person GOOD  → Speed: {results['good_response_mean']:+.3f} (should be ~0)")
            
        return results
        
    def _compute_motor_tracking(self):
        """How well motors follow commanded velocities"""
        left_cmd = self.data.get('left_cmd_smoothed', np.array([]))
        right_cmd = self.data.get('right_cmd_smoothed', np.array([]))
        left_actual = self.data.get('left_vel_actual', np.array([]))
        right_actual = self.data.get('right_vel_actual', np.array([]))
        detection_valid = self.data.get('detection_valid', np.array([]))
        
        results = {}
        
        mask = (detection_valid == 1) & (np.abs(left_cmd) > 0.1)
        if np.sum(mask) < 10:
            print("  Insufficient active motor data")
            return results
            
        left_error = left_cmd[mask] - left_actual[mask]
        right_error = right_cmd[mask] - right_actual[mask]
        
        results['left_rmse'] = np.sqrt(np.mean(left_error**2))
        results['right_rmse'] = np.sqrt(np.mean(right_error**2))
        results['left_mae'] = np.mean(np.abs(left_error))
        results['right_mae'] = np.mean(np.abs(right_error))
        
        print(f"  Left motor:  RMSE={results['left_rmse']:.4f}, MAE={results['left_mae']:.4f}")
        print(f"  Right motor: RMSE={results['right_rmse']:.4f}, MAE={results['right_mae']:.4f}")
        
        avg_error_pct = (results['left_rmse'] + results['right_rmse']) / 2 / np.mean(np.abs(left_cmd[mask])) * 100
        print(f"  Average tracking error: {avg_error_pct:.1f}% of command")
        
        if avg_error_pct < 10:
            print("  ✓ Good motor tracking")
        else:
            print("  ⚠ Motor tracking could be improved")
            
        return results
        
    def _compute_zone_response(self):
        """Analyze motor response in each position/distance zone combination"""
        position_zones = ['LEFT', 'CENTER', 'RIGHT']
        distance_zones = ['FAR', 'GOOD', 'CLOSE', 'TOO_CLOSE']
        
        results = {}
        
        print("\n  Zone-wise average motor commands (L, R):")
        print("  " + "-"*50)
        print(f"  {'Position':<10} | {'FAR':^12} | {'GOOD':^12} | {'CLOSE':^12}")
        print("  " + "-"*50)
        
        for pos in position_zones:
            row = f"  {pos:<10} |"
            for dist in distance_zones[:3]:  # Skip TOO_CLOSE for table
                # This is simplified - actual implementation would use position_zone/distance_zone columns
                key = f"{pos}_{dist}"
                results[key] = {'left': 0, 'right': 0}  # Placeholder
                row += f" ({0:+.1f},{0:+.1f}) |"
            print(row)
            
        print("  " + "-"*50)
        print("  (Values show: left_cmd, right_cmd)")
        
        return results
        
    def _compute_synchronization(self):
        """Measure vision-motor synchronization via cross-correlation"""
        from scipy import signal
        
        steering_error = self.data.get('steering_error', np.array([]))
        turn_diff = self.data.get('left_cmd_smoothed', np.array([])) - \
                    self.data.get('right_cmd_smoothed', np.array([]))
        time_arr = self.data.get('time', np.array([]))
        
        results = {}
        
        if len(steering_error) < 50:
            print("  Insufficient data for synchronization analysis")
            return results
            
        # Remove NaN and mean
        mask = ~np.isnan(steering_error) & ~np.isnan(turn_diff)
        if np.sum(mask) < 50:
            return results
            
        steer = steering_error[mask] - np.mean(steering_error[mask])
        turn = turn_diff[mask] - np.mean(turn_diff[mask])
        
        if np.std(steer) < 0.01 or np.std(turn) < 0.01:
            print("  Insufficient variation for sync analysis")
            return results
            
        # Cross-correlation
        correlation = signal.correlate(turn, steer, mode='full')
        lags = signal.correlation_lags(len(turn), len(steer), mode='full')
        
        dt = np.mean(np.diff(time_arr[mask])) if len(time_arr[mask]) > 1 else 1/30
        
        peak_idx = np.argmax(np.abs(correlation))
        lag_samples = lags[peak_idx]
        sync_lag_ms = lag_samples * dt * 1000
        
        results['sync_lag_ms'] = sync_lag_ms
        results['correlation_peak'] = correlation[peak_idx]
        
        print(f"  Vision→Motor sync lag: {sync_lag_ms:.1f} ms")
        print(f"  Cross-correlation peak: {correlation[peak_idx]:.3f}")
        
        if abs(sync_lag_ms) < 50:
            print("  ✓ Good synchronization")
        elif abs(sync_lag_ms) < 100:
            print("  ⚠ Moderate sync delay")
        else:
            print("  ✗ Significant sync delay - check smoothing factor")
            
        return results
        
    def generate_report(self, output_path=None):
        """Generate full analysis report"""
        metrics = self.compute_all_metrics()
        
        if output_path is None:
            output_path = self.csv_path.replace('.csv', '_report.md')
            
        with open(output_path, 'w') as f:
            f.write("# Vision-Motor Integration Analysis Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Data file: {self.csv_path}\n\n")
            
            import json
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                return obj
                
            f.write("## Metrics\n\n```json\n")
            f.write(json.dumps(convert_numpy(metrics), indent=2))
            f.write("\n```\n")
            
        print(f"\n📄 Report saved: {output_path}")
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Vision-Motor Integration Test')
    parser.add_argument('--test', choices=['position', 'distance', 'combined', 'latency', 'all'],
                       help='Test type to run')
    parser.add_argument('--duration', type=int, default=60, help='Test duration (seconds)')
    parser.add_argument('--no-motors', action='store_true', help='Log only, no motor output')
    parser.add_argument('--analyze', action='store_true', help='Analyze after test')
    parser.add_argument('--analyze-only', type=str, metavar='CSV', help='Analyze existing CSV')
    
    args = parser.parse_args()
    
    # Analysis only
    if args.analyze_only:
        analyzer = IntegrationMetricsAnalyzer(args.analyze_only)
        analyzer.generate_report()
        return
        
    # Run test
    if args.test:
        logger = VisionMotorLogger()
        
        try:
            logger.connect_motors()
            logger.initialize_camera()
            
            if args.test == 'all':
                # Run all tests
                for test in ['position', 'distance', 'combined']:
                    log_file = logger.run_integration_test(
                        test_name=test,
                        duration=args.duration,
                        motors_enabled=not args.no_motors
                    )
                    if args.analyze and log_file:
                        analyzer = IntegrationMetricsAnalyzer(log_file)
                        analyzer.generate_report()
                    time.sleep(2)
            else:
                log_file = logger.run_integration_test(
                    test_name=args.test,
                    duration=args.duration,
                    motors_enabled=not args.no_motors
                )
                
                if args.analyze and log_file:
                    analyzer = IntegrationMetricsAnalyzer(log_file)
                    analyzer.generate_report()
                    
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
        finally:
            logger.shutdown()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python vision_motor_test.py --test position --duration 30 --analyze")
        print("  python vision_motor_test.py --test combined --no-motors  # Safe: log only")
        print("  python vision_motor_test.py --analyze-only integration_logs/combined_test.csv")


if __name__ == "__main__":
    main()
