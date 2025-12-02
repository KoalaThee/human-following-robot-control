#!/usr/bin/env python3
"""
Wheel Calibration & Metrics Test System for Human Following Robot
=================================================================
Run with wheels OFF the ground to collect data and evaluate:
- Motor response characteristics
- Left/right wheel symmetry
- Command tracking accuracy
- Latency, rise time, settling time, overshoot
- Repeatability and noise
- Power & thermal metrics

Usage:
    python wheel_calibration_test.py --test all
    python wheel_calibration_test.py --test step --analyze
    python wheel_calibration_test.py --analyze-only path/to/log.csv
"""

import odrive
from odrive.enums import *
import time
import numpy as np
import csv
import os
import argparse
from datetime import datetime
from collections import deque
import threading

# =====================
# Configuration
# =====================

# Motor direction (match your human_follower.py)
LEFT_MOTOR_DIRECTION = -1
RIGHT_MOTOR_DIRECTION = -1

# ODrive serial numbers (match your human_follower.py)
LEFT_ODRIVE_SERIAL = "325735623133"
RIGHT_ODRIVE_SERIAL = "306F388B3533"

# Pulses per revolution for your encoder (adjust based on your setup)
# ODrive reports velocity in rev/s, so we convert to RPM
PPR = 8192  # Common for AMT102 encoder, adjust if different

# Logging rate
LOG_RATE_HZ = 100  # How often to sample data
LOG_INTERVAL = 1.0 / LOG_RATE_HZ

# Test parameters
MAX_CMD = 4.0  # Maximum command velocity (rev/s) - match your MAX_VELOCITY
STEP_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]  # Fraction of MAX_CMD


class WheelCalibrationLogger:
    """High-resolution data logger for wheel calibration tests"""
    
    def __init__(self, output_dir="calibration_logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.odrv0 = None  # Left motor
        self.odrv1 = None  # Right motor
        
        # Logging state
        self.log_data = []
        self.logging_active = False
        self.log_thread = None
        self.start_time = 0
        
        # Pulse counting (if available)
        self.left_pulse_count = 0
        self.right_pulse_count = 0
        
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
        print("Configuring motors for velocity control...")
        self.odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        self.odrv1.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        time.sleep(0.5)
        
        # Configure for velocity control
        self.odrv0.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.odrv0.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        
        self.odrv1.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.odrv1.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        
        # Initialize velocities to zero
        self.odrv0.axis0.controller.input_vel = 0
        self.odrv1.axis0.controller.input_vel = 0
        
        print("✓ Motors configured")
        
    def get_motor_data(self):
        """Sample all motor data at once"""
        t = time.time() - self.start_time
        
        # Commands sent
        left_cmd = self.odrv0.axis0.controller.input_vel * LEFT_MOTOR_DIRECTION
        right_cmd = self.odrv1.axis0.controller.input_vel * RIGHT_MOTOR_DIRECTION
        
        # Measured velocities (rev/s) - convert to RPM
        left_vel = self.odrv0.axis0.encoder.vel_estimate * LEFT_MOTOR_DIRECTION
        right_vel = self.odrv1.axis0.encoder.vel_estimate * RIGHT_MOTOR_DIRECTION
        left_rpm = left_vel * 60.0
        right_rpm = right_vel * 60.0
        
        # Position (for pulse counting)
        left_pos = self.odrv0.axis0.encoder.pos_estimate
        right_pos = self.odrv1.axis0.encoder.pos_estimate
        left_pulse = int(left_pos * PPR)
        right_pulse = int(right_pos * PPR)
        
        # Electrical measurements
        try:
            bus_voltage = self.odrv0.vbus_voltage
            left_current = self.odrv0.axis0.motor.current_control.Iq_measured
            right_current = self.odrv1.axis0.motor.current_control.Iq_measured
        except:
            bus_voltage = 0
            left_current = 0
            right_current = 0
            
        # Temperature (if available)
        try:
            left_temp = self.odrv0.axis0.motor.fet_thermistor.temperature
            right_temp = self.odrv1.axis0.motor.fet_thermistor.temperature
        except:
            left_temp = 0
            right_temp = 0
            
        return {
            'time': t,
            'left_cmd': left_cmd,
            'right_cmd': right_cmd,
            'left_vel': left_vel,
            'right_vel': right_vel,
            'left_rpm': left_rpm,
            'right_rpm': right_rpm,
            'left_pulse_count': left_pulse,
            'right_pulse_count': right_pulse,
            'battery_volt': bus_voltage,
            'left_current': left_current,
            'right_current': right_current,
            'left_temp': left_temp,
            'right_temp': right_temp,
            'offset_x': 0  # Placeholder for tracking offset
        }
        
    def _log_loop(self):
        """Background logging thread"""
        while self.logging_active:
            data = self.get_motor_data()
            self.log_data.append(data)
            time.sleep(LOG_INTERVAL)
            
    def start_logging(self):
        """Start background data logging"""
        self.log_data = []
        self.start_time = time.time()
        self.logging_active = True
        self.log_thread = threading.Thread(target=self._log_loop, daemon=True)
        self.log_thread.start()
        print(f"📊 Logging started at {LOG_RATE_HZ} Hz")
        
    def stop_logging(self):
        """Stop logging and return data"""
        self.logging_active = False
        if self.log_thread:
            self.log_thread.join(timeout=1.0)
        print(f"📊 Logging stopped. {len(self.log_data)} samples collected")
        return self.log_data
        
    def save_log(self, filename_prefix="calibration"):
        """Save log data to CSV"""
        if not self.log_data:
            print("⚠ No data to save")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.csv")
        
        fieldnames = [
            'time', 'left_cmd', 'right_cmd', 
            'left_vel', 'right_vel', 'left_rpm', 'right_rpm',
            'left_pulse_count', 'right_pulse_count',
            'battery_volt', 'left_current', 'right_current',
            'left_temp', 'right_temp', 'offset_x'
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.log_data)
            
        print(f"💾 Data saved to: {filename}")
        return filename
        
    def set_velocities(self, left_vel, right_vel):
        """Set motor velocities directly (apply direction)"""
        self.odrv0.axis0.controller.input_vel = left_vel * LEFT_MOTOR_DIRECTION
        self.odrv1.axis0.controller.input_vel = right_vel * RIGHT_MOTOR_DIRECTION
        
    def stop_motors(self):
        """Stop both motors"""
        self.odrv0.axis0.controller.input_vel = 0
        self.odrv1.axis0.controller.input_vel = 0
        
    def shutdown(self):
        """Safely shutdown motors"""
        self.stop_motors()
        time.sleep(0.3)
        self.odrv0.axis0.requested_state = AXIS_STATE_IDLE
        self.odrv1.axis0.requested_state = AXIS_STATE_IDLE
        print("✓ Motors set to idle")


class TestPatterns:
    """Test pattern generators for wheel calibration"""
    
    def __init__(self, logger: WheelCalibrationLogger):
        self.logger = logger
        
    def run_idle_test(self, duration=30):
        """
        Test 1: Zero/Idle Test
        Record baseline noise at command = 0
        """
        print(f"\n{'='*50}")
        print("TEST: Idle/Zero Command ({duration}s)")
        print("="*50)
        
        self.logger.stop_motors()
        self.logger.start_logging()
        
        time.sleep(duration)
        
        self.logger.stop_logging()
        return self.logger.save_log("idle_test")
        
    def run_step_tests(self, levels=None, hold_time=3.0, reps=5):
        """
        Test 2: Single Step Tests
        Step from 0 to X, hold, then back to 0
        Run for both wheels together and separately
        """
        if levels is None:
            levels = STEP_LEVELS
            
        print(f"\n{'='*50}")
        print(f"TEST: Step Response Tests")
        print(f"Levels: {levels} × MAX_CMD ({MAX_CMD})")
        print(f"Hold time: {hold_time}s, Repetitions: {reps}")
        print("="*50)
        
        self.logger.start_logging()
        
        for level in levels:
            cmd = level * MAX_CMD
            
            for rep in range(reps):
                print(f"  Step to {cmd:.2f} rev/s (rep {rep+1}/{reps})")
                
                # Both wheels together
                self.logger.stop_motors()
                time.sleep(0.5)
                self.logger.set_velocities(cmd, cmd)
                time.sleep(hold_time)
                self.logger.stop_motors()
                time.sleep(1.0)
                
        self.logger.stop_logging()
        return self.logger.save_log("step_test")
        
    def run_single_wheel_tests(self, level=0.5, hold_time=3.0, reps=3):
        """
        Test 3: Single Wheel Excitation (Cross-coupling test)
        Command left only, then right only
        """
        print(f"\n{'='*50}")
        print(f"TEST: Single Wheel Excitation (Cross-coupling)")
        print(f"Level: {level} × MAX_CMD, Reps: {reps}")
        print("="*50)
        
        cmd = level * MAX_CMD
        self.logger.start_logging()
        
        for rep in range(reps):
            print(f"  Left wheel only (rep {rep+1}/{reps})")
            self.logger.stop_motors()
            time.sleep(0.5)
            self.logger.set_velocities(cmd, 0)  # Left only
            time.sleep(hold_time)
            self.logger.stop_motors()
            time.sleep(1.0)
            
            print(f"  Right wheel only (rep {rep+1}/{reps})")
            self.logger.stop_motors()
            time.sleep(0.5)
            self.logger.set_velocities(0, cmd)  # Right only
            time.sleep(hold_time)
            self.logger.stop_motors()
            time.sleep(1.0)
            
        self.logger.stop_logging()
        return self.logger.save_log("single_wheel_test")
        
    def run_ramp_test(self, max_level=0.8, ramp_duration=5.0, hold_duration=2.0):
        """
        Test 4: Ramp Test
        Slow linear ramp from 0 to max and back
        """
        print(f"\n{'='*50}")
        print(f"TEST: Ramp Test (0 → {max_level*MAX_CMD:.2f} → 0)")
        print(f"Ramp duration: {ramp_duration}s each way")
        print("="*50)
        
        max_cmd = max_level * MAX_CMD
        self.logger.start_logging()
        
        # Ramp up
        print("  Ramping up...")
        start_t = time.time()
        while time.time() - start_t < ramp_duration:
            progress = (time.time() - start_t) / ramp_duration
            cmd = progress * max_cmd
            self.logger.set_velocities(cmd, cmd)
            time.sleep(LOG_INTERVAL)
            
        # Hold at peak
        print("  Holding at peak...")
        time.sleep(hold_duration)
        
        # Ramp down
        print("  Ramping down...")
        start_t = time.time()
        while time.time() - start_t < ramp_duration:
            progress = (time.time() - start_t) / ramp_duration
            cmd = (1 - progress) * max_cmd
            self.logger.set_velocities(cmd, cmd)
            time.sleep(LOG_INTERVAL)
            
        self.logger.stop_motors()
        time.sleep(1.0)
        
        self.logger.stop_logging()
        return self.logger.save_log("ramp_test")
        
    def run_sine_test(self, frequencies=[0.1, 0.2, 0.5, 1.0, 2.0], 
                      amplitude=0.5, cycles_per_freq=3):
        """
        Test 5: Sine Wave Test (Bandwidth/Frequency Response)
        Oscillate at different frequencies
        """
        print(f"\n{'='*50}")
        print(f"TEST: Sine Wave / Frequency Response")
        print(f"Frequencies: {frequencies} Hz")
        print(f"Amplitude: {amplitude} × MAX_CMD")
        print("="*50)
        
        amp = amplitude * MAX_CMD
        self.logger.start_logging()
        
        for freq in frequencies:
            print(f"  Testing at {freq} Hz...")
            period = 1.0 / freq
            duration = cycles_per_freq * period
            
            start_t = time.time()
            while time.time() - start_t < duration:
                t = time.time() - start_t
                cmd = amp * np.sin(2 * np.pi * freq * t)
                self.logger.set_velocities(cmd, cmd)
                time.sleep(LOG_INTERVAL)
                
            # Brief pause between frequencies
            self.logger.stop_motors()
            time.sleep(0.5)
            
        self.logger.stop_logging()
        return self.logger.save_log("sine_test")
        
    def run_asymmetry_test(self, levels=None, hold_time=3.0, reps=5):
        """
        Test 6: Asymmetry Test
        Same command to both wheels, compare outputs
        """
        if levels is None:
            levels = STEP_LEVELS
            
        print(f"\n{'='*50}")
        print(f"TEST: Asymmetry Test (Left vs Right)")
        print(f"Levels: {levels} × MAX_CMD")
        print("="*50)
        
        self.logger.start_logging()
        
        for level in levels:
            cmd = level * MAX_CMD
            
            for rep in range(reps):
                print(f"  Both wheels at {cmd:.2f} (rep {rep+1}/{reps})")
                self.logger.stop_motors()
                time.sleep(0.5)
                self.logger.set_velocities(cmd, cmd)
                time.sleep(hold_time)
                
        self.logger.stop_motors()
        time.sleep(0.5)
        
        self.logger.stop_logging()
        return self.logger.save_log("asymmetry_test")
        
    def run_long_hold_test(self, level=0.6, duration=120):
        """
        Test 7: Long Hold Test (Thermal Drift)
        Steady command for extended period
        """
        print(f"\n{'='*50}")
        print(f"TEST: Long Hold (Thermal Drift)")
        print(f"Level: {level} × MAX_CMD for {duration}s")
        print("="*50)
        
        cmd = level * MAX_CMD
        self.logger.start_logging()
        
        self.logger.set_velocities(cmd, cmd)
        
        for i in range(0, duration, 10):
            time.sleep(10)
            print(f"  {i+10}/{duration}s elapsed...")
            
        self.logger.stop_motors()
        time.sleep(1.0)
        
        self.logger.stop_logging()
        return self.logger.save_log("long_hold_test")


class MetricsAnalyzer:
    """Compute all metrics from logged data"""
    
    def __init__(self, csv_path=None, data=None):
        if csv_path:
            self.data = self._load_csv(csv_path)
        else:
            self.data = data
            
    def _load_csv(self, path):
        """Load CSV data into numpy-friendly format"""
        data = {
            'time': [], 'left_cmd': [], 'right_cmd': [],
            'left_vel': [], 'right_vel': [], 'left_rpm': [], 'right_rpm': [],
            'left_pulse_count': [], 'right_pulse_count': [],
            'battery_volt': [], 'left_current': [], 'right_current': [],
            'left_temp': [], 'right_temp': []
        }
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in data.keys():
                    if key in row:
                        data[key].append(float(row[key]))
                        
        # Convert to numpy arrays
        for key in data.keys():
            data[key] = np.array(data[key])
            
        return data
        
    def compute_all_metrics(self):
        """Compute and return all metrics"""
        metrics = {}
        
        print("\n" + "="*60)
        print("METRICS ANALYSIS")
        print("="*60)
        
        # A. Command Tracking Error
        print("\n📊 A. Command Tracking Error")
        metrics['tracking'] = self._compute_tracking_error()
        
        # B. Linearity
        print("\n📊 B. Linearity / Gain & Bias")
        metrics['linearity'] = self._compute_linearity()
        
        # C. Latency
        print("\n📊 C. Latency (Command → Response)")
        metrics['latency'] = self._compute_latency()
        
        # D. Step Response
        print("\n📊 D. Step Response Characteristics")
        metrics['step_response'] = self._compute_step_response()
        
        # F. Symmetry
        print("\n📊 F. Left-Right Symmetry")
        metrics['symmetry'] = self._compute_symmetry()
        
        # G. Cross-coupling
        print("\n📊 G. Cross-coupling / Isolation")
        metrics['coupling'] = self._compute_cross_coupling()
        
        # H. Noise/Jitter
        print("\n📊 H. Noise / Jitter")
        metrics['noise'] = self._compute_noise()
        
        # I. Repeatability (requires multiple runs - show method)
        print("\n📊 I. Repeatability")
        metrics['repeatability'] = self._compute_repeatability()
        
        # J. Power & Thermal
        print("\n📊 J. Power & Thermal Metrics")
        metrics['power'] = self._compute_power_metrics()
        
        return metrics
        
    def _compute_tracking_error(self):
        """A. RMSE and MAE between command and measured"""
        left_cmd = self.data['left_cmd']
        right_cmd = self.data['right_cmd']
        left_meas = self.data['left_vel']
        right_meas = self.data['right_vel']
        
        # Filter out near-zero commands for meaningful error
        mask_left = np.abs(left_cmd) > 0.1
        mask_right = np.abs(right_cmd) > 0.1
        
        if np.sum(mask_left) > 10:
            rmse_left = np.sqrt(np.mean((left_cmd[mask_left] - left_meas[mask_left])**2))
            mae_left = np.mean(np.abs(left_cmd[mask_left] - left_meas[mask_left]))
        else:
            rmse_left = mae_left = np.nan
            
        if np.sum(mask_right) > 10:
            rmse_right = np.sqrt(np.mean((right_cmd[mask_right] - right_meas[mask_right])**2))
            mae_right = np.mean(np.abs(right_cmd[mask_right] - right_meas[mask_right]))
        else:
            rmse_right = mae_right = np.nan
            
        result = {
            'left_rmse': rmse_left,
            'left_mae': mae_left,
            'right_rmse': rmse_right,
            'right_mae': mae_right
        }
        
        print(f"  Left wheel:  RMSE = {rmse_left:.4f} rev/s, MAE = {mae_left:.4f} rev/s")
        print(f"  Right wheel: RMSE = {rmse_right:.4f} rev/s, MAE = {mae_right:.4f} rev/s")
        
        # Pass/fail assessment
        avg_cmd = np.mean(np.abs(np.concatenate([left_cmd[mask_left], right_cmd[mask_right]])))
        if avg_cmd > 0:
            error_pct = (rmse_left + rmse_right) / 2 / avg_cmd * 100
            print(f"  Average error: {error_pct:.1f}% of command")
            if error_pct < 5:
                print("  ✓ PASS: Error < 5%")
            elif error_pct < 10:
                print("  ⚠ WARNING: Error 5-10%")
            else:
                print("  ✗ FAIL: Error > 10%")
                
        return result
        
    def _compute_linearity(self):
        """B. Linear regression: meas = k * cmd + b"""
        from scipy import stats
        
        results = {}
        
        for side, cmd_key, meas_key in [('left', 'left_cmd', 'left_vel'), 
                                         ('right', 'right_cmd', 'right_vel')]:
            cmd = self.data[cmd_key]
            meas = self.data[meas_key]
            
            # Filter for active commands
            mask = np.abs(cmd) > 0.1
            if np.sum(mask) < 10:
                results[side] = {'k': np.nan, 'b': np.nan, 'r2': np.nan}
                continue
                
            slope, intercept, r_value, _, _ = stats.linregress(cmd[mask], meas[mask])
            r2 = r_value**2
            
            results[side] = {'k': slope, 'b': intercept, 'r2': r2}
            
            print(f"  {side.capitalize()} wheel: k = {slope:.4f}, b = {intercept:.4f}, R² = {r2:.4f}")
            
            if abs(slope - 1.0) < 0.1 and abs(intercept) < 0.1:
                print(f"    ✓ Good linearity")
            else:
                if slope < 0.9:
                    print(f"    ⚠ Under-driving (k < 1)")
                elif slope > 1.1:
                    print(f"    ⚠ Over-driving (k > 1)")
                if abs(intercept) > 0.1:
                    print(f"    ⚠ Offset bias detected")
                    
        return results
        
    def _compute_latency(self):
        """C. Cross-correlation lag between command and response"""
        from scipy import signal
        
        results = {}
        
        time_arr = self.data['time']
        dt = np.mean(np.diff(time_arr)) if len(time_arr) > 1 else LOG_INTERVAL
        
        for side, cmd_key, meas_key in [('left', 'left_cmd', 'left_vel'),
                                         ('right', 'right_cmd', 'right_vel')]:
            cmd = self.data[cmd_key]
            meas = self.data[meas_key]
            
            # Normalize for cross-correlation
            cmd_norm = cmd - np.mean(cmd)
            meas_norm = meas - np.mean(meas)
            
            if np.std(cmd_norm) < 0.01 or np.std(meas_norm) < 0.01:
                results[side] = {'latency_s': np.nan}
                continue
                
            # Cross-correlation
            correlation = signal.correlate(meas_norm, cmd_norm, mode='full')
            lags = signal.correlation_lags(len(meas_norm), len(cmd_norm), mode='full')
            
            # Find peak
            peak_idx = np.argmax(correlation)
            lag_samples = lags[peak_idx]
            latency_s = lag_samples * dt
            
            results[side] = {'latency_s': latency_s, 'lag_samples': lag_samples}
            
            print(f"  {side.capitalize()} wheel: Latency = {latency_s*1000:.1f} ms ({lag_samples} samples)")
            
            if abs(latency_s) < 0.050:
                print(f"    ✓ Low latency (<50ms)")
            elif abs(latency_s) < 0.200:
                print(f"    ⚠ Moderate latency (50-200ms)")
            else:
                print(f"    ✗ High latency (>200ms) - consider feedforward")
                
        return results
        
    def _compute_step_response(self):
        """D. Rise time, settling time, overshoot from step responses"""
        results = {}
        
        # Find step edges in command
        time_arr = self.data['time']
        dt = np.mean(np.diff(time_arr)) if len(time_arr) > 1 else LOG_INTERVAL
        
        for side, cmd_key, meas_key in [('left', 'left_cmd', 'left_vel'),
                                         ('right', 'right_cmd', 'right_vel')]:
            cmd = self.data[cmd_key]
            meas = self.data[meas_key]
            
            # Detect rising edges (0 → positive)
            cmd_diff = np.diff(cmd)
            step_indices = np.where(cmd_diff > 0.3)[0]
            
            if len(step_indices) == 0:
                results[side] = {'rise_time': np.nan, 'settling_time': np.nan, 'overshoot_pct': np.nan}
                continue
                
            rise_times = []
            settling_times = []
            overshoots = []
            
            for step_idx in step_indices:
                if step_idx + 50 >= len(meas):  # Need data after step
                    continue
                    
                # Final value (steady state) - take mean of last portion
                window_end = min(step_idx + 200, len(meas))
                final_value = np.mean(meas[window_end-20:window_end])
                initial_value = meas[step_idx]
                delta = final_value - initial_value
                
                if abs(delta) < 0.1:  # No significant step
                    continue
                    
                # Rise time: 10% to 90%
                thresh_10 = initial_value + 0.1 * delta
                thresh_90 = initial_value + 0.9 * delta
                
                t_10 = t_90 = None
                for i in range(step_idx, min(step_idx + 100, len(meas))):
                    if t_10 is None and meas[i] >= thresh_10:
                        t_10 = i
                    if t_90 is None and meas[i] >= thresh_90:
                        t_90 = i
                        break
                        
                if t_10 is not None and t_90 is not None:
                    rise_times.append((t_90 - t_10) * dt)
                    
                # Overshoot
                peak_value = np.max(meas[step_idx:window_end])
                if delta > 0:
                    overshoot_pct = max(0, (peak_value - final_value) / delta * 100)
                else:
                    overshoot_pct = 0
                overshoots.append(overshoot_pct)
                
                # Settling time (±5% of final)
                tolerance = abs(delta) * 0.05
                settled_idx = None
                for i in range(step_idx, window_end):
                    if abs(meas[i] - final_value) < tolerance:
                        # Check if it stays settled
                        if all(abs(meas[j] - final_value) < tolerance 
                               for j in range(i, min(i+10, window_end))):
                            settled_idx = i
                            break
                            
                if settled_idx is not None:
                    settling_times.append((settled_idx - step_idx) * dt)
                    
            results[side] = {
                'rise_time': np.mean(rise_times) if rise_times else np.nan,
                'settling_time': np.mean(settling_times) if settling_times else np.nan,
                'overshoot_pct': np.mean(overshoots) if overshoots else np.nan
            }
            
            print(f"  {side.capitalize()} wheel:")
            print(f"    Rise time (10%→90%): {results[side]['rise_time']*1000:.1f} ms")
            print(f"    Settling time (±5%): {results[side]['settling_time']*1000:.1f} ms")
            print(f"    Overshoot: {results[side]['overshoot_pct']:.1f}%")
            
            if results[side]['overshoot_pct'] > 20:
                print(f"    ⚠ High overshoot - consider reducing proportional gains")
                
        return results
        
    def _compute_symmetry(self):
        """F. Compare left vs right when given same command"""
        left_cmd = self.data['left_cmd']
        right_cmd = self.data['right_cmd']
        left_meas = self.data['left_vel']
        right_meas = self.data['right_vel']
        
        # Find where commands are equal and non-zero
        mask = (np.abs(left_cmd - right_cmd) < 0.01) & (np.abs(left_cmd) > 0.1)
        
        if np.sum(mask) < 10:
            print("  Insufficient data for symmetry analysis")
            return {'mean_diff': np.nan, 'std_diff': np.nan}
            
        diff = left_meas[mask] - right_meas[mask]
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        # Ratio
        mean_left = np.mean(left_meas[mask])
        mean_right = np.mean(right_meas[mask])
        ratio = mean_left / mean_right if mean_right != 0 else np.nan
        
        print(f"  Mean difference (L - R): {mean_diff:.4f} rev/s")
        print(f"  Std of difference: {std_diff:.4f} rev/s")
        print(f"  Speed ratio (L/R): {ratio:.4f}")
        
        if abs(mean_diff) < 0.05:
            print("  ✓ Good symmetry")
        else:
            if mean_diff > 0:
                print("  ⚠ Left wheel faster - will drift right on ground")
            else:
                print("  ⚠ Right wheel faster - will drift left on ground")
            print(f"  → Suggested calibration: scale {'right' if mean_diff > 0 else 'left'} by {abs(1/ratio):.3f}")
            
        return {'mean_diff': mean_diff, 'std_diff': std_diff, 'ratio': ratio}
        
    def _compute_cross_coupling(self):
        """G. Measure response of inactive wheel when only one is commanded"""
        # This requires single-wheel test data
        left_cmd = self.data['left_cmd']
        right_cmd = self.data['right_cmd']
        left_meas = self.data['left_vel']
        right_meas = self.data['right_vel']
        
        results = {}
        
        # Left commanded, right should be zero
        mask_left_only = (np.abs(left_cmd) > 0.1) & (np.abs(right_cmd) < 0.01)
        if np.sum(mask_left_only) > 10:
            coupling_l2r = np.mean(np.abs(right_meas[mask_left_only])) / np.mean(np.abs(left_meas[mask_left_only]))
            results['left_to_right'] = coupling_l2r
            print(f"  Left→Right coupling: {coupling_l2r*100:.2f}%")
        else:
            results['left_to_right'] = np.nan
            
        # Right commanded, left should be zero
        mask_right_only = (np.abs(right_cmd) > 0.1) & (np.abs(left_cmd) < 0.01)
        if np.sum(mask_right_only) > 10:
            coupling_r2l = np.mean(np.abs(left_meas[mask_right_only])) / np.mean(np.abs(right_meas[mask_right_only]))
            results['right_to_left'] = coupling_r2l
            print(f"  Right→Left coupling: {coupling_r2l*100:.2f}%")
        else:
            results['right_to_left'] = np.nan
            
        if np.isnan(results.get('left_to_right', np.nan)) and np.isnan(results.get('right_to_left', np.nan)):
            print("  ⚠ Run single-wheel test for cross-coupling analysis")
        else:
            max_coupling = max(results.get('left_to_right', 0), results.get('right_to_left', 0))
            if max_coupling < 0.05:
                print("  ✓ Good isolation (<5%)")
            else:
                print("  ⚠ Significant coupling detected - check mechanical/electrical")
                
        return results
        
    def _compute_noise(self):
        """H. RMS noise during steady state"""
        left_cmd = self.data['left_cmd']
        right_cmd = self.data['right_cmd']
        left_meas = self.data['left_vel']
        right_meas = self.data['right_vel']
        
        results = {}
        
        # Find steady-state windows (constant command for >20 samples)
        for side, cmd, meas in [('left', left_cmd, left_meas), ('right', right_cmd, right_meas)]:
            # Find constant regions
            cmd_diff = np.abs(np.diff(cmd))
            steady_mask = cmd_diff < 0.01
            
            # Find runs of steady state
            steady_runs = []
            run_start = None
            for i, is_steady in enumerate(steady_mask):
                if is_steady and run_start is None:
                    run_start = i
                elif not is_steady and run_start is not None:
                    if i - run_start > 20:  # At least 20 samples
                        steady_runs.append((run_start, i))
                    run_start = None
                    
            if not steady_runs:
                results[side] = {'noise_std': np.nan, 'noise_rms': np.nan}
                continue
                
            # Compute noise in each steady run
            noise_stds = []
            for start, end in steady_runs:
                segment = meas[start:end]
                noise_stds.append(np.std(segment))
                
            results[side] = {
                'noise_std': np.mean(noise_stds),
                'noise_rms': np.sqrt(np.mean(np.array(noise_stds)**2))
            }
            
            print(f"  {side.capitalize()} wheel: Noise STD = {results[side]['noise_std']:.4f} rev/s")
            
        avg_noise = (results.get('left', {}).get('noise_std', 0) + 
                     results.get('right', {}).get('noise_std', 0)) / 2
        if avg_noise < 0.02:
            print("  ✓ Low noise")
        elif avg_noise < 0.05:
            print("  ⚠ Moderate noise - consider filtering")
        else:
            print("  ✗ High noise - check encoder/debounce")
            
        return results
        
    def _compute_repeatability(self):
        """I. Repeatability across multiple identical steps"""
        # Find repeated steps at same command level
        left_cmd = self.data['left_cmd']
        left_meas = self.data['left_vel']
        time_arr = self.data['time']
        
        # Detect step starts
        cmd_diff = np.diff(left_cmd)
        step_indices = np.where(cmd_diff > 0.3)[0]
        
        if len(step_indices) < 3:
            print("  ⚠ Need more step repetitions for repeatability analysis")
            return {'steady_state_std': np.nan}
            
        # Group by similar command levels
        step_targets = {}
        for idx in step_indices:
            target = round(left_cmd[idx + 1], 1)  # Round to 0.1
            if target not in step_targets:
                step_targets[target] = []
            # Get steady state value (last 0.5s of step)
            window_end = min(idx + 100, len(left_meas))
            steady_val = np.mean(left_meas[window_end-20:window_end])
            step_targets[target].append(steady_val)
            
        # Compute repeatability per level
        print("  Steady-state repeatability by command level:")
        overall_stds = []
        for target, values in sorted(step_targets.items()):
            if len(values) >= 2:
                std = np.std(values)
                mean = np.mean(values)
                cv = std / mean * 100 if mean != 0 else 0
                overall_stds.append(std)
                print(f"    Cmd={target:.1f}: mean={mean:.3f}, std={std:.4f}, CV={cv:.1f}%")
                
        avg_std = np.mean(overall_stds) if overall_stds else np.nan
        print(f"  Average repeatability STD: {avg_std:.4f} rev/s")
        
        if avg_std < 0.02:
            print("  ✓ Excellent repeatability")
        elif avg_std < 0.05:
            print("  ⚠ Moderate repeatability")
        else:
            print("  ✗ Poor repeatability - check for mechanical issues")
            
        return {'steady_state_std': avg_std, 'by_level': step_targets}
        
    def _compute_power_metrics(self):
        """J. Battery, current, temperature analysis"""
        results = {}
        
        # Battery voltage
        if 'battery_volt' in self.data and np.any(self.data['battery_volt'] > 0):
            voltage = self.data['battery_volt']
            v_min = np.min(voltage)
            v_max = np.max(voltage)
            v_sag = v_max - v_min
            
            results['voltage'] = {'min': v_min, 'max': v_max, 'sag': v_sag}
            print(f"  Battery: {v_min:.2f}V - {v_max:.2f}V (sag: {v_sag:.2f}V)")
            
            if v_sag > 2.0:
                print("  ⚠ Large voltage sag - check battery capacity")
        else:
            print("  Battery voltage: N/A")
            
        # Current
        if 'left_current' in self.data:
            left_current = self.data['left_current']
            right_current = self.data['right_current']
            
            # Filter for active commands
            mask = np.abs(self.data['left_cmd']) > 0.1
            if np.any(mask):
                left_rms = np.sqrt(np.mean(left_current[mask]**2))
                right_rms = np.sqrt(np.mean(right_current[mask]**2))
                left_peak = np.max(np.abs(left_current[mask]))
                right_peak = np.max(np.abs(right_current[mask]))
                
                results['current'] = {
                    'left_rms': left_rms, 'right_rms': right_rms,
                    'left_peak': left_peak, 'right_peak': right_peak
                }
                
                print(f"  Left current:  RMS={left_rms:.2f}A, Peak={left_peak:.2f}A")
                print(f"  Right current: RMS={right_rms:.2f}A, Peak={right_peak:.2f}A")
            else:
                print("  Current: No active commands in data")
        else:
            print("  Current data: N/A")
            
        # Temperature
        if 'left_temp' in self.data and np.any(self.data['left_temp'] > 0):
            left_temp = self.data['left_temp']
            right_temp = self.data['right_temp']
            
            results['temperature'] = {
                'left_max': np.max(left_temp),
                'right_max': np.max(right_temp),
                'left_rise': np.max(left_temp) - np.min(left_temp),
                'right_rise': np.max(right_temp) - np.min(right_temp)
            }
            
            print(f"  Left temp:  Max={np.max(left_temp):.1f}°C, Rise={results['temperature']['left_rise']:.1f}°C")
            print(f"  Right temp: Max={np.max(right_temp):.1f}°C, Rise={results['temperature']['right_rise']:.1f}°C")
            
            if np.max([np.max(left_temp), np.max(right_temp)]) > 60:
                print("  ⚠ High temperature - reduce duty cycle or improve cooling")
        else:
            print("  Temperature data: N/A")
            
        return results
        
    def generate_report(self, output_path=None):
        """Generate a full metrics report"""
        metrics = self.compute_all_metrics()
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write("# Wheel Calibration Metrics Report\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                import json
                # Convert numpy types to Python types for JSON
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
                    
                f.write("```json\n")
                f.write(json.dumps(convert_numpy(metrics), indent=2))
                f.write("\n```\n")
                
            print(f"\n📄 Report saved to: {output_path}")
            
        return metrics


def run_all_tests(logger, tests):
    """Run all test patterns"""
    log_files = []
    
    try:
        print("\n" + "="*60)
        print("RUNNING ALL CALIBRATION TESTS")
        print("="*60)
        print("⚠ Ensure wheels are OFF THE GROUND!")
        print("Press Enter to continue or Ctrl+C to abort...")
        input()
        
        # 1. Idle test
        log_files.append(tests.run_idle_test(duration=10))
        
        # 2. Step tests
        log_files.append(tests.run_step_tests(hold_time=3.0, reps=3))
        
        # 3. Single wheel tests
        log_files.append(tests.run_single_wheel_tests(reps=3))
        
        # 4. Ramp test
        log_files.append(tests.run_ramp_test())
        
        # 5. Sine test
        log_files.append(tests.run_sine_test(frequencies=[0.2, 0.5, 1.0, 2.0]))
        
        # 6. Asymmetry test
        log_files.append(tests.run_asymmetry_test(reps=3))
        
        # Skip long hold by default (takes 2 min)
        # log_files.append(tests.run_long_hold_test())
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETE")
        print("="*60)
        print("Log files generated:")
        for f in log_files:
            if f:
                print(f"  • {f}")
                
        return log_files
        
    except KeyboardInterrupt:
        print("\n⚠ Tests interrupted by user")
        logger.stop_motors()
        return log_files


def main():
    parser = argparse.ArgumentParser(description='Wheel Calibration & Metrics Test System')
    parser.add_argument('--test', choices=['all', 'idle', 'step', 'single', 'ramp', 'sine', 'asymmetry', 'long'],
                       help='Test pattern to run')
    parser.add_argument('--analyze', action='store_true', help='Analyze data after test')
    parser.add_argument('--analyze-only', type=str, metavar='CSV_PATH', 
                       help='Only analyze existing CSV file')
    parser.add_argument('--output-dir', default='calibration_logs', help='Output directory for logs')
    
    args = parser.parse_args()
    
    # Analysis only mode
    if args.analyze_only:
        print(f"Analyzing: {args.analyze_only}")
        analyzer = MetricsAnalyzer(csv_path=args.analyze_only)
        report_path = args.analyze_only.replace('.csv', '_report.md')
        analyzer.generate_report(output_path=report_path)
        return
        
    # Run tests
    if args.test:
        logger = WheelCalibrationLogger(output_dir=args.output_dir)
        
        try:
            logger.connect_motors()
            tests = TestPatterns(logger)
            
            log_file = None
            
            if args.test == 'all':
                log_files = run_all_tests(logger, tests)
                if args.analyze and log_files:
                    for lf in log_files:
                        if lf:
                            analyzer = MetricsAnalyzer(csv_path=lf)
                            analyzer.generate_report(output_path=lf.replace('.csv', '_report.md'))
                            
            elif args.test == 'idle':
                log_file = tests.run_idle_test()
            elif args.test == 'step':
                log_file = tests.run_step_tests()
            elif args.test == 'single':
                log_file = tests.run_single_wheel_tests()
            elif args.test == 'ramp':
                log_file = tests.run_ramp_test()
            elif args.test == 'sine':
                log_file = tests.run_sine_test()
            elif args.test == 'asymmetry':
                log_file = tests.run_asymmetry_test()
            elif args.test == 'long':
                log_file = tests.run_long_hold_test()
                
            if args.analyze and log_file:
                analyzer = MetricsAnalyzer(csv_path=log_file)
                analyzer.generate_report(output_path=log_file.replace('.csv', '_report.md'))
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
        finally:
            logger.stop_motors()
            time.sleep(0.3)
            logger.shutdown()
            
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python wheel_calibration_test.py --test all --analyze")
        print("  python wheel_calibration_test.py --test step --analyze")
        print("  python wheel_calibration_test.py --analyze-only calibration_logs/step_test_20241201.csv")


if __name__ == "__main__":
    main()
