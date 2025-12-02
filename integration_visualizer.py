#!/usr/bin/env python3
"""
Visualization tools for vision-motor integration test data.
Generates plots showing relationships between vision inputs and motor outputs.

Usage:
    python integration_visualizer.py integration_logs/*.csv
    python integration_visualizer.py --all integration_logs/
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import argparse
from datetime import datetime
from scipy import stats, signal


def load_csv(path):
    """Load CSV data"""
    data = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value) if value != '' else np.nan)
                except (ValueError, TypeError):
                    data[key].append(np.nan)
                    
    for key in data.keys():
        data[key] = np.array(data[key])
        
    return data


def plot_timing_breakdown(data, title="Timing Breakdown", save_path=None):
    """Plot latency components over time"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data.get('time', np.arange(len(data.get('frame_time', []))))
    
    # Stacked area for latency components
    ax1 = axes[0]
    frame_time = data.get('frame_time', np.zeros_like(time)) * 1000
    detection_time = data.get('detection_time', np.zeros_like(time)) * 1000
    control_time = data.get('control_time', np.zeros_like(time)) * 1000
    
    ax1.fill_between(time, 0, frame_time, alpha=0.7, label='Frame capture')
    ax1.fill_between(time, frame_time, frame_time + detection_time, alpha=0.7, label='Detection')
    ax1.fill_between(time, frame_time + detection_time, 
                     frame_time + detection_time + control_time, alpha=0.7, label='Control')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency Components (Stacked)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Total latency
    ax2 = axes[1]
    total_latency = data.get('total_latency', np.zeros_like(time)) * 1000
    ax2.plot(time, total_latency, 'b-', linewidth=1, alpha=0.7)
    ax2.axhline(y=np.nanmean(total_latency), color='r', linestyle='--', 
               label=f'Mean: {np.nanmean(total_latency):.1f}ms')
    ax2.axhline(y=50, color='g', linestyle=':', alpha=0.5, label='50ms target')
    ax2.set_ylabel('Total Latency (ms)')
    ax2.set_title('Total Vision-to-Motor Latency')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # FPS
    ax3 = axes[2]
    fps = data.get('fps', np.zeros_like(time))
    ax3.plot(time, fps, 'g-', linewidth=1)
    ax3.axhline(y=np.nanmean(fps[fps > 0]), color='r', linestyle='--',
               label=f'Mean: {np.nanmean(fps[fps > 0]):.1f}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('FPS')
    ax3.set_title('Frame Rate')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_position_response(data, title="Position Response Mapping", save_path=None):
    """Plot steering error vs motor differential"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data.get('time', np.arange(len(data.get('steering_error', []))))
    steering_error = data.get('steering_error', np.zeros_like(time))
    center_x = data.get('center_x', np.zeros_like(time))
    left_cmd = data.get('left_cmd_smoothed', np.zeros_like(time))
    right_cmd = data.get('right_cmd_smoothed', np.zeros_like(time))
    detection_valid = data.get('detection_valid', np.ones_like(time))
    
    turn_diff = left_cmd - right_cmd
    mask = detection_valid == 1
    
    # Time series: steering error and turn differential
    ax1 = axes[0, 0]
    ax1.plot(time, steering_error, 'b-', label='Steering Error', alpha=0.7, linewidth=1)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.08, color='g', linestyle=':', alpha=0.5, label='Deadband')
    ax1.axhline(y=-0.08, color='g', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Steering Error')
    ax1.set_title('Steering Error Over Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time, turn_diff, 'r-', label='Turn Diff', alpha=0.5, linewidth=1)
    ax1_twin.set_ylabel('Turn Differential (L-R)', color='r')
    
    # Scatter: steering error → turn differential
    ax2 = axes[0, 1]
    if np.sum(mask) > 10:
        ax2.scatter(steering_error[mask], turn_diff[mask], alpha=0.3, s=20, c=time[mask], cmap='viridis')
        
        # Linear fit
        slope, intercept, r_value, _, _ = stats.linregress(steering_error[mask], turn_diff[mask])
        x_fit = np.array([np.min(steering_error[mask]), np.max(steering_error[mask])])
        ax2.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=2,
                label=f'Fit: y = {slope:.2f}x + {intercept:.3f}\nR² = {r_value**2:.3f}')
        
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Steering Error (person offset from center)')
    ax2.set_ylabel('Turn Differential (L - R)')
    ax2.set_title('Steering Error → Motor Response')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Motor commands by position zone
    ax3 = axes[1, 0]
    zones = ['LEFT', 'CENTER', 'RIGHT']
    colors = ['blue', 'green', 'red']
    
    for zone, color in zip(zones, colors):
        if zone == 'LEFT':
            zone_mask = mask & (steering_error < -0.08)
        elif zone == 'RIGHT':
            zone_mask = mask & (steering_error > 0.08)
        else:
            zone_mask = mask & (steering_error >= -0.08) & (steering_error <= 0.08)
            
        if np.sum(zone_mask) > 0:
            ax3.scatter(left_cmd[zone_mask], right_cmd[zone_mask], 
                       alpha=0.4, s=30, label=f'{zone} ({np.sum(zone_mask)} pts)', c=color)
            
    ax3.plot([-2, 2], [-2, 2], 'k--', alpha=0.3, label='Equal speed')
    ax3.set_xlabel('Left Motor Command')
    ax3.set_ylabel('Right Motor Command')
    ax3.set_title('Motor Commands by Position Zone')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Position histogram
    ax4 = axes[1, 1]
    if np.sum(mask) > 0:
        ax4.hist(center_x[mask], bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(x=0.5, color='g', linestyle='-', linewidth=2, label='Center')
        ax4.axvline(x=0.5 - 0.08, color='g', linestyle=':', alpha=0.7)
        ax4.axvline(x=0.5 + 0.08, color='g', linestyle=':', alpha=0.7, label='Deadband')
    ax4.set_xlabel('Person Position (0=left, 0.5=center, 1=right)')
    ax4.set_ylabel('Count')
    ax4.set_title('Position Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_distance_response(data, title="Distance Response Mapping", save_path=None):
    """Plot distance error vs forward speed"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data.get('time', np.arange(len(data.get('distance_error', []))))
    distance_error = data.get('distance_error', np.zeros_like(time))
    height_ratio = data.get('height_ratio', np.zeros_like(time))
    left_cmd = data.get('left_cmd_smoothed', np.zeros_like(time))
    right_cmd = data.get('right_cmd_smoothed', np.zeros_like(time))
    detection_valid = data.get('detection_valid', np.ones_like(time))
    
    forward_speed = (left_cmd + right_cmd) / 2
    mask = detection_valid == 1
    
    # Time series
    ax1 = axes[0, 0]
    ax1.plot(time, height_ratio, 'b-', label='Height Ratio', alpha=0.7, linewidth=1)
    ax1.axhline(y=0.7, color='g', linestyle='-', alpha=0.7, label='Target (0.7)')
    ax1.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='Too close')
    ax1.set_ylabel('Height Ratio (bbox/frame)')
    ax1.set_title('Person Distance Over Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time, forward_speed, 'orange', label='Forward Speed', alpha=0.5, linewidth=1)
    ax1_twin.set_ylabel('Forward Speed', color='orange')
    
    # Scatter: distance error → forward speed
    ax2 = axes[0, 1]
    if np.sum(mask) > 10:
        ax2.scatter(distance_error[mask], forward_speed[mask], alpha=0.3, s=20, c=time[mask], cmap='viridis')
        
        slope, intercept, r_value, _, _ = stats.linregress(distance_error[mask], forward_speed[mask])
        x_fit = np.array([np.min(distance_error[mask]), np.max(distance_error[mask])])
        ax2.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=2,
                label=f'Fit: y = {slope:.2f}x + {intercept:.3f}\nR² = {r_value**2:.3f}')
        
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Distance Error (+ = too far, - = too close)')
    ax2.set_ylabel('Forward Speed')
    ax2.set_title('Distance Error → Motor Response')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Motor commands by distance zone
    ax3 = axes[1, 0]
    zones = ['FAR', 'GOOD', 'CLOSE']
    colors = ['blue', 'green', 'red']
    
    for zone, color in zip(zones, colors):
        if zone == 'FAR':
            zone_mask = mask & (distance_error > 0.05)
        elif zone == 'CLOSE':
            zone_mask = mask & (distance_error < -0.05)
        else:
            zone_mask = mask & (distance_error >= -0.05) & (distance_error <= 0.05)
            
        if np.sum(zone_mask) > 0:
            mean_fwd = np.mean(forward_speed[zone_mask])
            ax3.bar(zone, mean_fwd, color=color, alpha=0.7, 
                   label=f'{zone}: {mean_fwd:+.2f}')
            ax3.errorbar(zone, mean_fwd, yerr=np.std(forward_speed[zone_mask]),
                        color='black', capsize=5)
            
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_ylabel('Average Forward Speed')
    ax3.set_title('Forward Speed by Distance Zone')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Height ratio histogram
    ax4 = axes[1, 1]
    if np.sum(mask) > 0:
        ax4.hist(height_ratio[mask], bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(x=0.7, color='g', linestyle='-', linewidth=2, label='Target')
        ax4.axvline(x=0.6, color='r', linestyle='--', linewidth=2, label='Too close')
    ax4.set_xlabel('Height Ratio (larger = closer)')
    ax4.set_ylabel('Count')
    ax4.set_title('Distance Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_motor_tracking(data, title="Motor Command Tracking", save_path=None):
    """Plot commanded vs actual motor velocities"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data.get('time', np.arange(len(data.get('left_cmd_smoothed', []))))
    left_cmd = data.get('left_cmd_smoothed', np.zeros_like(time))
    right_cmd = data.get('right_cmd_smoothed', np.zeros_like(time))
    left_actual = data.get('left_vel_actual', np.zeros_like(time))
    right_actual = data.get('right_vel_actual', np.zeros_like(time))
    
    # Left wheel time series
    ax1 = axes[0, 0]
    ax1.plot(time, left_cmd, 'b-', label='Command', alpha=0.8, linewidth=1)
    ax1.plot(time, left_actual, 'r-', label='Actual', alpha=0.6, linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (rev/s)')
    ax1.set_title('Left Motor: Command vs Actual')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right wheel time series
    ax2 = axes[0, 1]
    ax2.plot(time, right_cmd, 'b-', label='Command', alpha=0.8, linewidth=1)
    ax2.plot(time, right_actual, 'r-', label='Actual', alpha=0.6, linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (rev/s)')
    ax2.set_title('Right Motor: Command vs Actual')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Tracking error
    ax3 = axes[1, 0]
    left_error = left_cmd - left_actual
    right_error = right_cmd - right_actual
    ax3.plot(time, left_error, 'b-', label='Left Error', alpha=0.7, linewidth=1)
    ax3.plot(time, right_error, 'r-', label='Right Error', alpha=0.7, linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tracking Error (cmd - actual)')
    ax3.set_title('Motor Tracking Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Scatter: cmd vs actual
    ax4 = axes[1, 1]
    mask = (np.abs(left_cmd) > 0.05) | (np.abs(right_cmd) > 0.05)
    if np.sum(mask) > 10:
        ax4.scatter(left_cmd[mask], left_actual[mask], alpha=0.3, s=20, label='Left', c='blue')
        ax4.scatter(right_cmd[mask], right_actual[mask], alpha=0.3, s=20, label='Right', c='red')
        
        all_cmd = np.concatenate([left_cmd[mask], right_cmd[mask]])
        all_actual = np.concatenate([left_actual[mask], right_actual[mask]])
        slope, intercept, r_value, _, _ = stats.linregress(all_cmd, all_actual)
        x_fit = np.array([np.min(all_cmd), np.max(all_cmd)])
        ax4.plot(x_fit, slope * x_fit + intercept, 'g-', linewidth=2,
                label=f'Fit: k={slope:.3f}, R²={r_value**2:.3f}')
        ax4.plot(x_fit, x_fit, 'k--', alpha=0.5, label='Ideal')
        
    ax4.set_xlabel('Commanded Velocity')
    ax4.set_ylabel('Actual Velocity')
    ax4.set_title('Command vs Actual (Both Motors)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_synchronization(data, title="Vision-Motor Synchronization", save_path=None):
    """Plot cross-correlation and synchronization analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data.get('time', np.arange(len(data.get('steering_error', []))))
    steering_error = data.get('steering_error', np.zeros_like(time))
    left_cmd = data.get('left_cmd_smoothed', np.zeros_like(time))
    right_cmd = data.get('right_cmd_smoothed', np.zeros_like(time))
    detection_valid = data.get('detection_valid', np.ones_like(time))
    
    turn_diff = left_cmd - right_cmd
    mask = (detection_valid == 1) & ~np.isnan(steering_error) & ~np.isnan(turn_diff)
    
    # Normalized time series overlay
    ax1 = axes[0, 0]
    if np.sum(mask) > 20:
        steer_norm = (steering_error[mask] - np.mean(steering_error[mask])) / (np.std(steering_error[mask]) + 1e-6)
        turn_norm = (turn_diff[mask] - np.mean(turn_diff[mask])) / (np.std(turn_diff[mask]) + 1e-6)
        
        ax1.plot(time[mask], steer_norm, 'b-', label='Steering Error (norm)', alpha=0.7)
        ax1.plot(time[mask], turn_norm, 'r-', label='Turn Diff (norm)', alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Normalized Value')
        ax1.set_title('Vision Input vs Motor Output (Normalized)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Cross-correlation
    ax2 = axes[0, 1]
    if np.sum(mask) > 50:
        steer = steering_error[mask] - np.mean(steering_error[mask])
        turn = turn_diff[mask] - np.mean(turn_diff[mask])
        
        if np.std(steer) > 0.01 and np.std(turn) > 0.01:
            correlation = signal.correlate(turn, steer, mode='full')
            correlation = correlation / (len(steer) * np.std(steer) * np.std(turn))
            lags = signal.correlation_lags(len(turn), len(steer), mode='full')
            
            dt = np.mean(np.diff(time[mask]))
            lag_ms = lags * dt * 1000
            
            ax2.plot(lag_ms, correlation, 'b-', linewidth=1)
            peak_idx = np.argmax(np.abs(correlation))
            ax2.axvline(x=lag_ms[peak_idx], color='r', linestyle='--',
                       label=f'Peak lag: {lag_ms[peak_idx]:.1f}ms')
            ax2.axvline(x=0, color='k', linestyle=':', alpha=0.5)
            ax2.set_xlabel('Lag (ms)')
            ax2.set_ylabel('Cross-correlation')
            ax2.set_title('Vision→Motor Cross-correlation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([-500, 500])
    
    # Response delay analysis
    ax3 = axes[1, 0]
    # Find step changes in steering error and measure motor response time
    steer_diff = np.abs(np.diff(steering_error))
    step_indices = np.where(steer_diff > 0.1)[0]
    
    if len(step_indices) > 0:
        delays = []
        for idx in step_indices[:20]:  # First 20 steps
            if idx + 30 < len(turn_diff):
                # Find when motor response starts
                window = turn_diff[idx:idx+30]
                baseline = turn_diff[max(0, idx-5):idx]
                threshold = np.mean(np.abs(baseline)) + 2 * np.std(baseline) if len(baseline) > 0 else 0.1
                
                response_idx = np.where(np.abs(window - window[0]) > threshold)[0]
                if len(response_idx) > 0:
                    delay = response_idx[0] * np.mean(np.diff(time)) * 1000
                    delays.append(delay)
                    
        if delays:
            ax3.hist(delays, bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(x=np.mean(delays), color='r', linestyle='--',
                       label=f'Mean: {np.mean(delays):.1f}ms')
            ax3.set_xlabel('Response Delay (ms)')
            ax3.set_ylabel('Count')
            ax3.set_title('Motor Response Delay to Vision Changes')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # Phase diagram
    ax4 = axes[1, 1]
    if np.sum(mask) > 10:
        ax4.scatter(steering_error[mask], turn_diff[mask], 
                   c=time[mask], cmap='viridis', alpha=0.5, s=20)
        ax4.set_xlabel('Steering Error')
        ax4.set_ylabel('Turn Differential')
        ax4.set_title('Phase Space (color = time)')
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Time (s)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_overview(data, title="Integration Test Overview", save_path=None):
    """Single comprehensive overview plot"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data.get('time', np.arange(len(data.get('center_x', []))))
    
    # Vision: position and distance
    ax1 = axes[0]
    center_x = data.get('center_x', np.zeros_like(time))
    height_ratio = data.get('height_ratio', np.zeros_like(time))
    ax1.plot(time, center_x, 'b-', label='Position X', alpha=0.7)
    ax1.axhline(y=0.5, color='b', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Position X', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time, height_ratio, 'r-', label='Height Ratio', alpha=0.7)
    ax1_twin.axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
    ax1_twin.set_ylabel('Height Ratio', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Vision Inputs')
    ax1.grid(True, alpha=0.3)
    
    # Errors
    ax2 = axes[1]
    steering_error = data.get('steering_error', np.zeros_like(time))
    distance_error = data.get('distance_error', np.zeros_like(time))
    ax2.plot(time, steering_error, 'b-', label='Steering Error', alpha=0.7)
    ax2.plot(time, distance_error, 'r-', label='Distance Error', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Error')
    ax2.set_title('Control Errors')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Motor commands
    ax3 = axes[2]
    left_cmd = data.get('left_cmd_smoothed', np.zeros_like(time))
    right_cmd = data.get('right_cmd_smoothed', np.zeros_like(time))
    ax3.plot(time, left_cmd, 'b-', label='Left Cmd', alpha=0.7)
    ax3.plot(time, right_cmd, 'r-', label='Right Cmd', alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_ylabel('Motor Command')
    ax3.set_title('Motor Commands (Smoothed)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Actual velocities
    ax4 = axes[3]
    left_actual = data.get('left_vel_actual', np.zeros_like(time))
    right_actual = data.get('right_vel_actual', np.zeros_like(time))
    ax4.plot(time, left_actual, 'b-', label='Left Actual', alpha=0.7)
    ax4.plot(time, right_actual, 'r-', label='Right Actual', alpha=0.7)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Actual Velocity')
    ax4.set_title('Actual Motor Velocities')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def generate_all_plots(csv_path, output_dir=None):
    """Generate all integration plots"""
    print(f"\nGenerating plots for: {csv_path}")
    
    data = load_csv(csv_path)
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
        
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    plot_overview(data, f"Overview - {base_name}",
                  os.path.join(output_dir, f"{base_name}_overview.png"))
    
    plot_timing_breakdown(data, f"Timing - {base_name}",
                          os.path.join(output_dir, f"{base_name}_timing.png"))
    
    plot_position_response(data, f"Position Response - {base_name}",
                           os.path.join(output_dir, f"{base_name}_position.png"))
    
    plot_distance_response(data, f"Distance Response - {base_name}",
                           os.path.join(output_dir, f"{base_name}_distance.png"))
    
    plot_motor_tracking(data, f"Motor Tracking - {base_name}",
                        os.path.join(output_dir, f"{base_name}_motor_tracking.png"))
    
    plot_synchronization(data, f"Synchronization - {base_name}",
                         os.path.join(output_dir, f"{base_name}_sync.png"))
    
    print(f"✓ All plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize vision-motor integration data')
    parser.add_argument('csv_files', nargs='*', help='CSV files to visualize')
    parser.add_argument('--all', metavar='DIR', help='Process all CSVs in directory')
    parser.add_argument('--output-dir', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.all:
        csv_files = [os.path.join(args.all, f) for f in os.listdir(args.all) 
                     if f.endswith('.csv')]
    else:
        csv_files = args.csv_files
        
    if not csv_files:
        parser.print_help()
        print("\nExamples:")
        print("  python integration_visualizer.py integration_logs/*.csv")
        print("  python integration_visualizer.py --all integration_logs/")
        return
        
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            generate_all_plots(csv_file, args.output_dir)
        else:
            print(f"⚠ File not found: {csv_file}")


if __name__ == "__main__":
    main()
