#!/usr/bin/env python3
"""
Visualization tools for wheel calibration data
Generates plots for all metrics from CSV logs

Usage:
    python calibration_visualizer.py calibration_logs/step_test_*.csv
    python calibration_visualizer.py --all calibration_logs/
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import argparse
from datetime import datetime
from scipy import signal, stats


def load_csv(path):
    """Load CSV data into numpy arrays"""
    data = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(0)
                    
    for key in data.keys():
        data[key] = np.array(data[key])
        
    return data


def plot_overview(data, title="Calibration Overview", save_path=None):
    """Plot overall command vs measured velocity"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data['time']
    
    # Left wheel
    ax1 = axes[0]
    ax1.plot(time, data['left_cmd'], 'b-', label='Command', alpha=0.8, linewidth=1)
    ax1.plot(time, data['left_vel'], 'r-', label='Measured', alpha=0.7, linewidth=1)
    ax1.set_ylabel('Left Wheel (rev/s)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Left Motor')
    
    # Right wheel
    ax2 = axes[1]
    ax2.plot(time, data['right_cmd'], 'b-', label='Command', alpha=0.8, linewidth=1)
    ax2.plot(time, data['right_vel'], 'r-', label='Measured', alpha=0.7, linewidth=1)
    ax2.set_ylabel('Right Wheel (rev/s)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Right Motor')
    
    # Error
    ax3 = axes[2]
    left_error = data['left_cmd'] - data['left_vel']
    right_error = data['right_cmd'] - data['right_vel']
    ax3.plot(time, left_error, 'g-', label='Left Error', alpha=0.7, linewidth=1)
    ax3.plot(time, right_error, 'm-', label='Right Error', alpha=0.7, linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Tracking Error (rev/s)')
    ax3.set_xlabel('Time (s)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Command Tracking Error')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_linearity(data, title="Linearity Analysis", save_path=None):
    """Plot command vs measured with linear fit"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for ax, side, cmd_key, meas_key in [
        (axes[0], 'Left', 'left_cmd', 'left_vel'),
        (axes[1], 'Right', 'right_cmd', 'right_vel')
    ]:
        cmd = data[cmd_key]
        meas = data[meas_key]
        
        # Filter for meaningful data
        mask = np.abs(cmd) > 0.1
        
        if np.sum(mask) > 10:
            # Scatter plot
            ax.scatter(cmd[mask], meas[mask], alpha=0.3, s=10, label='Data')
            
            # Linear fit
            slope, intercept, r_value, _, _ = stats.linregress(cmd[mask], meas[mask])
            x_fit = np.array([np.min(cmd[mask]), np.max(cmd[mask])])
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
                   label=f'Fit: y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.4f}')
            
            # Ideal line
            ax.plot(x_fit, x_fit, 'k--', alpha=0.5, label='Ideal (y=x)')
            
        ax.set_xlabel('Command (rev/s)')
        ax.set_ylabel('Measured (rev/s)')
        ax.set_title(f'{side} Wheel')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_step_response(data, title="Step Response Analysis", save_path=None):
    """Plot step responses with annotations"""
    time = data['time']
    cmd = data['left_cmd']
    meas = data['left_vel']
    
    # Find step edges
    cmd_diff = np.diff(cmd)
    step_indices = np.where(cmd_diff > 0.3)[0]
    
    if len(step_indices) == 0:
        print("  No step transitions found")
        return
        
    # Plot first few steps in detail
    n_steps = min(4, len(step_indices))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i, step_idx in enumerate(step_indices[:n_steps]):
        ax = axes[i // 2, i % 2]
        
        # Window around step
        start = max(0, step_idx - 10)
        end = min(len(time), step_idx + 100)
        
        t_window = time[start:end] - time[step_idx]
        cmd_window = cmd[start:end]
        meas_window = meas[start:end]
        
        ax.plot(t_window, cmd_window, 'b-', linewidth=2, label='Command')
        ax.plot(t_window, meas_window, 'r-', linewidth=2, label='Measured')
        
        # Calculate metrics
        final_cmd = cmd[min(step_idx + 50, len(cmd)-1)]
        final_meas = np.mean(meas[end-10:end])
        initial_meas = meas[step_idx]
        
        # Rise time markers
        delta = final_meas - initial_meas
        if delta > 0.1:
            thresh_10 = initial_meas + 0.1 * delta
            thresh_90 = initial_meas + 0.9 * delta
            
            ax.axhline(y=thresh_10, color='g', linestyle=':', alpha=0.5, label='10%')
            ax.axhline(y=thresh_90, color='g', linestyle=':', alpha=0.5, label='90%')
            ax.axhline(y=final_meas, color='orange', linestyle='--', alpha=0.5, label='Final')
            
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time from step (s)')
        ax.set_ylabel('Velocity (rev/s)')
        ax.set_title(f'Step {i+1}: 0 → {final_cmd:.2f}')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_symmetry(data, title="Left-Right Symmetry", save_path=None):
    """Plot left vs right wheel comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data['time']
    left_meas = data['left_vel']
    right_meas = data['right_vel']
    left_cmd = data['left_cmd']
    right_cmd = data['right_cmd']
    
    # Filter for same commands
    mask = (np.abs(left_cmd - right_cmd) < 0.01) & (np.abs(left_cmd) > 0.1)
    
    # Time series comparison
    ax1 = axes[0, 0]
    ax1.plot(time, left_meas, 'b-', label='Left', alpha=0.7)
    ax1.plot(time, right_meas, 'r-', label='Right', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (rev/s)')
    ax1.set_title('Time Series Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Difference
    ax2 = axes[0, 1]
    diff = left_meas - right_meas
    ax2.plot(time, diff, 'g-', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.axhline(y=np.mean(diff[mask]) if np.any(mask) else 0, color='r', linestyle='-', 
               label=f'Mean diff: {np.mean(diff[mask]) if np.any(mask) else 0:.4f}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('L - R (rev/s)')
    ax2.set_title('Left - Right Difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Scatter: Left vs Right
    ax3 = axes[1, 0]
    if np.any(mask):
        ax3.scatter(left_meas[mask], right_meas[mask], alpha=0.4, s=20)
        # Fit line
        slope, intercept, r_value, _, _ = stats.linregress(left_meas[mask], right_meas[mask])
        x_fit = np.array([np.min(left_meas[mask]), np.max(left_meas[mask])])
        ax3.plot(x_fit, slope * x_fit + intercept, 'r-', 
                label=f'Fit: slope={slope:.3f}, R²={r_value**2:.3f}')
        ax3.plot(x_fit, x_fit, 'k--', alpha=0.5, label='Ideal (y=x)')
    ax3.set_xlabel('Left Velocity (rev/s)')
    ax3.set_ylabel('Right Velocity (rev/s)')
    ax3.set_title('Left vs Right Correlation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Histogram of difference
    ax4 = axes[1, 1]
    if np.any(mask):
        ax4.hist(diff[mask], bins=50, alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax4.axvline(x=np.mean(diff[mask]), color='r', linestyle='-',
                   label=f'Mean: {np.mean(diff[mask]):.4f}')
        ax4.axvline(x=np.mean(diff[mask]) + np.std(diff[mask]), color='r', linestyle=':', alpha=0.5)
        ax4.axvline(x=np.mean(diff[mask]) - np.std(diff[mask]), color='r', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Left - Right (rev/s)')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Difference')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_noise_analysis(data, title="Noise Analysis", save_path=None):
    """Plot noise characteristics during steady state"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data['time']
    
    for col, (side, meas_key, cmd_key) in enumerate([
        ('Left', 'left_vel', 'left_cmd'),
        ('Right', 'right_vel', 'right_cmd')
    ]):
        meas = data[meas_key]
        cmd = data[cmd_key]
        
        # Find steady regions
        cmd_diff = np.abs(np.diff(cmd))
        steady_mask = np.concatenate([[True], cmd_diff < 0.01])
        
        # Find longest steady region
        runs = []
        run_start = None
        for i, is_steady in enumerate(steady_mask):
            if is_steady and run_start is None:
                run_start = i
            elif not is_steady and run_start is not None:
                if i - run_start > 20:
                    runs.append((run_start, i, np.mean(cmd[run_start:i])))
                run_start = None
                
        # Top row: steady state zoom
        ax1 = axes[0, col]
        if runs:
            # Plot a mid-level steady region
            mid_runs = [r for r in runs if r[2] > 0.5]
            if mid_runs:
                start, end, cmd_level = mid_runs[0]
                t_window = time[start:end] - time[start]
                meas_window = meas[start:end]
                
                ax1.plot(t_window, meas_window, 'b-', linewidth=1, alpha=0.8)
                ax1.axhline(y=np.mean(meas_window), color='r', linestyle='--',
                           label=f'Mean: {np.mean(meas_window):.4f}')
                ax1.axhline(y=np.mean(meas_window) + np.std(meas_window), 
                           color='g', linestyle=':', label=f'±STD: {np.std(meas_window):.4f}')
                ax1.axhline(y=np.mean(meas_window) - np.std(meas_window), 
                           color='g', linestyle=':')
                           
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity (rev/s)')
        ax1.set_title(f'{side} Wheel - Steady State')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Bottom row: noise by command level
        ax2 = axes[1, col]
        if runs:
            cmd_levels = []
            noise_stds = []
            for start, end, cmd_level in runs:
                cmd_levels.append(cmd_level)
                noise_stds.append(np.std(meas[start:end]))
                
            ax2.scatter(cmd_levels, noise_stds, alpha=0.7, s=50)
            ax2.set_xlabel('Command Level (rev/s)')
            ax2.set_ylabel('Noise STD (rev/s)')
            ax2.set_title(f'{side} Wheel - Noise vs Speed')
            ax2.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_power_metrics(data, title="Power & Thermal", save_path=None):
    """Plot battery voltage, current, and temperature"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data['time']
    
    # Battery voltage
    ax1 = axes[0]
    if 'battery_volt' in data and np.any(data['battery_volt'] > 0):
        ax1.plot(time, data['battery_volt'], 'b-', linewidth=1)
        ax1.axhline(y=np.mean(data['battery_volt']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(data["battery_volt"]):.2f}V')
    ax1.set_ylabel('Battery Voltage (V)')
    ax1.set_title('Battery Voltage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Current
    ax2 = axes[1]
    if 'left_current' in data:
        ax2.plot(time, data['left_current'], 'b-', label='Left', alpha=0.7)
        ax2.plot(time, data['right_current'], 'r-', label='Right', alpha=0.7)
    ax2.set_ylabel('Motor Current (A)')
    ax2.set_title('Motor Current')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Temperature
    ax3 = axes[2]
    if 'left_temp' in data and np.any(data['left_temp'] > 0):
        ax3.plot(time, data['left_temp'], 'b-', label='Left', alpha=0.7)
        ax3.plot(time, data['right_temp'], 'r-', label='Right', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Motor Temperature')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_frequency_response(data, title="Frequency Response", save_path=None):
    """Plot frequency response from sine test data"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = data['time']
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    
    for col, (side, meas_key, cmd_key) in enumerate([
        ('Left', 'left_vel', 'left_cmd'),
        ('Right', 'right_vel', 'right_cmd')
    ]):
        cmd = data[cmd_key]
        meas = data[meas_key]
        
        # Remove DC component
        cmd_ac = cmd - np.mean(cmd)
        meas_ac = meas - np.mean(meas)
        
        # FFT
        n = len(cmd)
        freqs = np.fft.rfftfreq(n, dt)
        cmd_fft = np.abs(np.fft.rfft(cmd_ac))
        meas_fft = np.abs(np.fft.rfft(meas_ac))
        
        # Magnitude ratio (transfer function magnitude)
        with np.errstate(divide='ignore', invalid='ignore'):
            tf_mag = meas_fft / cmd_fft
            tf_mag[cmd_fft < 0.01 * np.max(cmd_fft)] = np.nan  # Ignore low-power freqs
        
        # Top: FFT spectra
        ax1 = axes[0, col]
        ax1.semilogy(freqs, cmd_fft, 'b-', label='Command', alpha=0.7)
        ax1.semilogy(freqs, meas_fft, 'r-', label='Measured', alpha=0.7)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title(f'{side} Wheel - FFT')
        ax1.set_xlim([0, min(10, fs/2)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Transfer function magnitude
        ax2 = axes[1, col]
        valid = ~np.isnan(tf_mag) & (freqs > 0.05) & (freqs < 10)
        if np.any(valid):
            ax2.semilogx(freqs[valid], 20 * np.log10(tf_mag[valid]), 'g-', linewidth=2)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.axhline(y=-3, color='r', linestyle=':', alpha=0.7, label='-3dB')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Gain (dB)')
        ax2.set_title(f'{side} Wheel - Transfer Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
        
    plt.close()


def generate_all_plots(csv_path, output_dir=None):
    """Generate all plots for a CSV file"""
    print(f"\nGenerating plots for: {csv_path}")
    
    data = load_csv(csv_path)
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
        
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Generate all plots
    plot_overview(data, f"Overview - {base_name}", 
                  os.path.join(output_dir, f"{base_name}_overview.png"))
    
    plot_linearity(data, f"Linearity - {base_name}",
                   os.path.join(output_dir, f"{base_name}_linearity.png"))
    
    plot_step_response(data, f"Step Response - {base_name}",
                       os.path.join(output_dir, f"{base_name}_step_response.png"))
    
    plot_symmetry(data, f"Symmetry - {base_name}",
                  os.path.join(output_dir, f"{base_name}_symmetry.png"))
    
    plot_noise_analysis(data, f"Noise - {base_name}",
                        os.path.join(output_dir, f"{base_name}_noise.png"))
    
    plot_power_metrics(data, f"Power - {base_name}",
                       os.path.join(output_dir, f"{base_name}_power.png"))
    
    # Frequency response only for sine tests
    if 'sine' in base_name.lower():
        plot_frequency_response(data, f"Frequency Response - {base_name}",
                                os.path.join(output_dir, f"{base_name}_freq_response.png"))
        
    print(f"✓ All plots generated in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize wheel calibration data')
    parser.add_argument('csv_files', nargs='*', help='CSV files to visualize')
    parser.add_argument('--all', metavar='DIR', help='Process all CSVs in directory')
    parser.add_argument('--output-dir', help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    if args.all:
        # Process all CSVs in directory
        csv_files = [os.path.join(args.all, f) for f in os.listdir(args.all) 
                     if f.endswith('.csv')]
    else:
        csv_files = args.csv_files
        
    if not csv_files:
        parser.print_help()
        print("\nExamples:")
        print("  python calibration_visualizer.py calibration_logs/step_test_*.csv")
        print("  python calibration_visualizer.py --all calibration_logs/")
        return
        
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            generate_all_plots(csv_file, args.output_dir)
        else:
            print(f"⚠ File not found: {csv_file}")


if __name__ == "__main__":
    main()
