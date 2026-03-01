"""
Problem 2: G1 Robot Straight Walking Simulation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

# ================================================================
# 1. Path Configuration
# ================================================================
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_CONSTRAINTS = CONFIG_DIR / "motion_constraints_unified.json"
FILE_GEOMETRY = CONFIG_DIR / "robot_geometry.json"

# ================================================================
# 2. Configuration Loader
# ================================================================
def load_config():
    """Load robot configuration from JSON files."""
    with open(FILE_CONSTRAINTS, 'r', encoding='utf-8') as f:
        constraints = json.load(f)
    with open(FILE_GEOMETRY, 'r', encoding='utf-8') as f:
        geometry = json.load(f)
    
    # Extract joint limits
    joint_limits = constraints["joint_limits"]
    
    # Extract leg dimensions from geometry
    leg_dims = geometry["link_dimensions"]["legs"]
    thigh_mm = leg_dims["thigh_length_mm"]
    shank_mm = leg_dims["shank_length_mm"]
    
    return joint_limits, thigh_mm, shank_mm

# ================================================================
# 3. Velocity Planner
# ================================================================
class TrapezoidalVelocityPlanner:
    """Generate a trapezoidal COM velocity profile."""
    
    def __init__(self, total_dist, avg_speed, dt=0.01):
        self.dist = total_dist
        self.v_avg = avg_speed
        self.dt = dt
        self.total_time = total_dist / avg_speed

    def generate(self):
        """
        Generate time, position, and velocity arrays.
        
        Returns:
            tuple: (time_array, position_array, velocity_array)
        """
        t = np.arange(0, self.total_time + self.dt, self.dt)
        v = np.zeros_like(t)

        # Trapezoidal profile: 40% accel, 20% constant, 40% decel
        t_acc = 0.4 * self.total_time
        t_dec = 0.4 * self.total_time
        t_flat = self.total_time - t_acc - t_dec

        # Calculate max velocity to achieve desired average
        v_max = self.dist / (0.5 * t_acc + t_flat + 0.5 * t_dec)

        for i, ti in enumerate(t):
            if ti < t_acc:
                v[i] = v_max * (ti / t_acc)
            elif ti < t_acc + t_flat:
                v[i] = v_max
            else:
                v[i] = v_max * max(0.0, 1 - (ti - t_acc - t_flat) / t_dec)

        # Integrate velocity to get position
        x = np.cumsum(v) * self.dt
        return t, x, v

# ================================================================
# 4. Inverse Kinematics (Sagittal Plane)
# ================================================================
class LegIKSolver:
    """Inverse kinematics for a sagittal-plane thigh-shank leg."""
    
    def __init__(self, thigh_mm, shank_mm, limits):
        self.L1 = thigh_mm / 1000.0  # Convert to meters
        self.L2 = shank_mm / 1000.0
        self.lim = limits

    def _clamp(self, value, lo, hi):
        """Clamp value to joint limits."""
        return np.clip(value, lo, hi)

    def solve(self, x_rel, z_rel, is_swing=True, step_h=0.1):
        """
        Solve 2-link IK for leg configuration.
        
        Args:
            x_rel: Horizontal distance from hip to foot (m)
            z_rel: Vertical distance from hip to foot (m, negative downward)
            is_swing: Whether this is swing phase (affects ankle control)
            step_h: Step height for swing phase (m)
        
        Returns:
            tuple: (hip_pitch, knee, ankle_pitch, ankle_roll) angles in radians
        """
        D = np.hypot(x_rel, z_rel)
        max_len = 0.999 * (self.L1 + self.L2)
        D = min(D, max_len)

        # Solve knee angle using law of cosines
        cos_phi = (self.L1**2 + self.L2**2 - D**2) / (2 * self.L1 * self.L2)
        phi = np.arccos(np.clip(cos_phi, -1, 1))
        theta_knee = np.pi - phi

        # Solve hip angle
        alpha = np.arctan2(z_rel, x_rel)
        cos_beta = (self.L1**2 + D**2 - self.L2**2) / (2 * self.L1 * D)
        beta = np.arccos(np.clip(cos_beta, -1, 1))
        theta_hip = alpha + beta

        # Ankle control (simplified)
        if is_swing:
            theta_ankle_pitch = -theta_hip + theta_knee + np.arcsin(np.clip(z_rel / D, -1, 1))
        else:
            theta_ankle_pitch = 0.0
        
        theta_ankle_roll = 0.0

        # Apply joint limits
        theta_knee = self._clamp(
            theta_knee, 
            self.lim["left_knee_joint"]["lower"], 
            self.lim["left_knee_joint"]["upper"]
        )
        theta_hip = self._clamp(
            theta_hip, 
            self.lim["left_hip_pitch_joint"]["lower"], 
            self.lim["left_hip_pitch_joint"]["upper"]
        )
        theta_ankle_pitch = self._clamp(
            theta_ankle_pitch, 
            self.lim["left_ankle_pitch_joint"]["lower"], 
            self.lim["left_ankle_pitch_joint"]["upper"]
        )
        theta_ankle_roll = self._clamp(
            theta_ankle_roll, 
            self.lim["left_ankle_roll_joint"]["lower"], 
            self.lim["left_ankle_roll_joint"]["upper"]
        )

        return theta_hip, theta_knee, theta_ankle_pitch, theta_ankle_roll

# ================================================================
# 5. Problem 2 Solver
# ================================================================
def solve_problem_2():
    """Main solver for Problem 2."""
    print("=" * 60)
    print("APMCM 2025 — Problem 2 Solver")
    print("G1 Robot Walking Trajectory Planning")
    print("=" * 60)

    # Load configuration
    joint_limits, thigh_mm, shank_mm = load_config()
    
    print(f"\nRobot Leg Dimensions:")
    print(f"  Thigh length: {thigh_mm} mm")
    print(f"  Shank length: {shank_mm} mm")

    # Walking parameters
    TOTAL_DISTANCE = 10.0  # meters
    AVG_SPEED = 2.0        # m/s
    STEP_LENGTH = 0.8      # meters
    HIP_HEIGHT = 0.55      # meters (approximate)
    STEP_HEIGHT = 0.08     # meters (swing phase clearance)

    print(f"\nWalking Parameters:")
    print(f"  Total distance: {TOTAL_DISTANCE} m")
    print(f"  Average speed: {AVG_SPEED} m/s")
    print(f"  Step length: {STEP_LENGTH} m")
    print(f"  Step height: {STEP_HEIGHT} m")

    # Generate velocity profile
    planner = TrapezoidalVelocityPlanner(TOTAL_DISTANCE, AVG_SPEED)
    t_arr, x_com, v_com = planner.generate()

    # Initialize IK solver
    ik = LegIKSolver(thigh_mm, shank_mm, joint_limits)

    # Storage for joint trajectories
    hip_list, knee_list = [], []
    ankle_pitch_list, ankle_roll_list = [], []
    record = []

    # Simulate walking cycle
    for i, xc in enumerate(x_com):
        # Determine phase within gait cycle
        cycle_pos = xc % (2 * STEP_LENGTH)
        
        if cycle_pos < STEP_LENGTH:
            # Swing phase (left leg)
            s = cycle_pos / STEP_LENGTH
            x_rel = -STEP_LENGTH / 2 + s * STEP_LENGTH
            z_rel = -HIP_HEIGHT + STEP_HEIGHT * np.sin(np.pi * s)
            swing_flag = True
        else:
            # Stance phase (left leg)
            s = (cycle_pos - STEP_LENGTH) / STEP_LENGTH
            x_rel = STEP_LENGTH / 2 - s * STEP_LENGTH
            z_rel = -HIP_HEIGHT
            swing_flag = False

        # Solve IK
        th_hip, th_knee, th_ank_p, th_ank_r = ik.solve(
            x_rel, z_rel, is_swing=swing_flag, step_h=STEP_HEIGHT
        )

        hip_list.append(th_hip)
        knee_list.append(th_knee)
        ankle_pitch_list.append(th_ank_p)
        ankle_roll_list.append(th_ank_r)

        record.append({
            "time": float(t_arr[i]),
            "com_position_m": float(xc),
            "com_velocity_m_s": float(v_com[i]),
            "hip_pitch_rad": float(th_hip),
            "knee_rad": float(th_knee),
            "ankle_pitch_rad": float(th_ank_p),
            "ankle_roll_rad": float(th_ank_r)
        })

    # Convert to numpy arrays
    hip_list = np.array(hip_list)
    knee_list = np.array(knee_list)
    ankle_pitch_list = np.array(ankle_pitch_list)
    ankle_roll_list = np.array(ankle_roll_list)

    # Calculate knee angular velocity
    omega_knee = np.gradient(knee_list, planner.dt)
    
    # Find peaks in absolute angular velocity
    peaks, _ = find_peaks(np.abs(omega_knee))
    peak_times = t_arr[peaks]
    peak_vals = omega_knee[peaks]

    # Find global maximum
    max_idx = np.argmax(np.abs(peak_vals))
    max_time = peak_times[max_idx]
    max_vel = peak_vals[max_idx]

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Total walking time: {planner.total_time:.4f} s")
    print(f"  Maximum knee angular velocity: {abs(max_vel):.4f} rad/s")
    print(f"  Occurs at time: {max_time:.4f} s")
    print("=" * 60)

    # Export results as JSON
    output_file = OUTPUT_DIR / "02_walking_trajectory.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "problem": "Problem 2",
            "description": "Walking trajectory with joint angles and velocities",
            "total_time_s": float(planner.total_time),
            "max_knee_velocity": {
                "time_s": float(max_time),
                "value_rad_s": float(abs(max_vel))
            },
            "sample_rate_hz": int(1 / planner.dt),
            "trajectory_data": record
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")

    # Generate plots
    visualize_walking(t_arr, v_com, omega_knee, hip_list, knee_list, 
                      ankle_pitch_list, ankle_roll_list, 
                      peak_times, peak_vals, max_time, max_vel)


# ================================================================
# 6. Visualization
# ================================================================
def visualize_walking(t_arr, v_com, omega_knee, hip, knee, ankle_pitch, ankle_roll,
                      peak_times, peak_vals, max_time, max_vel):
    """Generate comprehensive walking trajectory plots."""
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(12, 10))

    # Subplot 1: COM velocity
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t_arr, v_com, linewidth=1.5, color='tab:blue')
    ax1.set_ylabel("Velocity (m/s)", fontsize=11)
    ax1.set_title("G1 Robot — Center of Mass Velocity Profile", fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Knee angular velocity
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(t_arr, omega_knee, linewidth=1.5, label="Knee Angular Velocity", color='tab:orange')
    ax2.scatter(peak_times, peak_vals, color="red", s=30, zorder=3, label="Local Peaks")
    ax2.scatter([max_time], [max_vel], color="black", s=100, marker='*', 
                zorder=4, label=f"Global Max: {abs(max_vel):.3f} rad/s")
    ax2.set_ylabel("Angular Velocity (rad/s)", fontsize=11)
    ax2.set_title("G1 Robot — Knee Joint Angular Velocity", fontsize=12, weight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: All joint angles
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(t_arr, np.degrees(hip), label="Hip Pitch", linewidth=1.5)
    ax3.plot(t_arr, np.degrees(knee), label="Knee", linewidth=1.5)
    ax3.plot(t_arr, np.degrees(ankle_pitch), label="Ankle Pitch", linewidth=1.5)
    ax3.plot(t_arr, np.degrees(ankle_roll), label="Ankle Roll", linewidth=1.5, linestyle='--')
    ax3.set_xlabel("Time (s)", fontsize=11)
    ax3.set_ylabel("Angle (degrees)", fontsize=11)
    ax3.set_title("G1 Robot — Single-Leg Joint Angles", fontsize=12, weight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    plot_file = OUTPUT_DIR / "02_walking_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    
    plt.show()


# ================================================================
# Entry Point
# ================================================================
if __name__ == "__main__":
    solve_problem_2()