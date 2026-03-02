# -*- coding: utf-8 -*-
"""
Problem 3: Multi-Joint Collaborative Motion Planning for Dance Performance
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# 1. Path Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_GEOMETRY = CONFIG_DIR / "robot_geometry.json"
FILE_CONSTRAINTS = CONFIG_DIR / "motion_constraints_unified.json"

# =============================================================================
# 2. Configuration Loader
# =============================================================================
class ConfigLoader:
    """Lightweight JSON loader with error handling."""

    @staticmethod
    def load(path: Path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ConfigLoader] Failed to read {path}: {e}")
            sys.exit(1)


# =============================================================================
# 3. Dance Motion Planner
# =============================================================================
class DancePlanner:
    """
    Planner for whole-body coordinated dance-like motion.
    
    Features:
    - Waist yaw rotation with smooth S-curve profile
    - Arms drawing circles in opposite directions
    - Simplified leg balancing (static stance)
    """

    def __init__(self, geometry_config, constraints_config):
        # Load arm dimensions from new JSON structure
        arm_dims = geometry_config["link_dimensions"]["arms"]
        self.L1 = arm_dims["upper_arm_length_mm"]  # Upper arm length
        self.L2 = arm_dims["forearm_length_mm"]    # Forearm length
        
        # Load problem-specific parameters
        prob3_params = geometry_config["problem_specific_parameters"]["problem_3"]
        self.period = prob3_params["arm_circle_period_s"]
        self.circle_radius = prob3_params["arm_circle_radius_mm"]
        self.target_yaw_deg = prob3_params["waist_rotation_deg"]
        
        # Shoulder position from torso dimensions
        torso_dims = geometry_config["link_dimensions"]["torso"]
        self.shoulder_offset_mm = torso_dims["shoulder_lateral_offset_mm"]
        
        # Joint limits for validation
        self.joint_limits = constraints_config["joint_limits"]
        
        print(f"[DancePlanner] Initialized:")
        print(f"  Upper arm: {self.L1} mm, Forearm: {self.L2} mm")
        print(f"  Circle radius: {self.circle_radius} mm")
        print(f"  Movement period: {self.period} s")
        print(f"  Waist rotation: {self.target_yaw_deg}°")

    # -------------------------------------------------------------------------
    def get_waist_yaw(self, t):
        """
        Smooth S-curve (quintic polynomial) for waist yaw rotation.
        
        Args:
            t: Time in seconds
        
        Returns:
            float: Waist yaw angle in degrees
        """
        tau = t / self.period
        if tau <= 0:
            return 0.0
        if tau >= 1:
            return self.target_yaw_deg
        
        # Quintic smoothing: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
        smooth = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        return self.target_yaw_deg * smooth

    # -------------------------------------------------------------------------
    def solve_arm_ik_circle(self, t, side='left'):
        """
        Solve inverse kinematics for arm drawing a circular path.
        
        The circle is defined in the YZ plane (perpendicular to body):
        - Y: lateral direction (left-right)
        - Z: vertical direction (up-down)
        
        Args:
            t: Time in seconds
            side: 'left' or 'right' arm
        
        Returns:
            dict: Joint angles and end-effector position
        """
        omega = 2 * np.pi / self.period
        
        # Opposite phase for left and right arms
        phase_offset = 0.0 if side == 'left' else np.pi

        # Circular trajectory in YZ plane
        y = self.circle_radius * np.cos(omega * t + phase_offset)
        z = self.circle_radius * np.sin(omega * t + phase_offset)

        # Shoulder position in body frame
        shoulder_lateral_offset = self.shoulder_offset_mm if side == 'left' else -self.shoulder_offset_mm

        # Distance from shoulder to target point
        D = np.sqrt(y**2 + z**2)
        D = np.clip(D, 1e-3, self.L1 + self.L2 - 0.1)  # Prevent singularity

        # Solve elbow angle using law of cosines
        cos_elbow = (self.L1**2 + self.L2**2 - D**2) / (2 * self.L1 * self.L2)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        theta_elbow = np.arccos(cos_elbow)

        # Solve shoulder angle
        cos_beta = (self.L1**2 + D**2 - self.L2**2) / (2 * self.L1 * D)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = np.arccos(cos_beta)
        
        alpha = np.arctan2(z, y)
        theta_shoulder_roll = alpha + beta

        # Apply joint limits (clamp to safe range)
        joint_prefix = f"{side}_shoulder_roll_joint"
        if joint_prefix in self.joint_limits:
            lim = self.joint_limits[joint_prefix]
            theta_shoulder_roll = np.clip(theta_shoulder_roll, lim["lower"], lim["upper"])
        
        elbow_joint = f"{side}_elbow_joint"
        if elbow_joint in self.joint_limits:
            lim = self.joint_limits[elbow_joint]
            theta_elbow = np.clip(theta_elbow, lim["lower"], lim["upper"])

        return {
            f"{side}_shoulder_pitch_rad": 0.0,  # Simplified: no pitch motion
            f"{side}_shoulder_roll_rad": float(theta_shoulder_roll),
            f"{side}_shoulder_yaw_rad": 0.0,    # Simplified: no yaw motion
            f"{side}_elbow_rad": float(theta_elbow),
            f"{side}_wrist_roll_rad": 0.0,      # Wrist remains neutral
            f"{side}_end_effector_mm": {
                "x": float(shoulder_lateral_offset),
                "y": float(y),
                "z": float(z)
            }
        }

    # -------------------------------------------------------------------------
    def get_leg_balance_posture(self):
        """
        Return static leg configuration for balance during upper body motion.
        
        Simplified approach: slight knee bend to lower center of mass.
        
        Returns:
            dict: Joint angles for both legs
        """
        # Balanced standing posture with slight knee flexion
        return {
            "left_hip_pitch_rad": 0.0,
            "left_hip_roll_rad": 0.0,
            "left_hip_yaw_rad": 0.0,
            "left_knee_rad": 0.2,  # ~11° knee bend for stability
            "left_ankle_pitch_rad": -0.1,
            "left_ankle_roll_rad": 0.0,
            
            "right_hip_pitch_rad": 0.0,
            "right_hip_roll_rad": 0.0,
            "right_hip_yaw_rad": 0.0,
            "right_knee_rad": 0.2,
            "right_ankle_pitch_rad": -0.1,
            "right_ankle_roll_rad": 0.0
        }


# =============================================================================
# 4. Main Solver
# =============================================================================
def solve_problem_3():
    """Main execution function for Problem 3."""
    print("=" * 60)
    print("APMCM 2025 — Problem 3 Solver")
    print("Multi-Joint Collaborative Dance Motion")
    print("=" * 60)

    # Load configurations
    geometry = ConfigLoader.load(FILE_GEOMETRY)
    constraints = ConfigLoader.load(FILE_CONSTRAINTS)
    
    # Initialize planner
    planner = DancePlanner(geometry, constraints)

    # Time discretization
    dt = 0.02  # 50 Hz sampling rate
    times = np.arange(0, planner.period + dt, dt)

    # Storage for trajectory data
    trajectory = {
        "problem": "Problem 3",
        "description": "Multi-joint coordinated dance motion with arms drawing circles and waist rotation",
        "time_array_s": times.tolist(),
        "sample_rate_hz": int(1 / dt),
        "frames": []
    }

    print(f"\nGenerating trajectory: {len(times)} frames over {planner.period} seconds...")

    # Generate motion for each time step
    for t in times:
        # Waist rotation
        waist_yaw = planner.get_waist_yaw(t)
        
        # Arm circular motion (opposite directions)
        left_arm = planner.solve_arm_ik_circle(t, 'left')
        right_arm = planner.solve_arm_ik_circle(t, 'right')
        
        # Leg balancing posture
        legs = planner.get_leg_balance_posture()
        
        # Combine all joint angles
        frame_data = {
            "time_s": float(t),
            "waist": {
                "yaw_deg": float(waist_yaw),
                "yaw_rad": float(np.radians(waist_yaw))
            },
            "left_arm": left_arm,
            "right_arm": right_arm,
            "legs": legs
        }
        
        trajectory["frames"].append(frame_data)

    # Save trajectory to JSON
    output_file = OUTPUT_DIR / "03_dance_trajectory.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=4, ensure_ascii=False)

    print(f"\n✓ Trajectory saved to: {output_file}")

    # Generate visualization
    visualize_dance(times, trajectory, planner)

    print("\n" + "=" * 60)
    print("Problem 3 Solver Completed Successfully")
    print("=" * 60)


# =============================================================================
# 5. Visualization
# =============================================================================
def visualize_dance(times, trajectory_data, planner):
    """Generate comprehensive visualization of dance motion."""
    
    print("\nGenerating visualization plots...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 10))

    # Extract data for plotting
    frames = trajectory_data["frames"]
    waist_yaw_deg = [f["waist"]["yaw_deg"] for f in frames]
    
    left_shoulder_roll = [f["left_arm"]["left_shoulder_roll_rad"] for f in frames]
    left_elbow = [f["left_arm"]["left_elbow_rad"] for f in frames]
    
    right_shoulder_roll = [f["right_arm"]["right_shoulder_roll_rad"] for f in frames]
    right_elbow = [f["right_arm"]["right_elbow_rad"] for f in frames]
    
    # Subplot 1: Waist yaw motion
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(times, waist_yaw_deg, linewidth=2, color='tab:blue')
    ax1.axhline(y=planner.target_yaw_deg, color='red', linestyle='--', 
                label=f'Target: {planner.target_yaw_deg}°', alpha=0.7)
    ax1.set_title("Torso Yaw Rotation (S-curve)", fontsize=11, weight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Yaw Angle (degrees)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Left arm joint angles
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(times, np.degrees(left_shoulder_roll), label="Shoulder Roll", linewidth=1.5)
    ax2.plot(times, np.degrees(left_elbow), '--', label="Elbow", linewidth=1.5)
    ax2.set_title("Left Arm Joint Angles", fontsize=11, weight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (degrees)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Right arm joint angles
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(times, np.degrees(right_shoulder_roll), label="Shoulder Roll", 
             linewidth=1.5, color='tab:orange')
    ax3.plot(times, np.degrees(right_elbow), '--', label="Elbow", 
             linewidth=1.5, color='tab:red')
    ax3.set_title("Right Arm Joint Angles", fontsize=11, weight='bold')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angle (degrees)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: 3D trajectory - Left hand
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    left_x = [planner.shoulder_offset_mm for _ in times]
    left_y = [f["left_arm"]["left_end_effector_mm"]["y"] for f in frames]
    left_z = [f["left_arm"]["left_end_effector_mm"]["z"] for f in frames]
    
    ax4.plot(left_x, left_y, left_z, linewidth=2, color='tab:blue', label="Left Hand Path")
    ax4.scatter([planner.shoulder_offset_mm], [0], [0], s=60, c='red', 
                marker='o', label="Left Shoulder")
    ax4.set_xlabel("X (mm)")
    ax4.set_ylabel("Y (mm)")
    ax4.set_zlabel("Z (mm)")
    ax4.set_title("Left Hand Circular Trajectory", fontsize=11, weight='bold')
    ax4.legend()
    ax4.view_init(elev=20, azim=45)

    # Subplot 5: 3D trajectory - Right hand
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    right_x = [-planner.shoulder_offset_mm for _ in times]
    right_y = [f["right_arm"]["right_end_effector_mm"]["y"] for f in frames]
    right_z = [f["right_arm"]["right_end_effector_mm"]["z"] for f in frames]
    
    ax5.plot(right_x, right_y, right_z, linewidth=2, color='tab:orange', label="Right Hand Path")
    ax5.scatter([-planner.shoulder_offset_mm], [0], [0], s=60, c='red', 
                marker='o', label="Right Shoulder")
    ax5.set_xlabel("X (mm)")
    ax5.set_ylabel("Y (mm)")
    ax5.set_zlabel("Z (mm)")
    ax5.set_title("Right Hand Circular Trajectory", fontsize=11, weight='bold')
    ax5.legend()
    ax5.view_init(elev=20, azim=45)

    # Subplot 6: Combined YZ plane view
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(left_y, left_z, linewidth=2, label="Left Hand", color='tab:blue')
    ax6.plot(right_y, right_z, linewidth=2, label="Right Hand", color='tab:orange')
    ax6.scatter([0], [0], s=80, c='red', marker='x', label="Body Center")
    ax6.set_xlabel("Y (mm) - Lateral")
    ax6.set_ylabel("Z (mm) - Vertical")
    ax6.set_title("Both Hands in YZ Plane", fontsize=11, weight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')

    plt.tight_layout()

    # Save figure
    fig_path = OUTPUT_DIR / "03_dance_visualization.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"✓ Visualization saved to: {fig_path}")

    plt.show()


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    solve_problem_3()