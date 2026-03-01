"""
Problem 1: Robotic Arm Geometry & Safety Analysis
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. Path Configuration
# ==========================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent  # scripts/ -> go up to project root

CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

FILE_CONSTRAINTS = CONFIG_DIR / "motion_constraints_unified.json"
FILE_GEOMETRY = CONFIG_DIR / "robot_geometry.json"
FILE_MOTOR = CONFIG_DIR / "motor_parameters.json"
FILE_OUTPUT = OUTPUT_DIR / "01_final_coordinates.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Utility Classes
# ==========================================

class ConfigLoader:
    """Load and validate JSON configuration files."""

    @staticmethod
    def load(file_path: Path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise SystemExit(f"[Fatal Error] Missing config file: {file_path}")
        except json.JSONDecodeError:
            raise SystemExit(f"[Fatal Error] Invalid JSON format in: {file_path}")


class GeometricSolver:
    """Compute end-effector positions for Problem 1."""

    def __init__(self, geometry_cfg):
        # 从新的JSON结构读取手臂长度
        self.arm_length = geometry_cfg["simplified_geometry"]["arm_length_total_mm"]
        self.origin = np.zeros(3)  # Left shoulder as origin

    def solve_end_position(self, theta_deg, phi_deg):
        """
        Calculate end-effector position using spherical coordinates.
        
        Args:
            theta_deg: Elevation angle from Z-axis (degrees)
            phi_deg: Azimuthal angle from X-axis toward Y-axis (degrees)
        
        Returns:
            np.array: [x, y, z] coordinates in mm
        """
        L = self.arm_length
        theta = np.radians(theta_deg)
        phi = np.radians(phi_deg)

        z = L * np.cos(theta)
        r_xy = L * np.sin(theta)
        x = r_xy * np.cos(phi)
        y = r_xy * np.sin(phi)
        return np.array([x, y, z])

    @staticmethod
    def estimate_joint_angles(theta_deg, phi_deg):
        """
        Estimate shoulder joint angles from task-space description.
        
        Returns:
            dict: Estimated pitch and yaw angles in radians
        """
        return {
            "pitch_rad": np.radians(90 - theta_deg),
            "yaw_rad": np.radians(phi_deg)
        }


class SafetyValidator:
    """Evaluate joint limits and static torque safety."""

    def __init__(self, constraints_cfg, motor_cfg, geometry_cfg):
        self.limits = constraints_cfg["joint_limits"]
        
        # 从新的motor_parameters.json读取手臂电机参数
        arm_motor_group = motor_cfg["joint_groups"]["arm_joints"]
        self.Kt_effective = arm_motor_group["electrical_parameters"]["Kt_effective_Nm_per_A"]
        self.friction_coulomb = arm_motor_group["mechanical_parameters"]["friction_coulomb_Nm"]
        
        # 从geometry读取手臂质量（使用所有手臂link的总和）
        self.estimated_arm_mass = self._calculate_arm_mass(geometry_cfg)

    def _calculate_arm_mass(self, geometry_cfg):
        """Calculate total arm mass from link mass properties."""
        major_links = geometry_cfg["mass_properties"]["major_links"]
        arm_links = [
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link", 
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_roll_rubber_hand"
        ]
        
        total_mass = 0.0
        for link_name in arm_links:
            if link_name in major_links:
                total_mass += major_links[link_name]["mass_kg"]
        
        return total_mass

    def check(self, joint_angles, arm_len_mm):
        """
        Validate joint angles and required torques against motor limits.
        
        Args:
            joint_angles: dict with 'pitch_rad' and 'yaw_rad'
            arm_len_mm: Total arm length in millimeters
        
        Returns:
            dict: Safety check results with status and details
        """
        results = {"safe": True, "details": []}

        pitch = joint_angles["pitch_rad"]
        limit_pitch = self.limits["left_shoulder_pitch_joint"]

        # Check angle limits
        if not (limit_pitch["lower"] <= pitch <= limit_pitch["upper"]):
            results["safe"] = False
            results["details"].append(
                f"Pitch angle {pitch:.2f} rad violates limits "
                f"[{limit_pitch['lower']:.2f}, {limit_pitch['upper']:.2f}]."
            )

        # Calculate required static holding torque
        g = 9.81  # m/s^2
        L_com = (arm_len_mm / 1000) / 2  # Assume COM at midpoint
        torque_req = self.estimated_arm_mass * g * L_com * np.sin(np.radians(60))
        
        # Add friction torque (static holding requires overcoming Coulomb friction)
        torque_total = torque_req + self.friction_coulomb

        # Use continuous torque limit for static holding
        max_eff_cont = limit_pitch["effort_continuous"]
        max_eff_peak = limit_pitch["effort_peak"]
        
        if torque_total > max_eff_peak:
            results["safe"] = False
            results["details"].append(
                f"Required torque {torque_total:.2f} Nm exceeds peak motor limit {max_eff_peak:.2f} Nm."
            )
        elif torque_total > max_eff_cont:
            results["safe"] = False
            results["details"].append(
                f"Required torque {torque_total:.2f} Nm exceeds continuous limit {max_eff_cont:.2f} Nm "
                "(acceptable for short duration only)."
            )

        results["calc"] = {
            "gravity_torque_nm": round(torque_req, 4),
            "friction_torque_nm": round(self.friction_coulomb, 4),
            "total_required_torque_nm": round(torque_total, 4),
            "max_continuous_torque_nm": round(max_eff_cont, 4),
            "max_peak_torque_nm": round(max_eff_peak, 4),
            "estimated_arm_mass_kg": round(self.estimated_arm_mass, 4)
        }
        return results

# ==========================================
# 3. Visualization
# ==========================================

def visualize(origin, end_pos):
    """Generate 3D visualization of arm configuration."""
    plt.style.use("default")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Arm vector
    ax.plot(
        [origin[0], end_pos[0]],
        [origin[1], end_pos[1]],
        [origin[2], end_pos[2]],
        lw=4, marker="o", markersize=7, color="#1f77b4", label="Arm Link"
    )

    # Projections
    ax.plot([end_pos[0], end_pos[0]], [end_pos[1], end_pos[1]], [0, end_pos[2]], 
            "k--", alpha=0.3, label="Vertical Projection")
    ax.plot([0, end_pos[0]], [0, end_pos[1]], [0, 0], 
            "r--", alpha=0.3, label="XY Projection")

    # Coordinate axes
    axis_L = 150
    ax.quiver(0, 0, 0, axis_L, 0, 0, color="r", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_L, 0, color="g", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_L, color="b", arrow_length_ratio=0.1)
    ax.text(axis_L + 10, 0, 0, "X", color="r", fontsize=12, weight="bold")
    ax.text(0, axis_L + 10, 0, "Y", color="g", fontsize=12, weight="bold")
    ax.text(0, 0, axis_L + 10, "Z", color="b", fontsize=12, weight="bold")

    # End-effector label
    ax.text(end_pos[0], end_pos[1], end_pos[2] + 20,
            f"({end_pos[0]:.1f}, {end_pos[1]:.1f}, {end_pos[2]:.1f}) mm",
            fontsize=10, weight="bold", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlim(-50, 400)
    ax.set_ylim(-50, 400)
    ax.set_zlim(0, 400)
    ax.set_xlabel("X [mm]", fontsize=11)
    ax.set_ylabel("Y [mm]", fontsize=11)
    ax.set_zlabel("Z [mm]", fontsize=11)
    ax.set_title("Problem 1 — Robot Arm Kinematic Configuration", fontsize=13, weight="bold")
    ax.legend(loc="upper left")
    ax.view_init(elev=22, azim=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_viz.png", dpi=300, bbox_inches="tight")
    print(f"[Visualization] Saved to {OUTPUT_DIR / '01_viz.png'}")
    plt.show()


# ==========================================
# 4. Main Execution
# ==========================================

def main():
    print("=" * 60)
    print("APMCM 2025 — Problem 1 Solver")
    print(f"Configuration directory: {CONFIG_DIR}")
    print("=" * 60)

    # Load configuration files
    constraints_cfg = ConfigLoader.load(FILE_CONSTRAINTS)
    geometry_cfg = ConfigLoader.load(FILE_GEOMETRY)
    motor_cfg = ConfigLoader.load(FILE_MOTOR)

    # Initialize solvers
    solver = GeometricSolver(geometry_cfg)
    validator = SafetyValidator(constraints_cfg, motor_cfg, geometry_cfg)

    # Problem parameters from problem statement
    THETA = 60  # Elevation angle from Z-axis (degrees)
    PHI = 30    # Rotation to the left (azimuthal angle, degrees)

    print(f"\nInput Parameters:")
    print(f"  Arm length: {solver.arm_length} mm")
    print(f"  Elevation angle (θ): {THETA}°")
    print(f"  Azimuthal angle (φ): {PHI}°")

    # Solve kinematics
    end_pos = solver.solve_end_position(THETA, PHI)
    joint_angles = solver.estimate_joint_angles(THETA, PHI)
    
    # Safety validation
    safety = validator.check(joint_angles, solver.arm_length)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS — End Effector Position:")
    print(f"  X = {end_pos[0]:.3f} mm")
    print(f"  Y = {end_pos[1]:.3f} mm")
    print(f"  Z = {end_pos[2]:.3f} mm")
    print("=" * 60)
    
    print("\nSafety Validation:")
    print(f"  Status: {'✓ PASS' if safety['safe'] else '✗ FAIL'}")
    print(f"  Gravity Torque: {safety['calc']['gravity_torque_nm']:.3f} Nm")
    print(f"  Friction Torque: {safety['calc']['friction_torque_nm']:.3f} Nm")
    print(f"  Total Required: {safety['calc']['total_required_torque_nm']:.3f} Nm")
    print(f"  Motor Limit (Continuous): {safety['calc']['max_continuous_torque_nm']:.3f} Nm")
    print(f"  Motor Limit (Peak): {safety['calc']['max_peak_torque_nm']:.3f} Nm")
    
    if not safety["safe"]:
        print("\n⚠ Warnings:")
        for msg in safety["details"]:
            print(f"  - {msg}")

    # Save output
    output = {
        "problem": "Problem 1",
        "description": "Left arm end-effector coordinates and safety validation",
        "units": "mm",
        "input_parameters": {
            "elevation_angle_deg": THETA,
            "azimuthal_angle_deg": PHI,
            "arm_length_mm": float(solver.arm_length)
        },
        "coordinates": {
            "x": float(f"{end_pos[0]:.4f}"),
            "y": float(f"{end_pos[1]:.4f}"),
            "z": float(f"{end_pos[2]:.4f}")
        },
        "estimated_joint_angles_rad": {
            "shoulder_pitch": float(f"{joint_angles['pitch_rad']:.4f}"),
            "shoulder_yaw": float(f"{joint_angles['yaw_rad']:.4f}")
        },
        "safety_check": safety
    }

    with open(FILE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Results saved to: {FILE_OUTPUT}")
    print("=" * 60)

    # Generate visualization
    visualize(solver.origin, end_pos)


if __name__ == "__main__":
    main()