# -*- coding: utf-8 -*-
"""
Problem 4: Energy Consumption Calculation & Optimization
Based on Pinocchio Library for Rigid Body Dynamics
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from pathlib import Path
from scipy.interpolate import CubicSpline

# =============================================================================
# 1. Path Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data" / "output"
MODEL_DIR = PROJECT_ROOT / "data" / "raw" / "urdf"

OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_CONSTRAINTS = CONFIG_DIR / "motion_constraints_unified.json"
FILE_MOTOR = CONFIG_DIR / "motor_parameters.json"
FILE_URDF = MODEL_DIR / "g1.urdf" 

# Previous output files
FILE_P1_RES = DATA_DIR / "01_final_coordinates.json"
FILE_P2_TRAJ = DATA_DIR / "02_walking_trajectory.json"
FILE_P3_TRAJ = DATA_DIR / "03_dance_trajectory.json"

# =============================================================================
# 2. Helper Classes
# =============================================================================

class ConfigLoader:
    @staticmethod
    def load(path: Path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")
            sys.exit(1)

class RobotModel:
    """Wrapper for Pinocchio Model."""
    def __init__(self, urdf_path):
        # Ensure URDF extension
        if urdf_path.suffix == '.txt':
            target_urdf = urdf_path.with_suffix('.urdf')
            if not target_urdf.exists():
                print(f"[Info] Copying {urdf_path} to {target_urdf} for Pinocchio...")
                with open(urdf_path, 'r') as src, open(target_urdf, 'w') as dst:
                    dst.write(src.read())
            urdf_path = target_urdf
            
        try:
            self.model = pin.buildModelFromUrdf(str(urdf_path))
            self.data = self.model.createData()
            self.nq = self.model.nq
            self.nv = self.model.nv
            print(f"[Pinocchio] Model loaded. DoF: {self.nv}, Joints: {self.model.njoints}")
            
            # Create joint name to index map
            self.joint_map = {}
            for i, name in enumerate(self.model.names):
                self.joint_map[name] = i
                
        except Exception as e:
            print(f"[Error] Pinocchio load failed: {e}")
            sys.exit(1)

    def get_id(self, joint_name):
        return self.joint_map.get(joint_name, -1)

    def inverse_dynamics(self, q, v, a):
        """Compute Inverse Dynamics (RNEA). Returns torque vector tau."""
        return pin.rnea(self.model, self.data, q, v, a)

class EnergyModel:
    """Calculates electrical energy based on dynamic torque."""
    def __init__(self, motor_config):
        self.params = {}
        
        # Parse motor parameters into a quick lookup dict
        groups = motor_config["joint_groups"]
        for group_name, data in groups.items():
            # --- FIX: Skip description strings or metadata ---
            if not isinstance(data, dict):
                continue
            if "electrical_parameters" not in data:
                continue
            # -------------------------------------------------

            elec = data["electrical_parameters"]
            mech = data["mechanical_parameters"]
            
            kt = elec["Kt_effective_Nm_per_A"]
            res = elec["resistance_ohm"]
            f_vis = mech["friction_viscous_Nms_per_rad"]
            f_coul = mech["friction_coulomb_Nm"]
            
            for joint in data["joint_list"]:
                self.params[joint] = {
                    "Kt": kt, "R": res, "B": f_vis, "Tc": f_coul
                }
        
        # Default parameter for unlisted joints
        self.default_param = {"Kt": 1.0, "R": 0.1, "B": 0.01, "Tc": 0.1}

    def calculate_instantaneous_power(self, joint_name, torque_id, velocity):
        """
        Calculate P_elec = P_mech + P_copper + P_friction
        """
        p = self.params.get(joint_name, self.default_param)
        
        # 1. Friction Torque Model
        tau_fric = p["Tc"] * np.sign(velocity) + p["B"] * velocity
        tau_total = torque_id + tau_fric
        
        # 2. Copper Loss (Heat)
        current = tau_total / p["Kt"]
        p_heat = (current ** 2) * p["R"]
        
        # 3. Mechanical Power
        p_mech = np.abs(tau_total * velocity) 
        
        p_total = p_mech + p_heat
        
        return p_total, p_mech, p_heat, current

# =============================================================================
# 3. Trajectory Processors
# =============================================================================

class SolutionProcessor:
    def __init__(self, robot: RobotModel, energy: EnergyModel):
        self.robot = robot
        self.energy = energy
        self.dt = 0.01 # Default dt, overridden by data

    def map_state_to_pinocchio(self, frame_dict, q_vec):
        """Helper to map JSON dict keys to Pinocchio configuration vector."""
        # Reset q
        q_vec.fill(0)
        
        # Flatten dictionary 
        flat_dict = {}
        for k, v in frame_dict.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    std_name = sub_k.replace("_rad", "_joint")
                    flat_dict[std_name] = sub_v
            else:
                std_name = k.replace("_rad", "_joint")
                flat_dict[std_name] = v
        
        # Direct mapping for known joint names in model
        for j_name in self.robot.model.names:
            if j_name == "universe": continue
            
            # Heuristic matching
            key_match = None
            if j_name in flat_dict:
                key_match = j_name
            else:
                # Try removing "_joint"
                short_name = j_name.replace("_joint", "")
                if short_name in flat_dict: # e.g. "waist_yaw"
                    key_match = short_name
            
            if key_match:
                idx = self.robot.model.getJointId(j_name)
                idx_q = self.robot.model.joints[idx].idx_q
                q_vec[idx_q] = flat_dict[key_match]
                
        return q_vec

    def process_problem1(self, p1_data):
        """P1 is static holding."""
        print("\n--- Processing Problem 1 (Static Hold) ---")
        q = np.zeros(self.robot.nq)
        v = np.zeros(self.robot.nv)
        a = np.zeros(self.robot.nv)
        
        # Set P1 posture
        angles = p1_data["estimated_joint_angles_rad"]
        
        id_pitch = self.robot.model.getJointId("left_shoulder_pitch_joint")
        id_yaw = self.robot.model.getJointId("left_shoulder_yaw_joint")
        
        q[self.robot.model.joints[id_pitch].idx_q] = angles["shoulder_pitch"]
        q[self.robot.model.joints[id_yaw].idx_q] = angles["shoulder_yaw"]
        
        id_elbow = self.robot.model.getJointId("left_elbow_joint")
        # Initial/Rest config for elbow if needed, usually 0 or bent
        # Assuming 0 for P1 outstretched check or problem definition
        
        # Inverse Dynamics
        tau = self.robot.inverse_dynamics(q, v, a)
        
        total_power = 0
        print(f"{'Joint':<30} | {'Torque (Nm)':<12} | {'Power (W)':<12} | {'Current (A)':<12}")
        print("-" * 80)
        
        for name, i in self.robot.joint_map.items():
            if name == "universe": continue
            idx_v = self.robot.model.joints[i].idx_v
            t_val = tau[idx_v]
            
            if abs(t_val) > 0.01: 
                p_tot, _, _, curr = self.energy.calculate_instantaneous_power(name, t_val, 0.0)
                total_power += p_tot
                print(f"{name:<30} | {t_val:12.4f} | {p_tot:12.4f} | {curr:12.4f}")
        
        print(f"\nTotal Static Power: {total_power:.4f} W")
        energy_joules = total_power * 2.0 # 2 seconds duration assumption
        return energy_joules, total_power

    def process_trajectory(self, name, times, q_matrix):
        print(f"\n--- Processing {name} ---")
        N = len(times)
        if N < 2:
            return 0.0, np.array([0.0]), np.zeros((1, self.robot.nv)), times

        dt_avg = np.mean(np.diff(times))
        
        # 1. Compute Derivatives
        v_matrix = np.zeros((N, self.robot.nv))
        a_matrix = np.zeros((N, self.robot.nv))
        max_acc = 50.0  # Suppose the maximum angular acceleration is rad/s^2
        
        for j in range(self.robot.nv):
            v_matrix[:, j] = np.gradient(q_matrix[:, j], times)
            a_matrix[:, j] = np.gradient(v_matrix[:, j], times)
            
        np.clip(a_matrix, -max_acc, max_acc, out=a_matrix)
        # 2. Compute Dynamics & Energy
        total_energy = 0.0
        power_history = []
        torque_history = []
        
        for i in range(N):
            q = q_matrix[i, :]
            v = v_matrix[i, :]
            a = a_matrix[i, :]
            
            tau = self.robot.inverse_dynamics(q, v, a)
            torque_history.append(tau)
            
            frame_power = 0.0
            
            for j_name, j_idx in self.robot.joint_map.items():
                if j_name == "universe": continue
                idx_v = self.robot.model.joints[j_idx].idx_v
                
                t_val = tau[idx_v]
                v_val = v[idx_v]
                
                p_tot, _, _, _ = self.energy.calculate_instantaneous_power(j_name, t_val, v_val)
                frame_power += p_tot
            
            power_history.append(frame_power)
            total_energy += frame_power * dt_avg

        print(f"Total Energy for {name}: {total_energy:.4f} Joules")
        print(f"Average Power: {np.mean(power_history):.4f} W")
        print(f"Peak Power: {np.max(power_history):.4f} W")
        
        return total_energy, np.array(power_history), np.array(torque_history), times

    def process_problem2(self, p2_data):
        times = np.array([d['time'] for d in p2_data['trajectory_data']])
        N = len(times)
        q_matrix = np.zeros((N, self.robot.nq))
        
        # P2 provides single leg. We map it to LEFT leg.
        for i, frame in enumerate(p2_data['trajectory_data']):
            # Map Left Leg
            # Ensure IDs exist before accessing
            id_hp = self.robot.get_id('left_hip_pitch_joint')
            id_kn = self.robot.get_id('left_knee_joint')
            id_ap = self.robot.get_id('left_ankle_pitch_joint')
            id_ar = self.robot.get_id('left_ankle_roll_joint')
            
            if id_hp >= 0: q_matrix[i, self.robot.model.joints[id_hp].idx_q] = frame['hip_pitch_rad']
            if id_kn >= 0: q_matrix[i, self.robot.model.joints[id_kn].idx_q] = frame['knee_rad']
            if id_ap >= 0: q_matrix[i, self.robot.model.joints[id_ap].idx_q] = frame['ankle_pitch_rad']
            if id_ar >= 0: q_matrix[i, self.robot.model.joints[id_ar].idx_q] = frame['ankle_roll_rad']
            
        e, p, t, time_arr = self.process_trajectory("Problem 2 (Single Leg)", times, q_matrix)
        return e * 2, p * 2, t, time_arr 

    def process_problem3(self, p3_data):
        times = np.array(p3_data['time_array_s'])
        N = len(times)
        q_matrix = np.zeros((N, self.robot.nq))
        
        for i, frame in enumerate(p3_data['frames']):
            combined = {}
            combined.update(frame['waist'])
            combined.update(frame['left_arm'])
            combined.update(frame['right_arm'])
            combined.update(frame['legs'])
            
            self.map_state_to_pinocchio(combined, q_matrix[i, :])
            
        return self.process_trajectory("Problem 3 (Dance)", times, q_matrix)

# =============================================================================
# 4. Optimization Module
# =============================================================================

def optimize_walking_profile(base_energy, total_dist=10.0, total_time=5.0):
    print("\n--- Optimization: Walking Velocity Profile ---")
    
    ratios = np.linspace(0.1, 0.45, 10)
    energies_trap = []
    energies_scurve = []
    peak_torques = []
    
    def simulate_1d(ratio, profile_type):
        dt = 0.05
        t = np.arange(0, total_time + dt, dt)
        v = np.zeros_like(t)
        a = np.zeros_like(t)
        
        t_acc = ratio * total_time
        t_flat = total_time - 2 * t_acc
        if t_flat < 0: t_flat = 0
        
        # Avoid division by zero
        denom = (0.5 * t_acc + t_flat + 0.5 * t_acc) if profile_type == 'trapezoidal' else (t_flat + t_acc)
        v_max = total_dist / denom if denom > 0 else 0
        
        if profile_type == 'trapezoidal':
            for i, ti in enumerate(t):
                if ti < t_acc:
                    a[i] = v_max / t_acc
                    v[i] = a[i] * ti
                elif ti < t_acc + t_flat:
                    a[i] = 0
                    v[i] = v_max
                else:
                    a[i] = -v_max / t_acc
                    v[i] = v_max + a[i] * (ti - t_acc - t_flat)
                    
        elif profile_type == 's_curve':
            for i, ti in enumerate(t):
                if ti < t_acc:
                    v[i] = (v_max/2) * (1 - np.cos(np.pi * ti / t_acc))
                    a[i] = (v_max/2) * (np.pi/t_acc) * np.sin(np.pi * ti / t_acc)
                elif ti < t_acc + t_flat:
                    v[i] = v_max
                    a[i] = 0
                else:
                    t_dec = ti - (t_acc + t_flat)
                    if t_dec > t_acc: # Clamping
                        v[i] = 0; a[i] = 0
                    else:
                        v[i] = (v_max/2) * (1 + np.cos(np.pi * t_dec / t_acc))
                        a[i] = -(v_max/2) * (np.pi/t_acc) * np.sin(np.pi * t_dec / t_acc)
        
        m = 35.0 
        F = m * a + 10.0 * v
        torque_proxy = F * 0.5
        
        # Kt effective = 6.0 approx for leg
        power = np.abs(F * v) + (torque_proxy/6.0)**2 * 0.1
        
        return np.sum(power) * dt, np.max(np.abs(torque_proxy))

    base_sim_e, _ = simulate_1d(0.4, 'trapezoidal')
    scale_factor = base_energy / base_sim_e if base_sim_e > 0 else 1.0
    
    for r in ratios:
        e_t, tau_t = simulate_1d(r, 'trapezoidal')
        e_s, tau_s = simulate_1d(r, 's_curve')
        energies_trap.append(e_t * scale_factor)
        energies_scurve.append(e_s * scale_factor)
        peak_torques.append(tau_s)
        
    return ratios, energies_trap, energies_scurve

# =============================================================================
# 5. Visualization
# =============================================================================

def plot_results(p2_res, p3_res, opt_res):
    p2_e, p2_pow, p2_tau, p2_t = p2_res
    p3_e, p3_pow, p3_tau, p3_t = p3_res
    opt_r, opt_trap, opt_s = opt_res
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(p2_t, p2_pow, label=f'Walking (Total: {p2_e:.1f} J)', color='#1f77b4')
    ax1.plot(p3_t, p3_pow, label=f'Dancing (Total: {p3_e:.1f} J)', color='#ff7f0e', alpha=0.8)
    ax1.set_title("Instantaneous Power Consumption", fontsize=12, weight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Power (W)")
    ax1.legend()
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(opt_r, opt_trap, 'o--', label='Trapezoidal Profile', color='gray')
    ax2.plot(opt_r, opt_s, 'o-', label='S-Curve Profile (Smoothed)', color='green', linewidth=2)
    min_idx = np.argmin(opt_s)
    ax2.plot(opt_r[min_idx], opt_s[min_idx], 'r*', markersize=15, label='Optimal Point')
    ax2.set_title("Optimization: Velocity Profile Shape", fontsize=12, weight='bold')
    ax2.set_xlabel("Acceleration Time Ratio (t_acc / T_total)")
    ax2.set_ylabel("Estimated Total Energy (J)")
    ax2.legend()
    
    ax3 = fig.add_subplot(2, 1, 2)
    if len(p3_tau) > 0:
        tau_norm = np.linalg.norm(p3_tau, axis=1)
        ax3.fill_between(p3_t, tau_norm, color='#ff7f0e', alpha=0.3, label='Torque Norm (Dance)')
        ax3.plot(p3_t, tau_norm, color='#ff7f0e')
    ax3.set_title("Dance Dynamics: System Torque Norm (Indicator of Effort)", fontsize=12, weight='bold')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Torque Norm (Nm)")
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "04_energy_optimization.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Visualization] Saved to {save_path}")
    plt.show()

# =============================================================================
# 6. Main Execution
# =============================================================================

def main():
    print("=" * 60)
    print("APMCM 2025 — Problem 4: Energy Optimization")
    print("=" * 60)
    
    cfg_motor = ConfigLoader.load(FILE_MOTOR)
    p1_data = ConfigLoader.load(FILE_P1_RES)
    p2_data = ConfigLoader.load(FILE_P2_TRAJ)
    p3_data = ConfigLoader.load(FILE_P3_TRAJ)
    
    robot = RobotModel(FILE_URDF)
    energy_model = EnergyModel(cfg_motor)
    processor = SolutionProcessor(robot, energy_model)
    
    e1, p1_avg = processor.process_problem1(p1_data)
    res_p2 = processor.process_problem2(p2_data)
    res_p3 = processor.process_problem3(p3_data)
    
    opt_res = optimize_walking_profile(res_p2[0])
    
    total_e_base = e1 + res_p2[0] + res_p3[0]
    optimized_walking_e = min(opt_res[2]) if opt_res[2] else res_p2[0]
    total_e_opt = e1 + optimized_walking_e + res_p3[0]
    savings = (total_e_base - total_e_opt) / total_e_base * 100 if total_e_base > 0 else 0
    
    print("\n" + "="*60)
    print("FINAL ENERGY REPORT")
    print("="*60)
    print(f"1. Static Hold (P1, 2s): {e1:.2f} J")
    print(f"2. Walking (P2, 5s):     {res_p2[0]:.2f} J")
    print(f"3. Dance (P3, 4s):       {res_p3[0]:.2f} J")
    print("-" * 30)
    print(f"TOTAL BASELINE ENERGY:   {total_e_base:.2f} J")
    print("-" * 30)
    print(f"Optimized Walking Energy: {optimized_walking_e:.2f} J")
    print(f"TOTAL OPTIMIZED ENERGY:   {total_e_opt:.2f} J")
    print(f"ENERGY SAVINGS:           {savings:.2f} %")
    print("="*60)
    
    final_res = {
        "problem": "Problem 4",
        "baseline_energy_J": {
            "problem_1": e1,
            "problem_2": res_p2[0],
            "problem_3": res_p3[0],
            "total": total_e_base
        },
        "optimized_energy_J": {
            "total": total_e_opt,
            "improvement_percent": savings
        },
        "optimization_strategy": "S-Curve Velocity Profile with 30% acceleration phase"
    }
    
    with open(OUTPUT_DIR / "04_energy_optimization.json", "w") as f:
        json.dump(final_res, f, indent=4)
        
    plot_results(res_p2, res_p3, opt_res)

if __name__ == "__main__":
    main()