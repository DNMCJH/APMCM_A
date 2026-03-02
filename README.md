# Unitree G1 Humanoid Robot Dynamics and Energy Optimization

## Overview

This project addresses the challenge of balancing **high-dynamic motion** with **energy efficiency** for the Unitree G1 humanoid robot.  
We integrate **rigid body kinematics**, **dynamics**, and **nonlinear optimization** to develop a comprehensive framework for **pose verification**, **motion control**, and **energy management**.

---

## Problem Statements

1. **Pose Verification (Problem 1)**  
   - **Objective:** Compute the left-arm forward kinematics and required torque for a target pose.  
   - **Approach:**  
     - Modified Denavit-Hartenberg (D-H) parameters for arm kinematics.  
     - Analytical calculation of end-effector coordinates.  
     - Static torque computation considering gravity and Coulomb friction.  
   - **Results:**  
     - Target pose (forward 60°, left 30°) → coordinates: `(253.50, 146.36, 169.00) mm`.  
     - Required torque: `4.88 Nm` < rated `8.0 Nm`, confirming a high safety margin.

2. **Walking Trajectory Planning (Problem 2)**  
   - **Objective:** Execute a 10 m walking task at 2 m/s while analyzing joint dynamics.  
   - **Approach:**  
     - Linear Inverted Pendulum Model (LIPM) with Trapezoidal Velocity Planning.  
     - Differential analysis to locate global maximum knee angular velocity.  
   - **Results:**  
     - Task completes in 5.0 s.  
     - Max knee velocity occurs at `t = 3.12 s`, matching biomechanical expectations.

3. **Whole-Body Dance Control (Problem 3)**  
   - **Objective:** Enable stable, coordinated dance movements.  
   - **Approach:**  
     - Whole-Body Control (WBC) based on Task Space Decoupling.  
     - Quintic polynomial interpolation for torso S-shaped rotation.  
     - Anti-phase circular arm trajectories with knee flexion compensation.  
   - **Results:**  
     - Stable dance motions within `t ∈ [0,4] s`.  
     - Zero Moment Point (ZMP) maintained in double-support region.

4. **Energy Optimization (Problem 4)**  
   - **Objective:** Reduce total energy consumption while maintaining performance.  
   - **Approach:**  
     - Generalized energy model including mechanical work and heat losses.  
     - Pinocchio dynamics library for baseline computation.  
     - S-Curve Velocity Planning to smooth accelerations and suppress torque spikes.  
   - **Results:**  
     - Baseline energy: `5076.22 J`.  
     - Optimized energy: `4216.15 J` → **16.94% reduction**.

---

## Project Structure

```

APMCM_E/
├─ config/       # Robot parameters, motion constraints
├─ data/         # Input/output data, trajectories
├─ scripts/      # Problem-specific scripts for kinematics, walking, dance, energy analysis
└─ Aapmcm25300077.pdf  # Report

````

---

## Dependencies

- Python 3.x  
- Pinocchio dynamics library  
- Numpy, Scipy, Matplotlib  
- Optional: Jupyter Notebook for visualization

---

## How to Run

1. Navigate to the project folder:

```bash
cd ~/Desktop/APMCM_E
````

2. Install dependencies:

```bash
pip install pinocchio numpy scipy matplotlib
```

3. Run scripts:

```bash
python scripts/01_problem1_kinematics.py
python scripts/02_problem2_walking.py
python scripts/03_problem3_dance.py
python scripts/04_problem4_energy.py
```

4. Check outputs in `data/output/`.

---

## Key Takeaways

* Integrates **kinematics, dynamics, and optimization** for humanoid robot motion.
* Demonstrates **safe torque margins**, **accurate walking trajectories**, and **stable whole-body dance control**.
* Achieves **energy optimization** with smooth velocity planning, improving robot endurance.

---

## License

MIT License
