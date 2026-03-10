# MuJoCo End-Effector Control for Acone Robot

This project demonstrates end-effector control for the Acone robot using inverse kinematics with the MuJoCo physics simulator.

## Overview

This implementation includes:
- Loading the Acone robot model from URDF
- Creating a MuJoCo simulation environment
- Implementing Jacobian-based inverse kinematics
- Controlling the end-effector to follow a desired trajectory
- Visualization in the MuJoCo viewer

## Files

- `final_mujoco_end_effector_control.py`: Main program demonstrating end-effector control
- `mujoco_end_effector_control_advanced.py`: Advanced version with improved IK algorithm
- `mujoco_end_effector_control.py`: Basic version of the controller

## Features

- **URDF Integration**: Converts URDF robot description to MuJoCo-compatible format
- **Inverse Kinematics**: Uses Jacobian transpose and damped least squares methods
- **Trajectory Following**: Controls the end-effector to follow a circular trajectory
- **Real-time Visualization**: Displays the robot in MuJoCo viewer
- **Dual Arm Control**: Supports both left and right arms of the Acone robot

## Requirements

- Python 3.10+
- MuJoCo
- NumPy
- SciPy

## Usage

```bash
python final_mujoco_end_effector_control.py
```

The program will:
1. Load the Acone robot model
2. Initialize the simulation
3. Control the left arm end-effector to follow a circular trajectory
4. Display real-time information in the console
5. Show the robot in the MuJoCo viewer

Press ESC in the viewer window to exit the simulation.

## Algorithm Details

The inverse kinematics implementation uses:
- Jacobian computation for relating joint velocities to end-effector velocities
- Damped least squares method for numerical stability
- Iterative approach to converge to target pose
- Separate handling of position and orientation errors

## Notes

- The robot model is simplified using basic geometric shapes instead of complex meshes
- Joint limits and physical properties are approximated from the URDF
- The control algorithm may need tuning depending on the specific robot configuration