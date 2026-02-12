#!/usr/bin/env python

"""
Script to load and visualize the Acone robot in Mujoco simulator.
This script converts a URDF robot model to Mujoco's MJCF format and displays it in a 3D viewer.
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from lerobot.model.kinematics import RobotKinematics

import mujoco
import mujoco.viewer
import argparse
import numpy as np
import time

# Define username variable to allow easy switching between users
USERNAME = "gyh"  # Change this to your username as needed


def urdf_to_mjcf(urdf_path):
    """
    Convert a URDF file to MJCF format that Mujoco can load.
    This is a simplified conversion focusing on the kinematic structure.
    
    Args:
        urdf_path (str): Path to the input URDF file
    
    Returns:
        str: XML string representation of the MJCF model
    """
    # Read the URDF file
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # Parse the URDF - root element contains all robot definitions
    # Value during execution: Element tag='robot' attrib={'name': 'acone'}
    root = ET.fromstring(urdf_content)
    
    # Create the MJCF root element with the robot's name
    # Value during execution: Element tag='mujoco' attrib={'model': 'acone'}
    mjcf_root = ET.Element('mujoco')
    mjcf_root.set('model', root.get('name', 'converted_model'))
    
    # Add compiler options
    compiler_elem = ET.SubElement(mjcf_root, 'compiler')
    compiler_elem.set('autolimits', 'true')
    
    # Add default settings for joints and geometries to standardize behavior
    # ET.SubElement creates a new XML element and adds it as a child to the parent element
    # Value during execution: Element tag='default' inside mjcf_root
    default_elem = ET.SubElement(mjcf_root, 'default')
    # Creates a 'joint' element with attributes limited='true' and damping='1' inside default_elem
    joint_default = ET.SubElement(default_elem, 'joint', limited='true', damping='1')
    # Creates a 'geom' element with specified attributes inside default_elem
    geom_default = ET.SubElement(default_elem, 'geom', solref='0.01 1', solimp='0.8 0.9 0.001')
    
    # Add worldbody - this is the main container for all physical objects in the scene
    # ET.SubElement creates a new XML element and adds it as a child to the parent element
    worldbody = ET.SubElement(mjcf_root, 'worldbody')
    
    # Add light and floor for visualization
    # ET.SubElement creates a new XML element and adds it as a child to the parent element
    ET.SubElement(worldbody, 'light', pos='0 0 3', dir='0 0 -1', directional='true')
    # ET.SubElement creates a new XML element and adds it as a child to the parent element
    ET.SubElement(worldbody, 'geom', name='floor', type='plane', size='5 5 0.1', pos='0 0 -0.5')
    
    # Store all links and joints from URDF for easy lookup
    all_links = {}  # Dictionary mapping link names to link elements
    all_joints = {}  # Dictionary mapping joint names to joint elements
    
    for link in root.findall('link'):
        all_links[link.get('name')] = link
    
    for joint in root.findall('joint'):
        all_joints[joint.get('name')] = joint
    
    # Find the root link (one without a parent joint) - typically the base of the robot
    # Value during execution: {'left_link1', 'left_link2', ..., 'right_link11', 'right_link12', ...}
    child_links = set()  # Set of all links that are children in some joint
    for joint in all_joints.values():
        child_links.add(joint.find('child').get('link'))
    
    # Value during execution: 'base_link' (since it's not a child of any joint)
    root_link_name = None  # Name of the base/root link
    for link_name in all_links.keys():
        if link_name not in child_links:
            root_link_name = link_name
            break
    
    # Process the kinematic tree starting from the root link
    if root_link_name:
        # Create a body for the root link in the MJCF world
        # ET.SubElement creates a new XML element and adds it as a child to the parent element
        # Value during execution: Element tag='body' attrib={'name': 'base_link'}
        root_body = ET.SubElement(worldbody, 'body', name=root_link_name)
        
        # Add geometry for the root link - this defines its visual appearance and collision properties
        root_link = all_links[root_link_name]
        for visual in root_link.findall('visual'):
            # Value during execution: Element tag='visual' inside the root_link
            geometry = visual.find('geometry')
            # Value during execution: Element tag='geometry' inside the visual element
            mesh = geometry.find('mesh')
            # Value during execution: Element tag='mesh' with attrib like {'filename': '../meshes/base_link.STL'}
            if mesh is not None:
                # Value during execution: '../meshes/base_link.STL'
                mesh_filename = mesh.get('filename')
                # Adjust path relative to URDF file
                abs_mesh_path = os.path.join(os.path.dirname(urdf_path), mesh_filename.replace('package://acone/', ''))
                
                # Add the geometry to the body
                # ET.SubElement creates a new XML element and adds it as a child to the parent element
                geom_elem = ET.SubElement(root_body, 'geom', type='mesh', mesh=mesh_filename.split('/')[-1])
                
                # Value during execution: Element tag='material' if material is defined for the visual
                material = visual.find('material')
                if material is not None:
                    color = material.find('color')
                    if color is not None:
                        rgba = color.get('rgba')
                        geom_elem.set('rgba', rgba)
        
        # Recursively add child bodies and joints to build the complete kinematic chain
        add_child_bodies_and_joints(root_body, root_link_name, all_joints, all_links, urdf_path)
    
    # Add actuator section - defines how joints can be controlled
    # ET.SubElement creates a new XML element and adds it as a child to the parent element
    # Value during execution: Element tag='actuator' inside mjcf_root
    actuator = ET.SubElement(mjcf_root, 'actuator')
    for joint_name, joint in all_joints.items():
        # Value during execution: e.g., 'left_joint1', 'left_joint2', etc.
        joint_type = joint.get('type')  # Type of joint: revolute, prismatic, etc.
        # Value during execution: e.g., 'revolute', 'prismatic'
        if joint_type in ['revolute', 'continuous', 'prismatic']:
            # Add motor for each joint to enable control
            # ET.SubElement creates a new XML element and adds it as a child to the parent element
            # Increase gain parameters for better control response
            motor_elem = ET.SubElement(actuator, 'motor', joint=joint_name, gear='100', ctrlrange='-1 1', forcerange='-100 100', forcelimited='true')
    
    # Add meshes section - defines the mesh assets used in the model
    # ET.SubElement creates a new XML element and adds it as a child to the parent element
    # Value during execution: Element tag='asset' inside mjcf_root
    meshes = ET.SubElement(mjcf_root, 'asset')
    for link in all_links.values():
        # Value during execution: Element tag='link' with name like 'base_link', 'left_link1', etc.
        for visual in link.findall('visual'):
            geometry = visual.find('geometry')
            mesh = geometry.find('mesh')
            if mesh is not None:
                mesh_filename = mesh.get('filename')
                abs_mesh_path = os.path.join(os.path.dirname(urdf_path), mesh_filename.replace('package://acone/', ''))
                # Create mesh asset
                # Value during execution: 'base_link.STL', 'left_link1.STL', etc.
                mesh_name = mesh_filename.split('/')[-1]  # Extract just the filename
                # ET.SubElement creates a new XML element and adds it as a child to the parent element
                ET.SubElement(meshes, 'mesh', name=mesh_name, file=abs_mesh_path)
    
    # Convert to string
    # Value during execution: XML string representation of the entire MJCF model
    mjcf_str = ET.tostring(mjcf_root, encoding='unicode')
    
    # Format the XML properly
    import xml.dom.minidom
    # Value during execution: DOM document object containing the parsed XML
    dom = xml.dom.minidom.parseString('<root>' + mjcf_str + '</root>')
    # Value during execution: Pretty-formatted XML string of the MJCF model
    pretty_xml = dom.toprettyxml().replace('<root>', '').replace('</root>', '').strip()
    
    return pretty_xml


def add_child_bodies_and_joints(parent_body, parent_link_name, all_joints, all_links, urdf_path):
    """
    Recursively add child bodies and joints to the MJCF structure.
    
    Args:
        parent_body: Parent body element in the MJCF XML tree
        parent_link_name (str): Name of the parent link
        all_joints: Dictionary of all joints in the URDF
        all_links: Dictionary of all links in the URDF
        urdf_path (str): Path to the original URDF file
    """
    # Find all joints that have the parent_link as their parent
    for joint_name, joint in all_joints.items():
        parent_elem = joint.find('parent')
        if parent_elem is not None and parent_elem.get('link') == parent_link_name:
            child_elem = joint.find('child')
            if child_elem is not None:
                child_link_name = child_elem.get('link')
                
                # Get joint properties from URDF
                # Value during execution: e.g., 'revolute', 'prismatic'
                joint_type = joint.get('type')  # Type: revolute, prismatic, continuous, etc.
                # Value during execution: Element tag='origin' with attrib like {'xyz': '0.02 0 0.04', 'rpy': '0 0 0'}
                origin = joint.find('origin')  # Position and orientation of the joint
                # Value during execution: Element tag='axis' with attrib like {'xyz': '0 1 0'}
                axis = joint.find('axis')      # Axis of rotation/translation
                # Value during execution: Element tag='limit' with attrib like {'lower': '0', 'upper': '3.665', 'effort': '27', 'velocity': '5.5'} or None
                limit = joint.find('limit') if joint.find('limit') is not None else None  # Joint limits
                
                # Create the child body in MJCF
                # ET.SubElement creates a new XML element and adds it as a child to the parent element
                child_body = ET.SubElement(parent_body, 'body', name=child_link_name)
                
                # Set position and orientation of the child body relative to parent
                if origin is not None:
                    xyz = origin.get('xyz', '0 0 0')  # Position offset
                    rpy = origin.get('rpy', '0 0 0')  # Orientation (roll, pitch, yaw)
                    child_body.set('pos', xyz)
                    child_body.set('quat', rpy_to_quat(rpy))  # Convert RPY to quaternion
                
                # Add joint definition in MJCF
                if joint_type in ['revolute', 'continuous', 'prismatic']:
                    # ET.SubElement creates a new XML element and adds it as a child to the parent element
                    joint_elem = ET.SubElement(child_body, 'joint')
                    joint_elem.set('name', joint_name)  # Unique name for the joint
                    
                    # Map URDF joint types to MJCF joint types
                    if joint_type in ['revolute', 'continuous']:
                        joint_elem.set('type', 'hinge')  # Rotational joint in MJCF
                    elif joint_type == 'prismatic':
                        joint_elem.set('type', 'slide')  # Linear sliding joint in MJCF
                    
                    # Set the axis of motion for the joint
                    if axis is not None:
                        joint_elem.set('axis', axis.get('xyz', '0 0 1'))  # Direction vector
                    
                    # Apply joint limits if they exist
                    if limit is not None:
                        joint_elem.set('range', f"{limit.get('lower', '-1.57')} {limit.get('upper', '1.57')}")
                        joint_elem.set('limited', 'true')
                    
                    # Add effort and velocity limits if available
                    if limit is not None:
                        effort = limit.get('effort')  # Maximum force/torque
                        velocity = limit.get('velocity')  # Maximum velocity
                        if effort:
                            # Calculate armature value (rotor inertia) based on effort
                            joint_elem.set('armature', str(float(effort) * 0.01))
                        
                # Add geometry for the child link - defines visual and collision properties
                child_link = all_links[child_link_name]
                for visual in child_link.findall('visual'):
                    geometry = visual.find('geometry')
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_filename = mesh.get('filename')
                        abs_mesh_path = os.path.join(os.path.dirname(urdf_path), mesh_filename.replace('package://acone/', ''))
                        
                        # Add the geometry to the body
                        # ET.SubElement creates a new XML element and adds it as a child to the parent element
                        geom_elem = ET.SubElement(child_body, 'geom', type='mesh', mesh=mesh_filename.split('/')[-1])
                        
                        # Add material if available
                        material = visual.find('material')
                        if material is not None:
                            color = material.find('color')
                            if color is not None:
                                rgba = color.get('rgba')  # Red, green, blue, alpha values
                                geom_elem.set('rgba', rgba)
                
                # Recursively process children of this child to continue building the kinematic tree
                add_child_bodies_and_joints(child_body, child_link_name, all_joints, all_links, urdf_path)


def rpy_to_quat(rpy_str):
    """
    Convert roll-pitch-yaw angles to quaternion (simplified).
    This is a placeholder - in practice, you'd use proper RPY to quaternion conversion.
    
    Args:
        rpy_str (str): Space-separated string of roll, pitch, yaw angles in radians
    
    Returns:
        str: Space-separated string representing a quaternion (w, x, y, z)
    """
    # For simplicity, returning identity quaternion
    # In a real implementation, you'd convert RPY to quaternion properly
    return "1 0 0 0"


def main():
    """
    Main function that orchestrates the URDF to MJCF conversion and visualization process.
    """
    # Path to the acone URDF file - this is the input robot model
    urdf_path = f"/home/{USERNAME}/Workspace/lerobot/src/lerobot/model/acone/urdf/acone.urdf"
    
    if not os.path.exists(urdf_path):
        print(f"URDF file not found: {urdf_path}")
        return
    
    print("Converting URDF to MJCF...")
    # Convert the URDF robot model to Mujoco's MJCF format
    # Value during execution: XML string representation of the MJCF model
    mjcf_xml = urdf_to_mjcf(urdf_path)
    
    # Write the MJCF to a temporary file for Mujoco to load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp_file:
        tmp_file.write(mjcf_xml)
        # Value during execution: e.g., '/tmp/tmp4pgw4ccs.xml'
        temp_mjcf_path = tmp_file.name  # Path to the temporary MJCF file
    
    print(f"Temporary MJCF file created: {temp_mjcf_path}")
    
    try:
        # Load the model in Mujoco - creates the physics simulation environment
        print("Loading model in Mujoco...")
        # Value during execution: MjModel object with properties like nq (degrees of freedom), nbody (number of bodies), etc.
        model = mujoco.MjModel.from_xml_path(temp_mjcf_path)  # MjModel contains the robot definition
        
        # Create data object - stores the dynamic state of the simulation (positions, velocities, etc.)
        # Value during execution: MjData object associated with the model, containing qpos, qvel, etc.
        data = mujoco.MjData(model)  # MjData contains the state of the model at a specific time
        print("Mujoco data:", data)

        print("Launching Mujoco viewer...")
        # Launch the viewer to visualize the robot in 3D
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Viewer launched successfully. Close the window to exit.")
            print("Using joint position control by default. Press Ctrl+C to exit.")
            
            import numpy as np
            import time
            
            # Control mode selection (default is joint position)
            control_mode = "end_effector"  # Options: "joint_pos", "joint_vel", "torque", "end_effector"
            
            # Define reasonable ranges for random sampling based on robot model
            joint_limits_low = np.full(model.nu, -np.pi)   # Lower limits for joints
            joint_limits_high = np.full(model.nu, np.pi)  # Upper limits for joints
            
            # For prismatic joints, use linear limits
            for i in range(model.nu):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint{i+1}" if i < 8 else f"right_joint{i-7 if i > 7 else i+11}")
                if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE:
                    joint_limits_low[i] = 0.5
                    joint_limits_high[i] = 0.5  # Based on the URDF limits for prismatic joints
            
            prev_time = time.time()
            control_update_interval = 0.1  # Update control every 0.1 seconds
            
            while viewer.is_running():
                current_time = time.time()
                
                # Update control input periodically
                if current_time - prev_time > control_update_interval:
                    if control_mode == "joint_pos":
                        # Randomly sample joint positions within reasonable limits
                        data.ctrl[:] = np.random.uniform(low=joint_limits_low, high=joint_limits_high, size=model.nu)
                    elif control_mode == "joint_vel":
                        # Randomly sample joint velocities within reasonable limits
                        vel_range = np.full(model.nu, 1.0)  # Max 1 rad/s for revolute, 0.1 m/s for prismatic
                        for i in range(model.nu):
                            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint{i+1}" if i < 8 else f"right_joint{i-7 if i > 7 else i+11}")
                            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE:
                                vel_range[i] = 0.01  # Smaller range for prismatic joints
                        data.ctrl[:] = np.random.uniform(low=-vel_range, high=vel_range, size=model.nu)
                    elif control_mode == "torque":
                        # Randomly sample torques within reasonable limits
                        torque_range = np.full(model.nu, 10.0)  # Max 10 Nm for most joints
                        # Reduce torque for gripper joints
                        for i in range(model.nu):
                            if i >= 6 and i <= 7 or i >= 15 and i <= 16:  # Gripper joints
                                torque_range[i] = 5.0  # Lower torque for grippers
                        data.ctrl[:] = np.random.uniform(low=-torque_range, high=torque_range, size=model.nu)
                    elif control_mode == "end_effector":
                        print("inside end effector control..")
                        # End-effector control using position (x,y,z) and orientation (roll, pitch, yaw)
                        # Predefined test poses for visualization
                        # Define several fixed poses to cycle through for testing
                        test_poses = [
                            # Pose 1: Front center
                            ([0.3, 0.0, 0.5], [0, 0, 0]),
                            # Pose 2: Front left
                            ([0.3, -0.2, 0.5], [0, 0, np.pi/4]),
                            # Pose 3: Front right
                            ([0.3, 0.2, 0.5], [0, 0, -np.pi/4]),
                            # Pose 4: Higher position
                            ([0.3, 0.0, 0.7], [np.pi/2, 0, 0]),
                            # Pose 5: Lower position
                            ([0.3, 0.0, 0.3], [-np.pi/2, 0, 0]),
                            # Pose 6: Back center
                            ([0.0, 0.0, 0.5], [0, np.pi/4, 0]),
                        ]
                        
                        # Use a single fixed pose for better control stability
                        pose_idx = 0  # Fixed pose for stability testing
                        ee_pos, ee_rpy = test_poses[pose_idx]
                        
                        # Alternative: Use a single fixed pose for testing
                        # ee_pos = [0.3, 0.0, 0.5]  # Position in meters
                        # ee_rpy = [0, 0, 0]       # Orientation in radians
                        
                        # Convert RPY to rotation matrix
                        cr, cp, cy = np.cos(ee_rpy)
                        sr, sp, sy = np.sin(ee_rpy)
                        
                        # Rotation matrix from RPY (ZYX Euler angles)
                        R = np.array([
                            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                            [-sp, cp*sr, cp*cr]
                        ])
                        
                        # Create 4x4 transformation matrix for target end-effector pose
                        target_pose = np.eye(4)
                        target_pose[:3, :3] = R
                        target_pose[:3, 3] = ee_pos
                            
                        # Use the RobotKinematics class from kinematics.py to solve IK
                        try:
                            # Initialize the kinematics solver with the acone URDF
                            urdf_path = f"/home/{USERNAME}/Workspace/lerobot/src/lerobot/model/acone/urdf/acone.urdf"

                            # For this example, we'll use the left arm - specify the end-effector frame name
                            # Note: The actual frame name needs to match what's defined in the URDF
                            kinematics_solver = RobotKinematics(
                                urdf_path=urdf_path,
                                target_frame_name="left_link8",  # Assuming this is the end-effector frame
                                joint_names=None  # Use all joints
                            )

                            # Get current joint positions from the simulation
                            # Use the number of joints from the kinematics solver
                            num_joints = len(kinematics_solver.joint_names)
                            current_joint_pos = data.qpos[:num_joints] if len(data.qpos) >= num_joints else np.zeros(num_joints)
                            
                            # Log current joint positions to file
                            with open(f'/home/{USERNAME}/Workspace/lerobot/src/lerobot/model/kinematics_log.txt', 'a') as log_file:
                                log_file.write(f"\ncurrent_joint_pos: {current_joint_pos}\n")
                            
                            # Solve inverse kinematics to get joint angles for the target pose
                            # Note: This assumes the kinematics solver can handle the conversion properly
                            target_joint_pos = kinematics_solver.inverse_kinematics(
                                current_joint_pos=current_joint_pos,
                                desired_ee_pose=target_pose
                            )

                            # Update control values with IK solution using PID-like control
                            all_ctrl_values = np.copy(data.ctrl)  # Start with current control values

                            # Apply the computed joint positions from IK with PID control
                            # This implements a closed-loop controller to drive joints toward target positions
                            num_ik_joints = min(len(target_joint_pos), len(all_ctrl_values))

                            # Implement a simple proportional controller to drive joints toward target
                            # This helps joints converge to target positions over time
                            kp = 0.1  # Proportional gain - adjust to control convergence speed
                            for i in range(num_ik_joints):
                                # Calculate error between current joint position (from simulation) and target
                                error = target_joint_pos[i] - current_joint_pos[i]
                                # Apply proportional control to determine control signal
                                all_ctrl_values[i] = current_joint_pos[i] + kp * error

                        except ImportError:
                            # If kinematics module is not available, fall back to random joint positions
                            print("Kinematics module not available, using random joint positions")
                            all_ctrl_values = np.random.uniform(low=joint_limits_low, high=joint_limits_high, size=model.nu)
                        except Exception as e:
                            # If IK fails, fall back to random joint positions
                            print(f"IK failed: {e}, using random joint positions")
                            all_ctrl_values = np.random.uniform(low=joint_limits_low, high=joint_limits_high, size=model.nu)
                        
                        # Set gripper joints to control opening/closing
                        # Random gripper opening (0.001 to 0.044 range based on URDF limits)
                        # left_gripper_pos = np.random.uniform(0.01, 0.04)  # Open gripper
                        # right_gripper_pos = np.random.uniform(0.01, 0.04)  # Open gripper
                        left_gripper_pos = 0.002
                        right_gripper_pos = 0.002
                        
                        # Set gripper positions in the control array
                        # For the acone robot: left gripper joints are at indices 6,7 and right gripper at indices 14,15
                        all_ctrl_values[6] = left_gripper_pos  # left_joint7 (left gripper)
                        all_ctrl_values[7] = left_gripper_pos  # left_joint8 (left gripper)
                        all_ctrl_values[14] = right_gripper_pos  # right_joint17 (right gripper)
                        all_ctrl_values[15] = right_gripper_pos  # right_joint18 (right gripper)
                        
                        # Log control values to file
                        with open(f'/home/{USERNAME}/Workspace/lerobot/src/lerobot/model/kinematics_log.txt', 'a') as log_file:
                            log_file.write(f"all_ctrl_values: {all_ctrl_values}\n")
                        data.ctrl[:] = all_ctrl_values

                        # NOTE: In a real implementation, you would:
                        # 1. Use the robot's FK to get current EE pose
                        # 2. Compute pose error
                        # 3. Use Jacobian transpose/pseudoinverse or optimization-based IK
                        # 4. Apply computed joint angles to data.ctrl[:]

                        # For gripper control specifically:
                        # - The prismatic joints (typically the last few joints) control gripper opening/closing
                        # - Positive values increase separation (open gripper)
                        # - Negative values decrease separation (close gripper)
                        # - Values near 0.044 fully open, values near 0 fully closed

                    prev_time = current_time

                # Print L2 norms for debugging convergence
                num_joints = min(len(data.qpos), len(data.ctrl))
                current_pos = data.qpos[:num_joints]
                current_ctrl = data.ctrl[:num_joints]
                
                l2_norm_current = np.linalg.norm(current_pos)
                l2_norm_ctrl = np.linalg.norm(current_ctrl)
                diff_norm = np.linalg.norm(current_pos - current_ctrl)

                with open(f'/home/{USERNAME}/Workspace/lerobot/src/lerobot/model/kinematics_log.txt', 'a') as log_file:
                    log_file.write(f"\nL2 norm of current joint pos: {l2_norm_current:.6f}, L2 norm of ctrl: {l2_norm_ctrl:.6f}, Diff norm: {diff_norm:.6f}")

                # Step the simulation forward by one timestep
                # This updates the physics state based on the model dynamics
                # print(data.ctrl[:])
                mujoco.mj_step(model, data)  # Performs one step of the physics simulation
                viewer.sync()  # Synchronize the viewer with the simulation state
                
    except Exception as e:
        print(f"Error launching viewer: {e}")
        # Also try to load without visualization to check if the model loads correctly
        try:
            print("Trying to load model without visualization...")
            model = mujoco.MjModel.from_xml_path(temp_mjcf_path)
            print(f"Model loaded successfully with {model.nq} degrees of freedom")
            print(f"Number of bodies: {model.nbody}")
            print(f"Number of joints: {model.njnt}")
        except Exception as e2:
            print(f"Failed to load model even without visualization: {e2}")
    finally:
        # Clean up the temporary file to avoid cluttering the system
        # temp_mjcf_path: Path to the temporary MJCF file created earlier
        if os.path.exists(temp_mjcf_path):
            os.remove(temp_mjcf_path)
            print(f"Temporary file removed: {temp_mjcf_path}")


if __name__ == "__main__":
    main()