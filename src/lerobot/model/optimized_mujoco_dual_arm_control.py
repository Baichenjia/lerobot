"""
Optimized MuJoCo End-Effector Control for Acone Robot

This script implements an optimized inverse kinematics solution with better convergence.
"""

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

# cache for actuated joint info keyed by model id (models are C structs so
# we avoid attaching attributes to them)
_act_cache = {}


def rpy_to_quat(rpy_str):
    """Convert RPY string to quaternion string."""
    rpy = [float(x) for x in rpy_str.split()]
    roll, pitch, yaw = rpy
    
    # Calculate quaternion from RPY
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return f"{w} {x} {y} {z}"


def add_child_bodies_and_joints(parent_body, parent_link_name, all_joints, all_links, urdf_path):
    """Recursively add child bodies and joints to the MJCF structure."""
    for joint_name, joint in all_joints.items():
        parent_elem = joint.find('parent')
        if parent_elem is not None and parent_elem.get('link') == parent_link_name:
            child_elem = joint.find('child')
            if child_elem is not None:
                child_link_name = child_elem.get('link')

                # Get joint properties from URDF
                joint_type = joint.get('type')
                origin = joint.find('origin')
                axis = joint.find('axis')
                limit = joint.find('limit') if joint.find('limit') is not None else None

                # Create the child body in MJCF
                child_body = ET.SubElement(parent_body, 'body', name=child_link_name)

                # Set position and orientation of the child body relative to parent
                if origin is not None:
                    xyz = origin.get('xyz', '0 0 0')
                    rpy = origin.get('rpy', '0 0 0')
                    child_body.set('pos', xyz)
                    child_body.set('quat', rpy_to_quat(rpy))

                # Add joint definition in MJCF
                if joint_type in ['revolute', 'continuous', 'prismatic']:
                    joint_elem = ET.SubElement(child_body, 'joint')
                    joint_elem.set('name', joint_name)

                    # Map URDF joint types to MJCF joint types
                    if joint_type in ['revolute', 'continuous']:
                        joint_elem.set('type', 'hinge')
                    elif joint_type == 'prismatic':
                        joint_elem.set('type', 'slide')

                    # Set the axis of motion for the joint
                    if axis is not None:
                        joint_elem.set('axis', axis.get('xyz', '0 0 1'))

                    # Apply joint limits if they exist
                    if limit is not None:
                        joint_elem.set('range', f"{limit.get('lower', '-1.57')} {limit.get('upper', '1.57')}")
                        joint_elem.set('limited', 'true')

                    # Add effort and velocity limits if available
                    if limit is not None:
                        effort = limit.get('effort')
                        if effort:
                            joint_elem.set('armature', str(float(effort) * 0.01))

                # Add geometry for the child link
                child_link = all_links[child_link_name]
                for visual in child_link.findall('visual'):
                    geometry = visual.find('geometry')
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_filename = mesh.get('filename')
                        abs_mesh_path = os.path.join(os.path.dirname(urdf_path), mesh_filename.replace('package://acone/', ''))

                        # Add the geometry to the body
                        geom_elem = ET.SubElement(child_body, 'geom', type='mesh', mesh=mesh_filename.split('/')[-1])

                        # Add material if available
                        material = visual.find('material')
                        if material is not None:
                            color = material.find('color')
                            if color is not None:
                                rgba = color.get('rgba')
                                geom_elem.set('rgba', rgba)

                # Recursively process children of this child
                add_child_bodies_and_joints(child_body, child_link_name, all_joints, all_links, urdf_path)


def create_model_from_urdf(urdf_path):
    """Create a MuJoCo model from the URDF file preserving the dual-arm structure."""
    # Read the URDF file
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    # Parse the URDF
    root = ET.fromstring(urdf_content)

    # Create the MJCF root element with the robot's name
    mjcf_root = ET.Element('mujoco')
    mjcf_root.set('model', root.get('name', 'converted_model'))

    # Add compiler options
    compiler_elem = ET.SubElement(mjcf_root, 'compiler')
    compiler_elem.set('autolimits', 'true')
    compiler_elem.set('angle', 'radian')

    # Add default settings for joints and geometries
    default_elem = ET.SubElement(mjcf_root, 'default')
    joint_default = ET.SubElement(default_elem, 'joint', limited='true', damping='0.1')
    geom_default = ET.SubElement(default_elem, 'geom', solref='0.01 1', solimp='0.8 0.9 0.001', condim='4')

    # Add option settings
    option_elem = ET.SubElement(mjcf_root, 'option')
    option_elem.set('timestep', '0.005')
    option_elem.set('iterations', '50')
    option_elem.set('solver', 'Newton')
    option_elem.set('tolerance', '1e-10')

    # Add worldbody - this is the main container for all physical objects in the scene
    worldbody = ET.SubElement(mjcf_root, 'worldbody')

    # Add light and floor for visualization
    ET.SubElement(worldbody, 'light', pos='0 0 3', dir='0 0 -1', directional='true')
    ET.SubElement(worldbody, 'geom', name='floor', type='plane', size='5 5 0.1', pos='0 0 0', rgba='.9 .9 .9 1')

    # Store all links and joints from URDF for easy lookup
    all_links = {}
    all_joints = {}

    for link in root.findall('link'):
        all_links[link.get('name')] = link

    for joint in root.findall('joint'):
        all_joints[joint.get('name')] = joint

    # Find the root link (one without a parent joint)
    child_links = set()
    for joint in all_joints.values():
        child_elem = joint.find('child')
        if child_elem is not None:
            child_links.add(child_elem.get('link'))

    root_link_name = None
    for link_name in all_links.keys():
        if link_name not in child_links:
            root_link_name = link_name
            break

    # Process the kinematic tree starting from the root link
    if root_link_name:
        # Create a body for the root link in the MJCF world
        root_body = ET.SubElement(worldbody, 'body', name=root_link_name)

        # Add geometry for the root link
        root_link = all_links[root_link_name]
        for visual in root_link.findall('visual'):
            geometry = visual.find('geometry')
            mesh = geometry.find('mesh')
            if mesh is not None:
                mesh_filename = mesh.get('filename')
                abs_mesh_path = os.path.join(os.path.dirname(urdf_path), mesh_filename.replace('package://acone/', ''))

                # Add the geometry to the body
                geom_elem = ET.SubElement(root_body, 'geom', type='mesh', mesh=mesh_filename.split('/')[-1])

                # Add material if available
                material = visual.find('material')
                if material is not None:
                    color = material.find('color')
                    if color is not None:
                        rgba = color.get('rgba')
                        geom_elem.set('rgba', rgba)

        # Recursively add child bodies and joints to build the complete kinematic chain
        add_child_bodies_and_joints(root_body, root_link_name, all_joints, all_links, urdf_path)

    # Add actuator section - defines how joints can be controlled
    actuator = ET.SubElement(mjcf_root, 'actuator')
    for joint_name, joint in all_joints.items():
        joint_type = joint.get('type')
        if joint_type in ['revolute', 'continuous', 'prismatic']:
            # Add motor for each joint to enable control
            motor_elem = ET.SubElement(actuator, 'motor', joint=joint_name, gear='100', ctrlrange='-1 1')

    # Add meshes section - defines the mesh assets used in the model
    meshes = ET.SubElement(mjcf_root, 'asset')
    for link in all_links.values():
        for visual in link.findall('visual'):
            geometry = visual.find('geometry')
            mesh = geometry.find('mesh')
            if mesh is not None:
                mesh_filename = mesh.get('filename')
                abs_mesh_path = os.path.join(os.path.dirname(urdf_path), mesh_filename.replace('package://acone/', ''))
                # Create mesh asset
                mesh_name = mesh_filename.split('/')[-1]
                ET.SubElement(meshes, 'mesh', name=mesh_name, file=abs_mesh_path)

    # Convert to string
    mjcf_str = ET.tostring(mjcf_root, encoding='unicode')

    # Format the XML properly
    dom = minidom.parseString('<root>' + mjcf_str + '</root>')
    pretty_xml = dom.toprettyxml().replace('<root>', '').replace('</root>', '').strip()

    # Write temporary MJCF file
    temp_mjcf_path = "/tmp/acone_optimized.xml"
    with open(temp_mjcf_path, 'w') as f:
        f.write(pretty_xml)
    
    # Load the model
    model = mujoco.MjModel.from_xml_path(temp_mjcf_path)
    return model


def get_end_effector_pose(model, data, body_name):
    """Get the pose of the end effector."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    # Position
    pos = data.xpos[body_id].copy()  # pos shape: (3,), dtype: float64  # (x, y, z)
    # Orientation (rotation matrix to quaternion)
    rot_matrix = data.xmat[body_id].copy().reshape(3, 3)  # rot_matrix shape: (3, 3), dtype: float64
    quat = R.from_matrix(rot_matrix).as_quat()  # quat shape: (4,), dtype: float64  # (x, y, z, w) or (qx, qy, qz, qw) depending on scipy
    return pos, quat  # returns: pos (3,), quat (4,)


def get_actuated_indices(model):
    """Return joint ids plus qpos/qvel addresses for actuated joints.

    We need the joint ids later when clamping to ``model.jnt_range``.  The
    tuple returned is ``(jids, qpos_addrs, qvel_addrs)``.
    """
    act_jids = []
    act_qpos = []
    act_qvel = []
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            act_jids.append(jid)
            act_qpos.append(model.jnt_qposadr[jid])
            act_qvel.append(model.jnt_dofadr[jid])
    # returns: act_jids (n_act,), act_qpos (n_act,) indices into data.qpos, act_qvel (n_act,) indices into data.qvel
    return (
        np.array(act_jids, dtype=int),  # (n_act,) joint ids
        np.array(act_qpos, dtype=int),  # (n_act,) qpos addresses
        np.array(act_qvel, dtype=int),  # (n_act,) qvel/dof addresses
    )


def get_actuated_indices_by_prefix(model, prefix):
    """Return actuated joint ids and qpos/qvel addresses for joints whose
    names start with ``prefix`` (e.g. 'left_' or 'right_').
    """
    act_jids = []
    act_qpos = []
    act_qvel = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        if not name.startswith(prefix):
            continue
        jtype = model.jnt_type[jid]
        if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            act_jids.append(jid)
            act_qpos.append(model.jnt_qposadr[jid])
            act_qvel.append(model.jnt_dofadr[jid])
    # returns arrays shaped (n_pref_act,)
    return np.array(act_jids, dtype=int), np.array(act_qpos, dtype=int), np.array(act_qvel, dtype=int)


def compute_jacobian(model, data, body_name, act_qvel=None):
    """Compute the Jacobian for the end effector.

    If ``act_qvel`` is provided the returned matrix contains only columns for
    the actuated degrees of freedom.  Otherwise the pair ``(jacp, jacr)`` is
    returned exactly as before to preserve backwards compatibility with other
    callers (there are none in this file, but it makes the API clearer).
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # full jacobian for all velocity DOFs
    # jacp: (3, nv) linear velocity partials
    # jacr: (3, nv) angular velocity partials
    jacp = np.zeros((3, model.nv))  # (3, nv)
    jacr = np.zeros((3, model.nv))  # (3, nv)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    if act_qvel is not None:
        # stacked full Jacobian: (6, nv) -> select columns for actuated dofs
        full = np.vstack([jacp, jacr])  # (6, nv)
        # returned: (6, n_act)
        return full[:, act_qvel]
    else:
        # return separate position and rotation Jacobians
        return jacp, jacr


def inverse_kinematics_position_control(model, data, target_pos, target_rpy, body_name, max_iter=50, tol=0.1, joint_prefix=None, pos_weight=1.0, rot_weight=0.6, svd_alpha=0.1, min_lambda=1e-4, max_lambda=1.0):
    """
    优化的逆运动学实现，使用自适应参数以获得更好的收敛性。

    *只对被驱动的关节计算增量*，避免改变浮动基础等非控制自由度。
    参考了 ``simple_mujoco_ik.py`` 中的做法并且添加了自适应阻尼/步长。
    """
    # choose actuated joints: either whole model or restricted by prefix
    mid = id(model)
    if joint_prefix is None:
        if mid not in _act_cache:
            _act_cache[mid] = get_actuated_indices(model)
        act_jids, act_qpos, act_qvel = _act_cache[mid]
    else:
        # do not cache prefix-specific lists for simplicity
        act_jids, act_qpos, act_qvel = get_actuated_indices_by_prefix(model, joint_prefix)

    # 将目标RPY欧拉角转换为四元数表示
    # target_pos: (3,) target position [x,y,z]
    # target_rpy: (3,) target euler angles [roll,pitch,yaw]
    target_quat = R.from_euler('xyz', target_rpy).as_quat()  # target_quat shape: (4,), dtype: float64
    # debug: initial pose and error
    init_pos, init_quat = get_end_effector_pose(model, data, body_name)
    init_pos_err = np.linalg.norm(target_pos - init_pos)
    # 初始状态打印（可注释以减少输出量）
    # print(f"IK start for {body_name}: init_pos={init_pos}, target={target_pos}, pos_err={init_pos_err:.4f}")
    
    # 以下为中文注释的关键逆运动学步骤说明：
    # 1) 计算末端执行器当前位置与目标位置/姿态之间的误差（位置误差与旋转误差）。
    # 2) 使用雅可比矩阵将任务空间误差映射到关节空间（dq 的估计）。
    # 3) 采用阻尼最小二乘（Damped Least Squares, DLS）来解决雅可比矩阵病态或奇异时的不稳定性，
    #    这里通过对加权雅可比进行 SVD 分解得到最小奇异值，并基于该值自适应设置阻尼因子 λ。
    # 4) 在计算得到的关节增量上应用小步长与逐关节截断，防止每次更新过大导致振荡或越界。
    # 5) 仅更新被驱动的关节（通过 jnt_qposadr/jnt_dofadr 映射），并在每次迭代后调用 mj_forward 以刷新 kinematics。
    for iteration in range(max_iter):
        # 获取当前末端执行器的姿态（位置和姿态）
        # current_pos: (3,), current_quat: (4,)
        current_pos, current_quat = get_end_effector_pose(model, data, body_name)  # (3,), (4,)

        # 计算位置误差：目标位置 - 当前位置
        # 目的：量化末端执行器在笛卡尔空间中的位置偏差
        pos_error = target_pos - current_pos  # pos_error shape: (3,), dtype: float64  # (dx, dy, dz)
        
        # 计算姿态误差：使用四元数差值来表示旋转误差
        # 原理：R.from_quat(target_quat).inv() * R.from_quat(current_quat) 得到从当前姿态到目标姿态的旋转变化
        # 目的：量化末端执行器在姿态上的偏差
        q_diff = R.from_quat(target_quat).inv() * R.from_quat(current_quat)  # q_diff: Rotation object
        
        # 将四元数差值转换为角速度向量（轴角表示）
        # 原理：as_rotvec() 将四元数转换为轴角表示，向量方向表示旋转轴，长度表示旋转角度
        # 目的：将姿态误差转换为可以直接用于控制的向量形式
        rot_error = q_diff.as_rotvec()  # rot_error shape: (3,), dtype: float64  # axis-angle (rx,ry,rz)

        # 合并位置误差和姿态误差到一个向量中 [dx, dy, dz, rx, ry, rz]
        # 目的：统一处理位置和姿态误差，形成6自由度的任务空间误差
        error = np.hstack([pos_error, rot_error])  # error shape: (6,), dtype: float64  # [dx,dy,dz,rx,ry,rz]
        error_norm = np.linalg.norm(error)  # error_norm: scalar, dtype: float64

        # 检查收敛条件：如果误差小于容差阈值，则认为已收敛
        # error_norm: scalar
        if error_norm < tol:
            print(f"IK converged in {iteration+1} iterations with error {error_norm:.6f}")
            return error_norm

        # conservative per-arm damped least squares solver (column-limited)
        # This is intentionally simple and uses small steps to avoid
        # overshoot and oscillation observed with larger steps.
        # J: (6, n_act) -- columns correspond to actuated DOFs (act_qvel)
        J = compute_jacobian(model, data, body_name, act_qvel)  # (6, n_act)
        # debug: jacobian singular values (only on first iteration)
        if iteration == 0:
            try:
                s = np.linalg.svd(J, compute_uv=False)
                print(f"  J shape={J.shape}, sv_min={s[-1]:.6f}, sv_max={s[0]:.6f}")
            except Exception:
                pass

        # zero actuated velocities for stability before applying qpos changes
        try:
            # zero actuated velocities for stability before applying qpos changes
            # data.qvel shape: (nv,), act_qvel indices into qvel (n_act,)
            data.qvel[act_qvel] = 0.0
        except Exception:
            pass

        # weighting of pos/rot error
        # W: (6,6) weighting matrix for pos/rot
        W = np.diag([pos_weight, pos_weight, pos_weight, rot_weight, rot_weight, rot_weight])  # (6,6)
        # weighted error and jacobian
        err_w = W @ error  # (6,)
        Jw = W @ J  # (6, n_act)

        # adaptive damping using Jacobian SVD: lam = svd_alpha / (sv_min + eps)
        # adaptive damping using Jacobian SVD: lam = svd_alpha / (sv_min + eps)
        eps = 1e-8
        try:
            s = np.linalg.svd(Jw, compute_uv=False)
            sv_min = float(s[-1]) if s.size > 0 else 0.0
        except Exception:
            sv_min = 0.0
        lam = svd_alpha / (sv_min + eps)
        lam = float(max(min_lambda, min(max_lambda, lam)))
        # JJ: (6,6) regularized task-space matrix
        JJ = Jw @ Jw.T + (lam * lam) * np.eye(6)  # (6,6)
        try:
            y = np.linalg.solve(JJ, err_w)  # y: (6,)
            joint_delta = Jw.T @ y  # joint_delta: (n_act,)
        except np.linalg.LinAlgError:
            joint_delta = np.linalg.pinv(Jw) @ err_w  # fallback (n_act,)

        # small step size, per-joint clipping
        step_size = 0.05  # scalar
        max_per_joint = 0.05  # scalar
        # delta: (n_act,) clipped joint position increments
        delta = np.clip(step_size * joint_delta, -max_per_joint, max_per_joint)  # (n_act,)

        # apply delta to data.qpos at addresses act_qpos (indices into qpos array)
        if len(act_qpos) == len(delta):
            # data.qpos shape: (nq,)
            data.qpos[act_qpos] += delta  # act_qpos: (n_act,), delta: (n_act,)
        else:
            n = min(len(act_qpos), len(delta))
            data.qpos[act_qpos[:n]] += delta[:n]

        # enforce joint limits on actuated joints only
        for idx, jid in enumerate(act_jids):
            # each entry in ``act_qpos`` corresponds to this joint
            qidx = act_qpos[idx]
            if jid < model.jnt_range.shape[0]:
                rmin, rmax = model.jnt_range[jid]
                if not (np.isnan(rmin) or np.isinf(rmin)):
                    data.qpos[qidx] = max(rmin, data.qpos[qidx])
                if not (np.isnan(rmax) or np.isinf(rmax)):
                    data.qpos[qidx] = min(rmax, data.qpos[qidx])

        # 更新模拟器状态：使新关节位置生效
        mujoco.mj_forward(model, data)

    # 返回最终误差：计算迭代结束后的总误差
    current_pos, current_quat = get_end_effector_pose(model, data, body_name)  # current_pos shape: (3,), current_quat shape: (4,), both dtype: float64
    pos_error = target_pos - current_pos  # pos_error shape: (3,), dtype: float64
 
    q_diff = R.from_quat(target_quat).inv() * R.from_quat(current_quat)  # q_diff: Rotation object
    rot_error = q_diff.as_rotvec()  # rot_error shape: (3,), dtype: float64

    final_error = np.hstack([pos_error, rot_error])  # final_error shape: (6,), dtype: float64
    return np.linalg.norm(final_error)  # scalar, dtype: float64


def set_gripper_position(model, data, left_gripper_pos, right_gripper_pos):
    """
    Set the gripper positions for both arms.
    """
    # Map gripper position (0-1) to joint position limits
    left_gripper_limit_min = 0.0  # scalar, value: 0.0
    left_gripper_limit_max = 0.044  # scalar, value: 0.044
    left_gripper_value = left_gripper_limit_min + left_gripper_pos * (left_gripper_limit_max - left_gripper_limit_min)  # scalar, dtype: float64

    right_gripper_limit_min = 0.0  # scalar, value: 0.0
    right_gripper_limit_max = 0.044  # scalar, value: 0.044
    right_gripper_value = right_gripper_limit_min + right_gripper_pos * (right_gripper_limit_max - right_gripper_limit_min)  # scalar, dtype: float64

    # Find the joint IDs
    left_joint7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_joint7")  # scalar, dtype: int
    left_joint8_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_joint8")  # scalar, dtype: int
    right_joint17_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_joint17")  # scalar, dtype: int
    right_joint18_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_joint18")  # scalar, dtype: int

    # Set the joint positions if they exist
    if left_joint7_id != -1:
        data.qpos[left_joint7_id] = left_gripper_value  # data.qpos shape: (model.nq,), dtype: float64
    if left_joint8_id != -1:
        data.qpos[left_joint8_id] = left_gripper_value  # data.qpos shape: (model.nq,), dtype: float64
    if right_joint17_id != -1:
        data.qpos[right_joint17_id] = right_gripper_value  # data.qpos shape: (model.nq,), dtype: float64
    if right_joint18_id != -1:
        data.qpos[right_joint18_id] = right_gripper_value  # data.qpos shape: (model.nq,), dtype: float64


def solve_dual_arm_ik(model, data, left_body, left_target_pos, left_target_rpy,
                      right_body, right_target_pos, right_target_rpy,
                      max_iter=500, tol=1e-3, pos_weight=1.0, rot_weight=0.2, svd_alpha=0.1, step=0.03):
    """Solve IK for both arms simultaneously by stacking their task equations.

    This operates directly on `data.qpos` and calls `mj_forward` each
    iteration but does not step dynamics (so actuators won't fight the
    solution).  It returns the final task-space norm.
    """
    # 获取左右臂的被驱动关节索引（关节ID、qpos、qvel）
    ljids, lqpos, lqvel = get_actuated_indices_by_prefix(model, 'left_')  # ljids: (nl,), lqpos: (nl,), lqvel: (nl,)
    rjids, rqpos, rqvel = get_actuated_indices_by_prefix(model, 'right_')  # rjids: (nr,), rqpos: (nr,), rqvel: (nr,)

    # 合并左右臂的qpos/qvel索引，得到所有被驱动关节的qpos/qvel索引
    act_qpos = np.concatenate([lqpos, rqpos])  # act_qpos: (nl+nr,), dtype: int
    act_qvel = np.concatenate([lqvel, rqvel])  # act_qvel: (nl+nr,), dtype: int

    # 打印关节映射信息，便于调试
    print(f"[IK] lqpos: {lqpos} (shape: {lqpos.shape}), rqpos: {rqpos} (shape: {rqpos.shape}), act_qpos: {act_qpos} (shape: {act_qpos.shape})")
    print(f"[IK] lqvel: {lqvel} (shape: {lqvel.shape}), rqvel: {rqvel} (shape: {rqvel.shape}), act_qvel: {act_qvel} (shape: {act_qvel.shape})")

    for it in range(max_iter):
        # 获取左臂末端执行器的当前位姿
        lpos, lquat = get_end_effector_pose(model, data, left_body)  # lpos: (3,), lquat: (4,)
        # 计算左臂末端位置误差
        lpos_err = left_target_pos - lpos  # lpos_err: (3,)
        # 计算左臂目标四元数
        lqdiff = R.from_euler('xyz', left_target_rpy).as_quat()  # lqdiff: (4,)
        # 计算左臂旋转误差（轴角）
        lrot_err = (R.from_quat(lqdiff).inv() * R.from_quat(lquat)).as_rotvec()  # lrot_err: (3,)

        # 获取右臂末端执行器的当前位姿
        rpos, rquat = get_end_effector_pose(model, data, right_body)  # rpos: (3,), rquat: (4,)
        # 计算右臂末端位置误差
        rpos_err = right_target_pos - rpos  # rpos_err: (3,)
        # 计算右臂目标四元数
        rqdiff = R.from_euler('xyz', right_target_rpy).as_quat()  # rqdiff: (4,)
        # 计算右臂旋转误差（轴角）
        rrot_err = (R.from_quat(rqdiff).inv() * R.from_quat(rquat)).as_rotvec()  # rrot_err: (3,)

        # 拼接左右臂的任务空间误差向量
        err = np.hstack([lpos_err, lrot_err, rpos_err, rrot_err])  # err: (12,)
        norm = np.linalg.norm(err)  # norm: 标量
        # 打印当前迭代的末端误差
        if it == 0 or norm < tol or it == max_iter-1:
            print(f"[IK] iter {it}: lpos={lpos} (3,), lpos_err={lpos_err} (3,), lrot_err={lrot_err} (3,) | rpos={rpos} (3,), rpos_err={rpos_err} (3,), rrot_err={rrot_err} (3,) | norm={norm:.6f}")
        # 收敛判断
        if norm < tol:
            print(f"[IK] Converged at iter {it}, norm={norm:.6f}")
            # 返回最终误差
            return norm

        # 计算左右臂的雅可比矩阵
        Jl = compute_jacobian(model, data, left_body, lqvel)  # Jl: (6, nl)
        Jr = compute_jacobian(model, data, right_body, rqvel)  # Jr: (6, nr)

        # 构造块对角雅可比矩阵（12, nl+nr）
        J_top = np.hstack([Jl, np.zeros((6, Jr.shape[1]))])  # J_top: (6, nl+nr)
        J_bottom = np.hstack([np.zeros((6, Jl.shape[1])), Jr])  # J_bottom: (6, nl+nr)
        J_total = np.vstack([J_top, J_bottom])  # J_total: (12, nl+nr)

        # 构造加权矩阵，对位置和旋转误差赋予不同权重
        Wl = np.diag([pos_weight, pos_weight, pos_weight, rot_weight, rot_weight, rot_weight])  # Wl: (6,6)
        Wr = np.diag([pos_weight, pos_weight, pos_weight, rot_weight, rot_weight, rot_weight])  # Wr: (6,6)
        W_total = np.block([[Wl, np.zeros((6, 6))], [np.zeros((6, 6)), Wr]])  # W_total: (12,12)

        # 对误差和雅可比矩阵加权
        err_w = W_total @ err  # err_w: (12,)
        Jw = W_total @ J_total  # Jw: (12, nl+nr)

        # SVD分解加权雅可比，估计最小奇异值
        eps = 1e-8
        try:
            s = np.linalg.svd(Jw, compute_uv=False)  # s: (min(12, nl+nr),)
            sv_min = float(s[-1]) if s.size > 0 else 0.0
        except Exception:
            sv_min = 0.0
        # 计算自适应阻尼系数lambda
        alpha = float(svd_alpha)
        lam = alpha / (sv_min + eps)
        lam = float(max(1e-4, min(1.0, lam)))  # lam: 标量
        # 构造正则化矩阵JJ
        JJ = Jw @ Jw.T + (lam * lam) * np.eye(Jw.shape[0])  # JJ: (12,12)
        try:
            y = np.linalg.solve(JJ, err_w)  # y: (12,)
            dq = Jw.T @ y  # dq: (nl+nr,)
        except np.linalg.LinAlgError:
            dq = np.linalg.pinv(Jw) @ err_w  # dq: (nl+nr,)

        # 步长缩放与逐关节裁剪
        dq = np.clip(step * dq, -step, step)  # dq: (nl+nr,)

        # 应用关节增量到qpos
        data.qpos[act_qpos] += dq  # data.qpos: (nq,), act_qpos: (nl+nr,), dq: (nl+nr,)
        # 打印关节增量和qpos变化
        if it == 0 or norm < tol or it == max_iter-1:
            print(f"[IK] dq: {dq} (shape: {dq.shape}), qpos[act_qpos]: {data.qpos[act_qpos]} (shape: {data.qpos[act_qpos].shape})")

        # 推进模拟器，刷新kinematics
        mujoco.mj_forward(model, data)

    # 返回最终误差
    print(f"[IK] Not converged after {max_iter} iters, final norm={norm:.6f}")
    return np.linalg.norm(err)


def main():
    """
    Main function with achievable target poses for optimal convergence.
    """
    # Path to the URDF file
    urdf_path = "/home/gyh/Workspace/lerobot/src/lerobot/model/acone/urdf/acone.urdf"  # string, value: "/home/gyh/Workspace/lerobot/src/lerobot/model/acone/urdf/acone.urdf"

    try:
        # Create model from URDF
        model = create_model_from_urdf(urdf_path)  # MjModel object

        # Create data
        data = mujoco.MjData(model)  # MjData object with properties like qpos, qvel, xpos, xmat

        # Define end effector body names
        left_end_effector = "left_link7"  # string, value: "left_link7"
        right_end_effector = "right_link17"  # string, value: "right_link17"
        
        # Initialize the simulation
        mujoco.mj_forward(model, data)
        
        print("="*70)
        print("Acone Dual-Arm Robot Optimized IK Control Demo")
        print("="*70)
        print("Features:")
        print("- Optimized inverse kinematics with adaptive parameters")
        print("- Achievable target poses for rapid convergence")
        print("- Controls both arms with position, rotation, and gripper control")
        print("- Press ESC in viewer window to exit")
        print("="*70)
        
        # Use achievable target poses near the robot's natural position
        left_target_pos = np.array([0.2, 0.3, 0.6])  # left_target_pos shape: (3,), dtype: float64
        left_target_rpy = np.array([0, 0, np.pi/4])   # left_target_rpy shape: (3,), dtype: float64

        right_target_pos = np.array([0.2, -0.3, 0.6])  # right_target_pos shape: (3,), dtype: float64
        right_target_rpy = np.array([0, 0, -np.pi/4])    # right_target_rpy shape: (3,), dtype: float64

        # Set gripper positions
        left_gripper_pos = 1.0  # scalar, value: 1.0 (Open)
        right_gripper_pos = 0.0  # scalar, value: 0.0 (Closed)
        
        print(f"\nLeft target: pos={left_target_pos}, rpy={left_target_rpy}")
        print(f"Right target: pos={right_target_pos}, rpy={right_target_rpy}")
        
        # Solve IK for both arms initially
        print("\nSolving IK for both arms (combined)...")
        both_error = solve_dual_arm_ik(
            model, data,
            left_end_effector, left_target_pos, left_target_rpy,
            right_end_effector, right_target_pos, right_target_rpy,
            max_iter=800, tol=1e-3,
        )
        print(f"Dual-arm final error (stacked task): {both_error:.6f}")
        
        # Set gripper positions
        set_gripper_position(model, data, left_gripper_pos, right_gripper_pos)

        # Update simulation
        mujoco.mj_forward(model, data)

        # If running headless (environment variable), emulate the main loop
        # but without opening the viewer. This runs a few maintenance IK
        # cycles and prints L_Error/R_Error so we can validate convergence.
        if os.environ.get('LEROBOT_HEADLESS', '0') == '1':
            print('Headless mode: running emulated main loop (no viewer).')
            step_count = 0
            # emulate 500 simulation steps (0.01s per step ~5s)
            for _ in range(500):
                if step_count % 100 == 0:
                    # Use stacked dual-arm IK for maintenance
                    both_error = solve_dual_arm_ik(
                        model, data,
                        left_end_effector, left_target_pos, left_target_rpy,
                        right_end_effector, right_target_pos, right_target_rpy,
                        max_iter=500, tol=1e-3, pos_weight=1.5, rot_weight=0.1, svd_alpha=0.05, step=0.03
                    )
                    left_current_pos, _ = get_end_effector_pose(model, data, left_end_effector)
                    right_current_pos, _ = get_end_effector_pose(model, data, right_end_effector)
                    left_pos_error = np.linalg.norm(left_target_pos - left_current_pos)
                    right_pos_error = np.linalg.norm(right_target_pos - right_current_pos)
                    print(f"Headless Time: {data.time:.2f}s | L_Error: {left_pos_error:.6f} | R_Error: {right_pos_error:.6f} | Both_Error: {both_error:.6f}")
                step_count += 1
                mujoco.mj_step(model, data)
            print('Headless emulation finished.')
            return

        # Open the viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Main simulation loop
            step_count = 0  # scalar, dtype: int
            print("\nStarting simulation...")
            while viewer.is_running():
                # Periodically recompute IK to maintain position
                if step_count % 100 == 0:  # Every 2 seconds (200 steps * 0.01s)
                    print("\nRe-solving dual-arm IK to maintain targets...")
                    # Use stacked dual-arm IK for maintenance
                    both_error = solve_dual_arm_ik(
                        model, data,
                        left_end_effector, left_target_pos, left_target_rpy,
                        right_end_effector, right_target_pos, right_target_rpy,
                        max_iter=500, tol=1e-3, pos_weight=1.5, rot_weight=0.1, svd_alpha=0.05, step=0.03
                    )
                    # Set gripper positions
                    set_gripper_position(model, data, left_gripper_pos, right_gripper_pos)

                # Get current poses
                left_current_pos, _ = get_end_effector_pose(model, data, left_end_effector)  # left_current_pos shape: (3,), dtype: float64
                right_current_pos, _ = get_end_effector_pose(model, data, right_end_effector)  # right_current_pos shape: (3,), dtype: float64

                # Print info periodically
                step_count += 1
                if step_count % 100 == 0:  # Print every second (100 steps * 0.01s)
                    # Re-calculate position errors
                    left_pos_error = np.linalg.norm(left_target_pos - left_current_pos)  # scalar, dtype: float64
                    right_pos_error = np.linalg.norm(right_target_pos - right_current_pos)  # scalar, dtype: float64

                    print(f"Time: {data.time:.2f}s | "
                          f"L_Target: [{left_target_pos[0]:.2f}, {left_target_pos[1]:.2f}, {left_target_pos[2]:.2f}] | "
                          f"L_Current: [{left_current_pos[0]:.2f}, {left_current_pos[1]:.2f}, {left_current_pos[2]:.2f}] | "
                          f"L_Error: {left_pos_error:.4f} | "
                          f"L_Gripper: {'Open' if left_gripper_pos > 0.5 else 'Closed'} | "
                          f"R_Target: [{right_target_pos[0]:.2f}, {right_target_pos[1]:.2f}, {right_target_pos[2]:.2f}] | "
                          f"R_Current: [{right_current_pos[0]:.2f}, {right_current_pos[1]:.2f}, {right_current_pos[2]:.2f}] | "
                          f"R_Error: {right_pos_error:.4f} | "
                          f"R_Gripper: {'Open' if right_gripper_pos > 0.5 else 'Closed'} | "
                          f"Both_Error: {both_error:.6f}")

                # Step the simulation
                mujoco.mj_step(model, data)

                # Sync the viewer
                if step_count % 10 == 0:
                    viewer.sync()

                # Slow down the loop to visualize properly
                time.sleep(0.01)
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()