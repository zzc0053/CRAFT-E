import time

import numpy as np
from scipy.spatial.transform import Rotation as R

def transform_grasp_pose_cam_to_base(R_cam, t_cam, T_cam2gripper, T_gripper2base):
    T_grasp_cam = np.eye(4)
    T_grasp_cam[:3, :3] = R_cam
    T_grasp_cam[:3, 3] = t_cam

    T_cam2base = T_gripper2base @ T_cam2gripper
    T_grasp_base = T_cam2base @ T_grasp_cam

    R_base = T_grasp_base[:3, :3]
    t_base = T_grasp_base[:3, 3] * 1000.0  # meters → mm

    R_extra = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    R_corrected = R_base @ R_extra

    euler_base = R.from_matrix(R_corrected).as_euler('xyz', degrees=True)

    grasp_pose_base = [
        float(t_base[0]), float(t_base[1]), float(t_base[2]),
        float(euler_base[0]), float(euler_base[1]), float(euler_base[2])
    ]
    return grasp_pose_base


def euler_pose_to_matrix(pose):
    pos = np.array([pose['x'], pose['y'], pose['z']]) / 1000.0  # mm → m
    rpy = np.deg2rad([pose['rx'], pose['ry'], pose['rz']])      # deg → rad
    rot = R.from_euler('xyz', rpy).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def safe_inverse_and_move(dashboard, pose, user=0, tool=2):
    x, y, z, rx, ry, rz = pose
    response = dashboard.InverseSolution(x, y, z+5, rx, ry, rz, user, tool)

    try:
        parts = response.strip().split(";", 1)[0].split(",", 1)
        error_id = int(parts[0].strip())

        joint_str = parts[1].split("}")[0].replace("{", "").strip()
        joint_angles = [float(x) for x in joint_str.split(",")]

        if error_id != 0 or all(abs(j) < 1e-6 for j in joint_angles):
            print("Pose unreachable or no solution found")
            return False, None

        return True, joint_angles

    except Exception as e:
        print("Parsing failed:", e)
        print("Response:", response)
        return False, None


def try_grasp_candidates(grasps, T_cam2gripper, cr3_robot):
    pose = cr3_robot.get_pose()  # [x, y, z, rx, ry, rz] in mm / deg
    T_griper2base = euler_pose_to_matrix({
        "x": pose[0], "y": pose[1], "z": pose[2],
        "rx": pose[3], "ry": pose[4], "rz": pose[5]
    })

    for i, grasp in enumerate(grasps):
        t_cam = grasp.translation  # (3,)
        R_cam = grasp.rotation_matrix  # (3,3)

        try:
            grasp_pose = transform_grasp_pose_cam_to_base(R_cam, t_cam, T_cam2gripper, T_griper2base)
            x, y, z, rx, ry, rz = grasp_pose
            if z < 0:
                print(f"Grasp #{i} has invalid z < 0, skipped.")
                continue

            success, joints = safe_inverse_and_move(cr3_robot.dashboard, grasp_pose, user=0, tool=cr3_robot.tool_index)

            if success:
                print(f"[SUCCESS] Grasp #{i} succeeded at base pose: {grasp_pose}")
                cr3_robot.move_joint(*joints)
                time.sleep(4)
                cr3_robot.open_gripper(0)
                return True, i
            else:
                print(f"FAIL: Grasp #{i} failed to move.")

        except Exception as e:
            print(f"ERROR: Grasp #{i} exception: {e}")

    print("ERROR: All grasp candidates failed.")
    return False, None