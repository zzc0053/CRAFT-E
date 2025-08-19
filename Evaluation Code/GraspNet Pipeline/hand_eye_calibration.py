import traceback

import cv2
import os
import time
import json
import numpy as np
from camera import RealSenseCapture
from dobot_cr3 import DobotController
from scipy.spatial.transform import Rotation as R

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

SAVE_DIR = "calib_samples"
CALIB_FILE = 'T_cam2gripper.npy'
os.makedirs(SAVE_DIR, exist_ok=True)

def estimate_pose_from_corners(corners, marker_length, camera_matrix, dist_coeffs):
    objp = np.array([
        [-marker_length/2,  marker_length/2, 0],
        [ marker_length/2,  marker_length/2, 0],
        [ marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)

    corners = corners.reshape(-1, 2).astype(np.float32)
    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success:
        raise ValueError("solvePnP failed to estimate pose.")
    return rvec, tvec

def detect_aruco_pose(color_img, camera_matrix, dist_coeffs, marker_length):
    corners, ids, _ = aruco_detector.detectMarkers(color_img)
    if ids is not None and len(ids) > 0:
        rvec, tvec = estimate_pose_from_corners(corners[0], marker_length, camera_matrix, dist_coeffs)
        rot_mat, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = tvec.flatten()
        return T
    return None

def pose_to_rt(pose):
    pos = np.array([pose['x'], pose['y'], pose['z']]) / 1000.0  # mm → m
    rot = R.from_euler('xyz', [pose['rx'], pose['ry'], pose['rz']], degrees=True)
    return rot.as_matrix(), pos

def run_hand_eye_calibration(sample_dir):
    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []

    for fname in sorted(os.listdir(sample_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(sample_dir, fname), 'r') as f:
                data = json.load(f)
            Rg, tg = pose_to_rt(data["T_tool_in_base"])
            Tc = np.array(data["T_marker_in_cam"])
            Rc, tc = Tc[:3, :3], Tc[:3, 3]
            R_gripper2base.append(Rg)
            t_gripper2base.append(tg)
            R_target2cam.append(Rc)
            t_target2cam.append(tc)

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    T = np.eye(4)
    T[:3, :3] = R_cam2gripper
    T[:3, 3] = t_cam2gripper.flatten()
    print("Hand-Eye Calibration Result (T_cam_to_gripper):\n", T)
    return T

def collect_and_calibrate(cr3_robot, cam, pose_joints, marker_length, save_dir="calib_samples"):
    try:
        camera_matrix, dist_coeffs = cam.get_color_intrinsics()
        print(camera_matrix, dist_coeffs)
        cr3_robot.move_joint(*pose_joints[0])
        time.sleep(3)
        for i, j in enumerate(pose_joints):
            print(f"\nMoving to position {i + 1}...")
            time.sleep(2)
            cr3_robot.move_joint(*j)
            time.sleep(5)

            color, depth = cam.capture(color_name=f"color_{i + 1}.png", depth_name=f"depth_{i + 1}.png", save=True)

            T_marker_in_cam = detect_aruco_pose(color, camera_matrix, dist_coeffs, marker_length)
            if T_marker_in_cam is None:
                print("ArUco not detected. Skipping this pose.")
                continue

            pose = cr3_robot.get_pose()
            T_tool_in_base = {"x": pose[0], "y": pose[1], "z": pose[2], "rx": pose[3], "ry": pose[4], "rz": pose[5]}

            sample = {
                "T_tool_in_base": T_tool_in_base,
                "T_marker_in_cam": T_marker_in_cam.tolist()
            }

            with open(os.path.join(SAVE_DIR, f"sample_{i + 1}.json"), 'w') as f:
                json.dump(sample, f, indent=2)
            print(f"Sample {i + 1} saved")

        print("\nStarting hand-eye calibration...")
        T_cam2gripper = run_hand_eye_calibration(SAVE_DIR)
        np.save("T_cam2gripper.npy", T_cam2gripper)
        return T_cam2gripper
    except Exception as e:
        print("Error during data collection or calibration:", e)
        traceback.print_exc()
        return None


if __name__ == '__main__':
    cr3_robot = DobotController()
    cam = RealSenseCapture()
    try:
        if cr3_robot.connect():
            cr3_robot.clear_error_and_enable()
            cr3_robot.init_gripper()

            pose_joints = [
                [1.7332, 38.1981, -124.1951, -1.1087, 95.1194, -20.6142],
                [14.5700, 50.9889, -129.7119, -4.6239, 99.6224, 27.9647],
                [13.1868, -37.4615, -33.6647, -66.7348, 94.3931, 3.4685],
                [-0.0688, -43.2591, -37.0228, -45.3982, 125.8943, 101.9319],
                [28.2057, 7.7391, -97.6200, -25.2207, 86.1598, -20.6143],
                [11.6747, -16.5183, -68.7473, -44.2796, 103.9020, 36.9803],
                [32.7891, -0.0011, -107.7133, -0.7689, 105.7133, 112.7655],
                [47.8717, -1.1297, -99.8336, -31.6495, 83.2212, 32.8316],
                [0.573, 4.2820, -104.9113, -13.1773, 121.7359, 46.5973],
                [-29.2295, 16.8835, -135.5524, 31.0132, 113.5000, -87.5301],
                [0.0858, 58.3794, -137.8229, -3.4698, 110.9610, -20.6149]
            ]
            T_cam2gripper = collect_and_calibrate(cr3_robot, cam, pose_joints, marker_length=0.0948, save_dir="calib_samples")
            print("Saved to:", CALIB_FILE)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cam.stop()
        cr3_robot.shutdown()
