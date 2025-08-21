import time

import os
import numpy as np
import cv2
import socket
import struct
import pickle
import zlib
import io
from PIL import Image
import datetime

from gknet.datasets.dataset_factory import dataset_factory
from gknet.detectors.detector_factory import detector_factory
from gknet.opts import opts

from camera import RealSenseCapture
from dobot_cr3 import DobotController
from scipy.spatial.transform import Rotation as R


def send_grasps_wait_return(bboxes, rgb_image, verb, timestamp):
    img_pil = Image.fromarray(rgb_image)
    img_bytes_io = io.BytesIO()
    img_pil.save(img_bytes_io, format="PNG")
    img_bytes = img_bytes_io.getvalue()

    data = pickle.dumps((bboxes, img_bytes, verb, timestamp))
    data = zlib.compress(data)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('255.255.255.255', 5000))

        data_len = len(data)
        s.sendall(struct.pack('>I', data_len))
        s.sendall(data)
        print(f"Sent {data_len} bytes to server with length prefix, waiting for associated_point_ids...")

        raw_len = s.recv(4)
        if len(raw_len) < 4:
            raise RuntimeError("Failed to receive data length from server.")
        recv_data_len = struct.unpack('>I', raw_len)[0]
        print(f"Expecting {recv_data_len} bytes from server.")

        received_data = b""
        while len(received_data) < recv_data_len:
            packet = s.recv(min(4096, recv_data_len - len(received_data)))
            if not packet:
                raise RuntimeError("Connection closed before all data received.")
            received_data += packet

    associated_point_ids = pickle.loads(zlib.decompress(received_data))
    print("Received associated_point_ids:", associated_point_ids)
    return associated_point_ids


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
    response = dashboard.InverseSolution(x, y, z+3, rx, ry, rz, user, tool)

    try:
        parts = response.strip().split(";", 1)[0].split(",", 1)
        error_id = int(parts[0].strip())

        joint_str = parts[1].split("}")[0].replace("{", "").strip()
        joint_angles = [float(x) for x in joint_str.split(",")]

        if error_id != 0 or all(abs(j) < 1e-6 for j in joint_angles):
            print("Pose unreachable or unsolvable")
            return False, None

        return True, joint_angles

    except Exception as e:
        print("Parsing failed:", e)
        print("Response:", response)
        return False, None

def project(pixel, depth_image, cameraMatrix, T_cam2base):
    depth = depth_image[pixel[1], pixel[0]]
    nei_range = 1
    while depth == 0:
        for dx in range(-nei_range, nei_range + 1):
            for dy in range(-nei_range, nei_range + 1):
                x, y = pixel[0] + dx, pixel[1] + dy
                depth = depth_image[y, x]
                if depth != 0:
                    break
            if depth != 0:
                break
        nei_range += 1
    depth = depth/1000
    pxl_cam = np.linalg.inv(cameraMatrix) @ np.array([pixel[0] * depth, pixel[1] * depth, depth])
    point_cam = np.array([pxl_cam[0], pxl_cam[1], pxl_cam[2], 1.0])
    point_gripper = T_cam2base @ point_cam
    return point_gripper[:3]

def pre_process(rgb_img, depth_img):
    inp_image = rgb_img.copy()
    inp_image[:, :, 0] = depth_img
    inp_image = cv2.resize(inp_image, (512, 512))
    inp_image = inp_image[:, :, ::-1]
    return inp_image

def isWithinRange(pxl, w, h):
    x, y = pxl[:]
    return w / 12.0 <= x <= 11 * w / 12 and h / 12.0 <= y <= 11 * h / 12

def KpsToGrasppose(net_output, depth_map, camera_matrix, T_cam2base):
    kps_pr = []
    for category_id, preds in net_output.items():
        if len(preds) == 0:
            continue
        for pred in preds:
            kps = pred[:4]
            score = pred[-1]
            kps_pr.append([kps[0], kps[1], kps[2], kps[3], score])

    if len(kps_pr) == 0:
        return [0, 0, 0, 0]

    kps_pr = sorted(kps_pr, key=lambda x: x[-1], reverse=True)
    res = None
    for kp_pr in kps_pr:
        kp_lm = (int(kp_pr[0]), int(kp_pr[1]))
        kp_rm = (int(kp_pr[2]), int(kp_pr[3]))
        if isWithinRange(kp_lm, width, height) and isWithinRange(kp_rm, width, height):
            res = kp_pr
            break
    if res is None:
        return [0, 0, 0, 0]

    kp_lm = (int(res[0]), int(res[1]))
    kp_rm = (int(res[2]), int(res[3]))
    center = (int((kp_lm[0] + kp_rm[0]) / 2), int((kp_lm[1] + kp_rm[1]) / 2))

    kp_lm_3d = project(kp_lm, depth_map, camera_matrix, T_cam2base)
    kp_rm_3d = project(kp_rm, depth_map, camera_matrix, T_cam2base)
    center_3d = project(center, depth_map, camera_matrix, T_cam2base)
    orientation = np.arctan2(kp_rm_3d[1] - kp_lm_3d[1], kp_rm_3d[0] - kp_lm_3d[0])

    if orientation > np.pi / 2:
        orientation = np.pi - orientation
    elif orientation < -np.pi / 2:
        orientation = -np.pi - orientation
    else:
        orientation = -orientation

    dist = np.linalg.norm(kp_lm_3d[:2] - kp_rm_3d[:2])

    return [center_3d[0], center_3d[1], center_3d[2], orientation, dist]

def run(opt, camera, cr3_robot, w, h):
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    camera_matrix, _ = camera.get_color_intrinsics()

    scale_x = w / 512.0  # width / 512
    scale_y = h / 512.0  # height / 512
    verb_dict = {1:"play", 2:"contain", 3:"construct", 4:"support", 5:"design", 6:"serve", 7:"transfer", 8:"grow", 9:"install",
                10:"activate", 11:"cover", 12:"cut", 13:"write", 14:"feed", 15:"pick", 16:"open", 17:"wear", 18:"pull", 19:"eat", 20:"rotate"}

    try:
        if cr3_robot.connect():
            cr3_robot.clear_error_and_enable()
            cr3_robot.init_gripper()
        T_cam2gripper = np.load('T_cam2gripper.npy')
        # grasp point
        time.sleep(1)
        init_grasp_pose = [1.0385, -8.9023, -29.9562, -49.6342, 91.3102, 1.7568]
        cr3_robot.move_joint(*init_grasp_pose)
        cr3_robot.sync()
        cr3_robot.open_gripper(100)
        time.sleep(2)

        while True:
            cmd = input("Press 'a' to execute grasping, press 'q' to quit the program, press 'c' to clear error.").strip().lower()
            if cmd == 'q':
                print("Detected 'q', preparing to exit...")
                break
            elif cmd == 'c':
                cr3_robot.clear_error_and_enable()
                cr3_robot.move_joint(*init_grasp_pose)
                cr3_robot.sync()
                cr3_robot.open_gripper(100)
            elif cmd == 'a':
                for k, v in verb_dict.items():
                    print(f"{k}: {v}")
                try:
                    verb_id = int(input("Enter action ID: ").strip())
                    verb = verb_dict.get(verb_id)
                    if verb is None:
                        print("Invalid ID, skipping this execution")
                        continue
                    print(f"Selected action: {verb}")
                except ValueError:
                    print("Invalid input, please enter a numeric ID")
                    continue

                try:
                    color_frame, depth_frame = camera.capture(save=False)
                    depth_raw = depth_frame
                    depth = depth_raw.astype(np.uint8)
                    img = color_frame
                    time.sleep(1)
                    # get the image detect results
                    inp_image = pre_process(img, depth)
                    ret = detector.run(inp_image)["results"]

                    # get current arm position
                    pose = cr3_robot.get_pose()  # [x, y, z, rx, ry, rz] in mm / deg
                    T_griper2base = euler_pose_to_matrix({
                        "x": pose[0], "y": pose[1], "z": pose[2],
                        "rx": pose[3], "ry": pose[4], "rz": pose[5]
                    })
                    T_cam2base = T_griper2base @ T_cam2gripper

                    # get the center of each grasp to send to server
                    # save rgb picture and grasps
                    save_dir_color = "log/real_world_kpnet"
                    os.makedirs(save_dir_color, exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(save_dir_color, f"{timestamp}_{verb}.png")
                    img_vis = img.copy()

                    bboxes = []
                    grasp_preds = []
                    for category_id, preds in ret.items():
                        for pred in preds:
                            x1, y1, x2, y2, score = pred[:5]
                            x1 *= scale_x
                            y1 *= scale_y
                            x2 *= scale_x
                            y2 *= scale_y
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            bboxes.append([cx, cy])
                            grasp_preds.append([x1, y1, x2, y2, score])

                            pt1 = (int(x1), int(y1))
                            pt2 = (int(x2), int(y2))
                            center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                            cv2.circle(img_vis, pt1, 4, (255, 0, 0), -1)
                            cv2.circle(img_vis, pt2, 4, (0, 255, 0), -1)
                            cv2.circle(img_vis, center, 4, (0, 0, 255), -1)
                            cv2.line(img_vis, pt1, pt2, (150, 150, 150), 1)

                    cv2.imwrite(save_path, img_vis)

                    color_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    associated_point_ids = send_grasps_wait_return(bboxes, color_rgb, verb, timestamp)

                    # resort associated_point_ids by score
                    if not associated_point_ids:
                        print("No valid grasp index received, skipping this round")
                        return

                    scored_grasps = []
                    for idx in associated_point_ids:
                        if idx < len(grasp_preds):
                            scored_grasps.append((idx, grasp_preds[idx][-1]))  # (index, score)
                        else:
                            print(f"idx {idx} exceeds grasp_preds length, skipping")
                    # Sort by score in descending order
                    scored_grasps.sort(key=lambda x: x[1], reverse=True)

                    for idx, score in scored_grasps:
                        x1, y1, x2, y2, _ = grasp_preds[idx]
                        pred = [x1, y1, x2, y2, score]
                        ret_single = {0: [pred]}

                        loc_ori = KpsToGrasppose(ret_single, depth_raw, camera_matrix, T_cam2base)
                        grasp_pose = [loc_ori[0]*1000, loc_ori[1]*1000, loc_ori[2]*1000+7.5, 180, 0, np.rad2deg(loc_ori[3])]
                        ik_success, joints = safe_inverse_and_move(cr3_robot.dashboard, grasp_pose, user=0, tool=cr3_robot.tool_index)

                        if ik_success:
                            print(f'Success! idx: {idx}, score: {score}, loc: {loc_ori}')
                            cr3_robot.move_joint(*joints)
                            time.sleep(4)
                            cr3_robot.open_gripper(0)
                            time.sleep(2)
                            cr3_robot.move_joint(*init_grasp_pose)
                            time.sleep(4)
                            cr3_robot.open_gripper(100)
                            break
                        else:
                            print("IK has no solution, trying the next one")

                except Exception as e:
                    print(f"Error occurred during grasping: {e}")
            else:
                print("Invalid command, please enter 'a' or 'q' or 'c'.")
    except Exception as e:
        print(f"Error occurred during execution: {e}")
    finally:
        cam.stop()
        cr3_robot.shutdown()



if __name__ == "__main__":
    opt = opts().parse()

    cr3_robot = DobotController()
    width, height = 1280, 720
    # width, height = 640, 480
    cam = RealSenseCapture(width, height)

    run(opt, cam, cr3_robot, width, height)
