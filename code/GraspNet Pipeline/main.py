import os
import datetime

import cv2

from util import *
import numpy as np
from camera import RealSenseCapture
from dobot_cr3 import DobotController
from graspnet_pipeline import GraspNetPipeline

CALIB_FILE = 'T_cam2gripper.npy'


if __name__ == '__main__':
    cr3_robot = DobotController()
    cam = RealSenseCapture()
    pipeline = GraspNetPipeline(
        checkpoint_path='dataset/model/checkpoint-rs.tar',
        workspace_mask_path='dataset/workspace_mask/workspace_1280_720_0.png',
        root_dir='~/graspnet-baseline/'
    )
    print("GraspNet initialized.")
    meta = cam.get_depth_meta()

    verb_dict = {1: "play", 2: "contain", 3: "construct", 4: "support", 5: "design", 6: "serve", 7: "transfer",
                 8: "grow", 9: "install",
                 10: "activate", 11: "cover", 12: "cut", 13: "write", 14: "feed", 15: "pick", 16: "open", 17: "wear",
                 18: "pull", 19: "eat", 20: "rotate"}

    try:
        if cr3_robot.connect():
            cr3_robot.clear_error_and_enable()
            cr3_robot.init_gripper()

            '''
            hand eye calibration
            '''
            if os.path.exists(CALIB_FILE):
                T_cam2gripper = np.load(CALIB_FILE)
                print("Load T_cam2gripper:\n", T_cam2gripper)
            else:
                raise Exception('NO EYE CALIBRATION')

            # grasp point
            time.sleep(1)
            grasp_pose = [1.0385, -8.9023, -29.9562, -49.6342, 91.3102, 1.7568]
            cr3_robot.move_joint(*grasp_pose)
            cr3_robot.sync()
            cr3_robot.open_gripper(100)
            time.sleep(2)

            print("Press 'a' to execute grasping, press 'q' to quit the program, press 'c' to clear error.")

            while True:
                cmd = input("Enter command (a: execute grasping, q: quit): ").strip().lower()

                if cmd == 'q':
                    print("Detected 'q', preparing to exit...")
                    break
                elif cmd == 'c':
                    cr3_robot.clear_error_and_enable()
                    cr3_robot.move_joint(*grasp_pose)
                    cr3_robot.sync()
                    cr3_robot.open_gripper(100)
                elif cmd == 'a':
                    print("Detected 'a', starting grasping process...")
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
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_dir_color = "log/real_world"
                        save_dir = os.path.join(save_dir_color, timestamp)
                        os.makedirs(save_dir, exist_ok=True)

                        cam.save_dir = save_dir
                        color, depth = cam.capture(save=False)
                        print('Image shape', color.shape)

                        save_path = os.path.join(save_dir, f"color_{verb}.png")
                        cv2.imwrite(save_path, color)
                        print(f"Saved color image to {save_path}")

                        grasps, cloud, points_2d, color_rgb = pipeline.run(color, depth, meta, gg_num=70, plot=True)
                        associated_point_ids = pipeline.send_grasps_wait_return(points_2d, color_rgb, verb, timestamp)

                        selected_grasps = [grasps[i] for i in associated_point_ids if i < len(grasps)]
                        if not selected_grasps:
                            print("SAM+CLIP did not return valid grasps, skipping this attempt.")
                            continue

                        _, index = try_grasp_candidates(selected_grasps, T_cam2gripper, cr3_robot)
                        if index is not None:
                            pipeline.plot_one_grasp(selected_grasps[index:index + 1], cloud)

                        time.sleep(2)
                        cr3_robot.move_joint(*grasp_pose)
                        time.sleep(2)
                        cr3_robot.open_gripper(100)

                        pipeline.save_grasp_scene(cloud, grasps, save_dir)
                        print("Grasping finished, waiting for next command.")

                    except Exception as e:
                        print(f"Error occurred during grasping: {e}")

                else:
                    print("Invalid command, please enter 'a' or 'q' or 'c'.")

        else:
            print("lease check network/IP/port or whether the robot is started.")
    except Exception as e:
        print(f"Error occurred during execution: {e}")
    finally:
        cam.stop()
        cr3_robot.shutdown()
