import json
import struct
import time

import os
import io
import zlib
import socket
import pickle

import cv2
import numpy as np
import scipy.io as scio
from PIL import Image

from gknet.datasets.dataset_factory import dataset_factory
from gknet.detectors.detector_factory import detector_factory
from gknet.opts import opts


def project(pixel, depth_image, cameraMatrix):
    depth = depth_image[pixel[1], pixel[0]]
    nei_range = 1
    while depth == 0:
        for dx in range(-nei_range, nei_range + 1):
            for dy in range(-nei_range, nei_range + 1):
                nx, ny = pixel[0] + dx, pixel[1] + dy
                if 0 <= nx < depth_image.shape[1] and 0 <= ny < depth_image.shape[0]:
                    depth = depth_image[ny, nx]
                    if depth != 0:
                        break
            if depth != 0:
                break
        nei_range += 1
    pxl = np.linalg.inv(cameraMatrix).dot([pixel[0] * depth, pixel[1] * depth, depth])
    return np.array([pxl[0], pxl[1], pxl[2]])


def pre_process(rgb_img, depth_img):
    inp_image = rgb_img.copy()
    inp_image[:, :, 0] = depth_img

    inp_image = cv2.resize(inp_image, (512, 512))
    inp_image = inp_image[:, :, ::-1]

    return inp_image


def isWithinRange(pxl, w, h):
    x, y = pxl[:]
    return w / 12.0 <= x <= 11 * w / 12 and h / 12.0 <= y <= 11 * h / 12


def send_data(center, rgb, verb, rgb_name):
    img_bytes = io.BytesIO()
    rgb.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    data = pickle.dumps((center, img_bytes, verb, rgb_name))
    data = zlib.compress(data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('255.255.255.255', 5000))
        data_len = len(data)
        s.sendall(struct.pack('>I', data_len))
        s.sendall(data)
        print(f"{len(center)} 个 Grasp have been sent")


def KpsToGrasppose(net_output, rgb_img, depth_map, visualize=True, save_path=None):
    f_w, f_h = 640.0 / 512.0, 480.0 / 512.0
    left_right = []
    centers = []
    rgb_img = rgb_img.copy()

    for category_id, preds in net_output.items():
        if preds.shape[0] == 0:
            continue

        for pred in preds:
            x1, y1, x2, y2, score = pred
            kp_lm = (int(x1 * f_w), int(y1 * f_h))
            kp_rm = (int(x2 * f_w), int(y2 * f_h))

            # skip if not in range
            if not (isWithinRange(kp_lm, 640, 480) and isWithinRange(kp_rm, 640, 480)):
                print(f'skip {category_id} one grasp')
                continue

            center = (int((kp_lm[0] + kp_rm[0]) / 2), int((kp_lm[1] + kp_rm[1]) / 2))

            left_right.append([kp_lm, kp_rm])
            centers.append(center)

            if visualize:
                rgb_img = cv2.line(rgb_img, kp_lm, kp_rm, (150, 150, 150), 1)
                rgb_img = cv2.circle(rgb_img, kp_lm, 2, (0, 0, 255), 1)
                rgb_img = cv2.circle(rgb_img, kp_rm, 2, (0, 255, 0), 1)
                rgb_img = cv2.circle(rgb_img, center, 2, (255, 0, 0), 1)

    if visualize and save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, rgb_img)
        print(f"Saved to {save_path}")

    return left_right, centers


def run(opt):
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    scene_path = 'Cornell_Dataset_Annotation/rgbd'

    rgb_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".jpg")])
    depth_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".png")])
    meta = scio.loadmat('Cornell_Dataset_Annotation/Cornell.mat')
    depth_scale = meta['factor_depth']
    print("Depth Scale is: ", depth_scale)
    assert len(rgb_files) == len(depth_files)
    times = []

    for rgb_file, depth_file in zip(rgb_files, depth_files):
        verb_dir = "Cornell_Dataset_Annotation/annotation"
        file_prefix = os.path.splitext(rgb_file)[0]
        verb_file = f"{file_prefix}.json"
        verb_path = os.path.join(verb_dir, verb_file)
        with open(verb_path, "r") as f:
            data = json.load(f)

        verb = data.get("verb")

        rgb_path = os.path.join(scene_path, rgb_file)
        depth_path = os.path.join(scene_path, depth_file)

        rgb = Image.open(rgb_path).convert('RGB')
        color = np.array(rgb, dtype=np.float32)
        depth = np.array(Image.open(depth_path))

        start = time.time()

        # Convert images to numpy arrays
        depth_raw = depth * depth_scale
        depth = (depth_raw / depth_scale).astype(np.uint8)
        img = color

        # pre-process rgb and depth images
        inp_image = pre_process(img, depth)

        # pass the image into the network
        ret = detector.run(inp_image)
        ret = ret["results"]

        left_right, center = KpsToGrasppose(ret, img, depth_raw)

        send_data(center, rgb, verb, verb_file)
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        print(f"Cost: {elapsed:.4f} second")
        time.sleep(0.5)
    if times:
        avg_time = np.mean(times)
        print(f"Average cost: {avg_time:.4f} second")

if __name__ == "__main__":
    opt = opts().parse()

    try:
        run(opt)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting safely...")
    except Exception as e:
        print(f"{e}")

    finally:
        time.sleep(1)
        print("Shutdown complete.")
