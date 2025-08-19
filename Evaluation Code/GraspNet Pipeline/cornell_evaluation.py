import io
import json
import os
import cv2
import sys
import time
import zlib

import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
from graspnet_pipeline import GraspNetPipeline


if __name__=='__main__':
    pipeline = GraspNetPipeline(
        checkpoint_path='dataset/model/checkpoint-rs.tar',
        workspace_mask_path='multiObj_multiGrasp_cornell/workspace_cornell_640_480.png',
        root_dir='~/graspnet-baseline/'
    )
    print("GraspNet initialized.")

    scene_path = 'multiObj_multiGrasp_cornell/Cornell_Dataset_Annotation/rgbd'

    rgb_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".jpg")])
    depth_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".png")])

    assert len(rgb_files) == len(depth_files)
    times = []
    for rgb_file, depth_file in zip(rgb_files, depth_files):
        file_prefix = os.path.splitext(rgb_file)[0]
        verb_file = f"{file_prefix}.json"
        verb_path = os.path.join('multiObj_multiGrasp_cornell/Cornell_Dataset_Annotation/annotation', verb_file)

        with open(verb_path, "r") as f:
            data = json.load(f)

        verb = data.get("verb")
        gt_object = data.get("object")
        if not verb or not gt_object:
            print('no gt verb or object')
            continue
        print("Verb:", verb)
        print("Object:", gt_object)
        rgb_path = os.path.join(scene_path, rgb_file)
        depth_path = os.path.join(scene_path, depth_file)
        meta = scio.loadmat('multiObj_multiGrasp_cornell/Cornell.mat')

        rgb = Image.open(rgb_path).convert('RGB')
        rgb_np = np.array(rgb)
        color = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        depth = np.array(Image.open(depth_path))

        start = time.time()
        grasps, cloud, points_2d, color_rgb = pipeline.run(color, depth, meta, gg_num=70, plot=False)

        # pipeline.save_grasp_scene(cloud, grasps, file_prefix)
        pipeline.send_grasps(points_2d, color_rgb, verb, verb_file)

        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        print(f"Cost: {elapsed:.4f} 秒")
        time.sleep(0.5)

    if times:
        avg_time = np.mean(times)
        print(f"Average cost: {avg_time:.4f} 秒")