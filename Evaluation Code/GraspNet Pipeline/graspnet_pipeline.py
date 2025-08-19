import io
import os
import sys
import cv2
import zlib
import struct
import numpy as np
import open3d as o3d
from PIL import Image
import torch
from graspnetAPI import GraspGroup
import socket
import pickle


ROOT_DIR = os.path.dirname('~/graspnet-baseline/')
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


class GraspNetPipeline:
    def __init__(self, checkpoint_path, workspace_mask_path, root_dir):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                            cylinder_radius=0.05, hmin=-0.02,
                            hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
        self.net.to(self.device)

        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()

        self.workspace_mask = np.array(Image.open(workspace_mask_path))

        sys.path.append(os.path.join(root_dir, 'models'))
        sys.path.append(os.path.join(root_dir, 'dataset'))
        sys.path.append(os.path.join(root_dir, 'utils'))

    def process_input(self, color, depth, meta):
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        camera = CameraInfo(
            width=color.shape[1], height=color.shape[0],
            fx=intrinsic[0][0], fy=intrinsic[1][1],
            cx=intrinsic[0][2], cy=intrinsic[1][2],
            scale=factor_depth
        )
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        mask = (self.workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        num_points = 20000
        if len(cloud_masked) >= num_points:
            idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        end_points = {
            'point_clouds': torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(self.device),
            'cloud_colors': color_sampled
        }

        cloud_vis = o3d.geometry.PointCloud()
        cloud_vis.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_vis.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        return end_points, cloud_vis

    def predict_grasps(self, end_points):
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        return GraspGroup(gg_array)

    def filter_collision(self, gg, cloud):
        detector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
        collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
        return gg[~collision_mask]

    def project_to_2d(self, grasp_3d, intrinsic):
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        X, Y, Z = grasp_3d[:, 0], grasp_3d[:, 1], grasp_3d[:, 2]
        u = (fx * X / Z + cx).astype(int)
        v = (fy * Y / Z + cy).astype(int)
        return np.column_stack((u, v))

    def send_grasps(self, bboxes, rgb_image, verb, verb_file):
        img_pil = Image.fromarray(rgb_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        data = pickle.dumps((bboxes, img_bytes, verb, verb_file))
        data = zlib.compress(data)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('255.255.255.255', 5000))  # change here with your server address
            data_len = len(data)
            s.sendall(struct.pack('>I', data_len))
            s.sendall(data)
            print(f"Sent {data_len} bytes to server, waiting for associated_point_ids...")

    def plot_one_grasp(self, grasp, cloud):
        grippers = grasp.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    def send_grasps_wait_return(self, bboxes, rgb_image, verb, timestamp):
        img_pil = Image.fromarray(rgb_image)
        img_bytes_io = io.BytesIO()
        img_pil.save(img_bytes_io, format="PNG")
        img_bytes = img_bytes_io.getvalue()

        data = pickle.dumps((bboxes, img_bytes, verb, timestamp))
        data = zlib.compress(data)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('255.255.255.255', 5000))  # change here with your server address

            data_len = len(data)
            s.sendall(struct.pack('>I', data_len))
            s.sendall(data)
            print(f"Sent {data_len} bytes to server, waiting for associated_point_ids...")

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

    def run(self, color, depth, meta, gg_num=50, plot=False):
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color_norm = color_rgb.astype(np.float32) / 255.0
        end_points, cloud = self.process_input(color_norm, depth, meta)
        gg = self.predict_grasps(end_points)
        gg = self.filter_collision(gg, np.asarray(cloud.points))
        gg.nms()
        gg.sort_by_score()

        top_gg = gg[:gg_num]
        grasp_centers_3d = np.array([grasp.translation for grasp in top_gg])
        bboxes_2d = self.project_to_2d(grasp_centers_3d, meta['intrinsic_matrix'])

        if plot:
            plot_gg = gg[:gg_num]
            grippers = plot_gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])

        return top_gg, cloud, bboxes_2d, color_rgb

    def save_grasp_scene(self, cloud, gg, save_dir):

        grippers = gg.to_open3d_geometry_list()

        cloud_path = os.path.join(save_dir, "cloud.ply")
        o3d.io.write_point_cloud(cloud_path, cloud)
        print(f"Saved point cloud: {cloud_path}")

        for i, gripper in enumerate(grippers):
            gripper_path = os.path.join(save_dir, f"gripper_{i}.ply")
            o3d.io.write_triangle_mesh(gripper_path, gripper)
        print(f"Saved {len(grippers)} grippers to {save_dir}")
