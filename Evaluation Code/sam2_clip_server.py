import os
import struct
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import io
import zlib
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import socket
import pickle
from util import *
import clip
import time
import torch.nn.functional as F


def classify_images_with_clip_batch(cropped_images, objects_list):
    # Encode all text prompts
    text_tokens = clip.tokenize(objects_list).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Encode all images
    image_tensors = []
    for img_np in cropped_images:
        img_pil = Image.fromarray(img_np)
        img_tensor = preprocess(img_pil).unsqueeze(0)  
        image_tensors.append(img_tensor)
    image_batch = torch.cat(image_tensors, dim=0).to(device) 

    with torch.no_grad():
        image_features = clip_model.encode_image(image_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        likelihoods = text_features @ image_features.T
        likelihoods = likelihoods.cpu().numpy()
        image_scores = np.max(likelihoods, axis=0)
        pred = np.argmax(image_scores)
    return pred


def receive_and_process_grasps(predictor, verb_object_dict):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("Waiting for connection...")
        count = 0
        while True:  
            conn, addr = s.accept()
            with conn:
                print(f"Connect from {addr}")
                raw_len = conn.recv(4)
                if len(raw_len) < 4:
                    print("Did not receive length prefix.")
                    continue
                data_len = struct.unpack('>I', raw_len)[0]
                data = b""
                while len(data) < data_len:
                    packet = conn.recv(min(4096, data_len - len(data)))
                    if not packet:
                        print("Connection closed before all data received.")
                        break
                    data += packet

                if len(data) < data_len:
                    print(" Incomplete data received")
                    continue

                start_time = time.time()
                points, rgb, verb, timestamp = pickle.loads(zlib.decompress(data))
                print(f"Received {len(points)} grasp points from client.")

                process_with_sam(predictor, points, rgb, verb, count, conn, verb_object_dict, timestamp)
                count += 1
                print("--------- %s seconds ---------" % (time.time() - start_time))
    
def process_with_sam(predictor, points, rgb, verb, count, conn, verb_object_dict, timestamp):  
    image = np.array(Image.open(io.BytesIO(rgb)).convert("RGB"))
    point_coords = np.array(points).reshape(-1, 1, 2)
    point_labels = np.ones((len(points), 1), dtype=np.int32)
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=True
    )
    if masks.ndim == 3:
        # shape = (3, H, W) -> reshape to (1, 3, H, W)
        masks = masks[np.newaxis, ...]
        scores = scores[np.newaxis, ...]
    print(masks.shape, scores.shape)

    # Select the best single mask per object
    best_masks = masks[np.arange(len(masks)), np.argmax(scores, axis=-1)]

    filtered_masks, filtered_point_list, keep_indices_list = filter_masks(best_masks)

    refined_masks = refine_masks(filtered_masks)
    
    bounding_boxes = [get_bounding_box(mask) for mask in refined_masks]
    cropped_images = [crop_bbox(image, bbox) for bbox in bounding_boxes]
    assert len(cropped_images) == len(bounding_boxes), f"Length mismatch: {len(cropped_images)} cropped vs {len(bounding_boxes)} bounding boxes"

    if len(cropped_images) == 0:
        data_to_send = zlib.compress(pickle.dumps([]))
        data_len = len(data_to_send)
        conn.sendall(struct.pack('>I', data_len))
        conn.sendall(data_to_send)
        print("Sent empty associated_point_ids back to client due to no valid masks.")
        return

    objects_list = verb_object_dict.get(verb, [])

    best_image_idx = classify_images_with_clip_batch(cropped_images, objects_list)
    associated_point_ids = filtered_point_list[best_image_idx]
    print("Selected associated_point_ids:", associated_point_ids)

    data_to_send = zlib.compress(pickle.dumps(associated_point_ids))
    data_len = len(data_to_send)
    conn.sendall(struct.pack('>I', data_len))
    conn.sendall(data_to_send)

    print("Sent associated_point_ids back to client.")
    print("Waiting for next data...")


if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 5000
    count = 0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('device: ', device)

    text_label = []
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    print("SAM model loaded.")

    with open('real_world_craft_e.json', 'r') as f:
        verb_object_dict = json.load(f)

    receive_and_process_grasps(predictor, verb_object_dict)


