import numpy as np
import cv2
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def filter_masks(masks, iou_th=0.8, inter_th=0.8):
    masks = list(masks)
    height, width = masks[0].shape

    num_masks = len(masks)
    keep_indices = set(range(num_masks))

    merged_point_indices = {i: [i] for i in range(num_masks)}
    mask_sizes = np.array([mask.sum() for mask in masks])

    for i in range(num_masks):
        if i not in keep_indices:
            continue

        if (mask_sizes[i] > height * width / 4) or (mask_sizes[i] < 900):
            keep_indices.discard(i)
            continue

        for j in range(i + 1, num_masks):
            if j not in keep_indices:
                continue

            iou = compute_iou(masks[i], masks[j])
            intersection = np.logical_and(masks[i], masks[j]).sum()

            if iou > iou_th or intersection / mask_sizes[i] >= inter_th or intersection / mask_sizes[j] >= inter_th:
                if mask_sizes[i] >= mask_sizes[j]:
                    keep_indices.discard(j)
                    merged_point_indices[i].extend(merged_point_indices[j])
                else:
                    keep_indices.discard(i)
                    merged_point_indices[j].extend(merged_point_indices[i])
                    break

    keep_indices_list = sorted(keep_indices)
    filtered_masks = np.array([masks[i] for i in keep_indices_list])
    discarded_point_indices_list = [sorted(list(set(merged_point_indices[i]))) for i in keep_indices_list]

    return filtered_masks, discarded_point_indices_list, keep_indices_list

def get_bounding_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, y_min = np.min(xs), np.min(ys)
    x_max, y_max = np.max(xs), np.max(ys)
    return (x_min, y_min, x_max, y_max)


def crop_bbox(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

def refine_masks(masks, min_contour_area=600):
    refined_masks = []
    for mask in masks:
        mask = mask.astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

        if not contours:
            refined_masks.append(np.zeros_like(mask))
            continue

        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, contours, -1, 255, thickness=cv2.FILLED)
        refined_masks.append(new_mask)

    return np.array(refined_masks)  # BxHxW