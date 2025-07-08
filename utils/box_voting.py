##### CARLOS CODE FILE ########
## Based on box voting implementation from original mxnet RetinaFace implementation: https://github.com/deepinsight/insightface/blob/master/detection/retinaface/retinaface.py

import numpy as np

def bbox_vote(dets, nms_thresh, keep_top_k):
    """
    Box voting: merge overlapping boxes by weighted averaging.
    Args:
        dets: numpy array of shape (N, 5) with detection boxes: [x1, y1, x2, y2, score]
        nms_thresh: IoU threshold for merging (e.g., 0.4)
        keep_top_k: maximum number of detections to keep (e.g., 750)
    Returns:
        merged detections, numpy array of shape (M, 5)
    """
    if dets.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)
    # Sort detections in descending order by score.
    order = dets[:, 4].argsort()[::-1]
    dets = dets[order]
    merged_dets = []
    while dets.shape[0] > 0:
        # Early exit if we already have enough detections
        #if len(merged_dets) >= keep_top_k:
        #    break
        # Use the first (highest scored) detection as the reference.
        ref_box = dets[0]
        xx1 = np.maximum(ref_box[0], dets[:, 0])
        yy1 = np.maximum(ref_box[1], dets[:, 1])
        xx2 = np.minimum(ref_box[2], dets[:, 2])
        yy2 = np.minimum(ref_box[3], dets[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        area_ref = (ref_box[2] - ref_box[0] + 1) * (ref_box[3] - ref_box[1] + 1)
        areas = (dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1)
        ious = inter / (area_ref + areas - inter)

        # Find indices with IoU above threshold.
        merge_inds = np.where(ious >= nms_thresh)[0]
        boxes_to_merge = dets[merge_inds]
        # Remove these detections from the pool.
        dets = np.delete(dets, merge_inds, axis=0)

        if boxes_to_merge.shape[0] == 1:
            merged_dets.append(ref_box)
        else:
            # Weight each coordinate by its detection score.
            scores = boxes_to_merge[:, 4:5]
            weighted_coords = np.sum(boxes_to_merge[:, :4] * scores, axis=0) / np.sum(scores)
            max_score = np.max(boxes_to_merge[:, 4])
            merged_box = np.hstack((weighted_coords, max_score))
            merged_dets.append(merged_box)
    if len(merged_dets) > 0:
        merged_dets = np.vstack(merged_dets)
        order_final = merged_dets[:, 4].argsort()[::-1]
        merged_dets = merged_dets[order_final]
        merged_dets = merged_dets[:keep_top_k]
        return merged_dets
    else:
        return np.zeros((0, 5), dtype=np.float32)