"""
CARLOS CODE FILE
I modified the original evaluation script so that we don't use the fixed intervals of 0.001 (from 0 to 1), but that we sort along the confidence scores.
"""

from __future__ import absolute_import
import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps

def np_around(array, num_decimals=0):
    return array

def np_round(val, decimals=4):
    return val

def get_gt_boxes(gt_dir):
    """
    Load the ground truth annotations from the WiderFace .mat files.

    Parameters:
        gt_dir (str): Directory containing the following MATLAB files:
                      'wider_face_val.mat', 'wider_easy_val.mat',
                      'wider_medium_val.mat', and 'wider_hard_val.mat'.

    Returns:
        tuple: A tuple containing six elements:
            - facebox_list: Array of ground truth face bounding boxes.
            - event_list: Array of event identifiers. This array lists the "events" (groups or categories) in the dataset. Each event groups a collection of images.
            - file_list: Array of image file names per event.
            - hard_gt_list: Array of indices/flags for hard faces.
            - medium_gt_list: Array of indices/flags for medium faces.
            - easy_gt_list: Array of indices/flags for easy faces.
    """

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list

def read_pred_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ') if a], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    '''
    Walks through subdirectories (organized by “event”) in the predictions folder and reads all prediction files.
    It builds a nested dictionary structure: top level keys are event names, and each value is a dictionary mapping image names (without extension) to their prediction arrays.

    Args:
        pred_dir (str): The path to the directory containing event subdirectories with prediction text files.

    Returns:
        boxes (dict): A nested dictionary where each key is an event name, and the corresponding value is another dictionary.
            This inner dictionary maps image names (without file extension) to their prediction arrays (NumPy arrays).

    '''
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """
    Normalize detection confidence scores to [0, 1].

    Args:
        pred (dict): Nested dictionary with structure:
                     { event_name: { image_name: np.array([[x1, y1, x2, y2, s], ...]) } },
                     where each row corresponds to a detection box and 's' is its confidence score.

    Returns:
        dict: The input 'pred' with each confidence score rescaled so that the global minimum becomes 0 and the global maximum becomes 1.
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score).astype(np.float64) / diff
    return pred


def image_eval(pred, gt, ignore, iou_thresh):
    """
    Evaluate detections on a single image by matching predicted boxes to ground truth.

    This function converts boxes from (x, y, w, h) to (x1, y1, x2, y2), computes the IoU
    between each prediction and each ground truth box, and then, for each prediction,
    determines if it matches a valid ground truth (based on the IoU threshold and the 'ignore' flag).
    It returns a cumulative count of matched ground truths and a flag array for valid predictions.

    Args:
        pred (np.ndarray): Array of predicted boxes with shape (N, 5) in the format [x, y, w, h, score].
        gt (np.ndarray): Array of ground truth boxes with shape (M, 4) in the format [x, y, w, h].
        ignore (np.ndarray): Array of length M indicating if a ground truth box should be ignored.
        iou_thresh (float): IoU threshold to consider a prediction as matching a ground truth box.

    Returns:
        tuple:
            - pred_recall (np.ndarray): Cumulative count of valid ground truths matched per prediction. -> Computes basically the True Positives (TPs)
            - proposal_list (np.ndarray): Array indicating for each prediction if it is valid (or flagged as ignored).
    """

    _pred = pred.copy()
    _gt = gt.copy()

    # Array to track cumulative count of recalled ground truths for each prediction
    pred_recall = np.zeros(_pred.shape[0])
    # Tracks which ground truth boxes have been recalled (0 = not recalled, 1 = recalled, -1 = ignored)
    recall_list = np.zeros(_gt.shape[0])
    # Tracks which predictions are valid (1 = valid, -1 = ignored/invalid)
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    # overlaps -> Result is an N×M matrix where N is the number of predictions and M is the number of ground truth
    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    # For each prediction (h), find the ground truth with maximum overlap
    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            # If the ground truth hasn't been claimed yet (recall_list[max_idx] == 0), mark it as recalled
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)

    # pred_recall: For each prediction, gives the count of valid ground truths matched up to that point -> Computes basically the True Positives (TPs)
    # proposal_list: For each prediction, indicates if it's valid (1) or should be ignored (-1)
    return pred_recall, proposal_list


def img_pr_info(false_positive_scores, pred_info, proposal_list, pred_recall):
    """
    Compute per-image precision and recall counts over a set of confidence thresholds.

    For each of the 'thresh_num' thresholds (ranging from near 1 down to 0),
    this function:
      - Finds the last prediction with a confidence score above the threshold.
      - Counts valid predictions (as indicated in 'proposal_list') up to that point.
      - Retrieves the cumulative matched value from 'pred_recall'.

    Args:
        #ADD false_positive_scores
        pred_info (np.ndarray): Array of detections (shape: Nx5), where column 4 holds confidence scores.
        proposal_list (np.ndarray): Array indicating valid predictions (1) or invalid ones.
        pred_recall (np.ndarray): Array of cumulative recall counts for predictions.

    Returns:
        np.ndarray: A (thresh_num x 2) array where each row contains:
                    [number of valid proposals, cumulative recall] for that threshold.
    """
    pr_info = np.zeros((len(false_positive_scores), 3)).astype('float')
    t = 0
    for thresh in false_positive_scores:
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index) # Number of proposals or "Predicted Positive" (TPs + FPs) (at threshold t)
            pr_info[t, 1] = pred_recall[r_index] # Number of TPs (at threshold t)
        pr_info[t, 2] = thresh
        t += 1
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    """
        Compute normalized precision and recall at each confidence threshold.

        For each threshold, precision is calculated as the number of recalled detections
        divided by the number of valid proposals, and recall is calculated as the number
        of recalled detections divided by the total number of ground truth faces.

        Precision: How many of the valid detections were actually correct.
        Recall: What fraction of all the ground truth objects have been detected.

        Args:
            thresh_num (int): Number of confidence thresholds.
            pr_curve (np.ndarray): Array of shape (thresh_num, 2) with raw counts:
                                   [valid proposals, cumulative recall] per threshold.
            count_face (int): Total number of ground truth faces in the dataset.

        Returns:
            np.ndarray: Array of shape (thresh_num, 2) where each row contains:
                        [precision, recall] for that threshold.
        """
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = round(pr_curve[i, 1] / pr_curve[i, 0], 4) # TPs / (TPs + FPs) = TPs / PREDICTED_POSTIVES
        _pr_curve[i, 1] = round(pr_curve[i, 1] / count_face, 4) # TPs / (TPs + FNs) = TPs / POSITIVES
    return _pr_curve


def voc_ap(rec, prec):
    """
    Calculate Average Precision

    The function computes the area under the precision-recall curve by:
    1. Adding sentinel values to the precision and recall arrays
    2. Making the precision monotonically decreasing
    3. Finding points where recall changes
    4. Computing the area as a weighted sum of precision values

    Args:
        rec (np.ndarray): Array of recall values (between 0 and 1)
        prec (np.ndarray): Array of precision values (between 0 and 1)

    Returns:
        float: The average precision value (area under the PR curve)
    """
    #check_monotonicity(rec, prec)

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        # creates a step-like function where precision only decreases as recall increases
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    # These are the transition points where we need to calculate the area
    #test = mrec[1:]
    #test2 = mrec[:-1]
    #test3 = mrec[1:] != mrec[:-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    # Each rectangle's width is the change in recall (mrec[i + 1] - mrec[i])
    # Each rectangle's height is the precision at that recall level (mpre[i + 1])
    ap = np_round(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def check_monotonicity(rec, prec):
    """
    Check if recall is monotonically increasing and precision is monotonically decreasing.

    Args:
        rec (np.ndarray): Array of recall values
        prec (np.ndarray): Array of precision values

    Returns:
        tuple: (recall_is_increasing, precision_is_decreasing)
    """
    # Check if recall is monotonically increasing
    recall_diffs = np.diff(rec)
    recall_is_increasing = np.all(recall_diffs >= 0)

    # Check if precision is monotonically decreasing
    precision_diffs = np.diff(prec)
    precision_is_decreasing = np.all(precision_diffs <= 0)

    # Print detailed information
    if not recall_is_increasing:
        non_increasing_indices = np.where(recall_diffs < 0)[0]
        print(f"Recall is not monotonically increasing at indices: {non_increasing_indices}")
        print(f"Values at these points: {[(i, rec[i], rec[i + 1]) for i in non_increasing_indices]}")

    if not precision_is_decreasing:
        non_decreasing_indices = np.where(precision_diffs > 0)[0]
        print(f"Precision is not monotonically decreasing at indices: {non_decreasing_indices}")
        print(f"Values at these points: {[(i, prec[i], prec[i + 1]) for i in non_decreasing_indices]}")

    return recall_is_increasing, precision_is_decreasing


def evaluation_ap50(pred, gt_path):
    pred = get_preds(pred)
    pred = norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)

    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        # different setting
        iou_th = 0.5
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        #pr_curve = np.zeros((thresh_num, 2)).astype('float')
        pr_curve_list = []
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))

        scores_buf = []  # list of 1-D arrays
        tp_flags_buf = []  # list of 1-D uint8 arrays

        # for each setting (e.g., Parade)
        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]

            #for each image in the current setting
            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_info = np_round(pred_info, 1)
                pred_sort_idx = np.argsort(pred_info[:, 4])
                pred_info = pred_info[pred_sort_idx][::-1]

                gt_boxes = np_round(gt_boxes)
                ignore = np_round(ignore)
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_th)

                # keep only the predictions that are to be evaluated
                valid_mask = (proposal_list == 1)
                if not np.any(valid_mask):
                    continue  # nothing to record for this image

                valid_scores = pred_info[valid_mask, 4]  # (K,)
                valid_recall = pred_recall[valid_mask]  # (K,)

                # a TP is the first time the per-image recall increases
                tp_flags_img = np.empty_like(valid_recall, dtype=np.uint8)
                tp_flags_img[0] = 1 if valid_recall[0] > 0 else 0
                tp_flags_img[1:] = (np.diff(valid_recall) > 0).astype(np.uint8)

                # accumulate
                scores_buf.append(valid_scores)
                tp_flags_buf.append(tp_flags_img)

        # concatenate everything once
        scores = np.concatenate(scores_buf, axis=0).astype(np.float32)
        tp_flags = np.concatenate(tp_flags_buf, axis=0).astype(np.uint8)

        # sort detections by confidence (high to low)
        order = scores.argsort()[::-1]
        tp_flags = tp_flags[order]

        tp_cum = np.cumsum(tp_flags)  # true positives so far
        fp_cum = np.arange(1, tp_flags.size + 1) - tp_cum  # false positives so far

        recall = tp_cum / float(count_face)  # monotone :D
        precision = tp_cum / (tp_cum + fp_cum)  # will be envelope-corrected in voc_ap
        ap = voc_ap(recall, precision)
        """
        import matplotlib.pyplot as plt

        # Plot PR curve for current setting
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'{settings[setting_id].capitalize()} (AP={ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {settings[setting_id].capitalize()}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # Save individual PR curve
        plt.savefig(f'pr_curve_{settings[setting_id]}.png', dpi=300, bbox_inches='tight')
        plt.close()
        """

        aps.append(ap)

    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    print("Medium Val AP: {}".format(aps[1]))
    print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/save-checkpoints/2025-06-09_CROSSATTENTION_FROMUPPERANDLOWER_PYRAMIDIAL_NOPOSBIAS/like-mxnet/')
    parser.add_argument('-g', '--gt', default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/widerface_evaluate/ground_truth')
    #parser.add_argument('-i', '--iter', default='140')
    #parser.add_argument('-d', '--det_result_txt', default=None)

    args = parser.parse_args()
    #evaluation_ap50(args.pred, args.gt, args.iter, args.det_result_txt)
    evaluation_ap50(args.pred, args.gt)