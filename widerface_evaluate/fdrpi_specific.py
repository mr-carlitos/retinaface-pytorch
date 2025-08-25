# CARLOS CODE FILE: INSPIRED FROM FURKANS PAPER
# We compute the TPDR-FPDPI evaluation results (and plot) for a specific model variant.

import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import box_iou
import torch
from termcolor import cprint
import logging
import argparse
import os
import pickle
import tqdm
from evaluation_mogface_final import get_preds, norm_score, get_gt_boxes, np_round, image_eval

def get_iou_pairwise(ground_truth_bboxes, detection_bboxes):
    """
    It computes the standard iou scores between ground truth and detection bboxes
    """
    # convert it to tensor and x1,y1,x2,y2 format
    gr = torch.Tensor(ground_truth_bboxes)
    gr[:, 2] += gr[:, 0]
    gr[:, 3] += gr[:, 1]
    dt = torch.Tensor(detection_bboxes)
    dt[:, 2] += dt[:, 0]
    dt[:, 3] += dt[:, 1]

    assert gr.shape == (len(gr), 4) and dt.shape == (len(dt), 4)

    # get the pairwise of iou scores between ground truth and detections
    iou_pairwise = box_iou(gr, dt)

    return iou_pairwise.numpy()


def assign_detections(ground_truth, proposals, overlap_threshold=0.5, exclude=None):
    """
    It matches bounding boxes(proposals) with ground truth (if possible) or assigns it as a misdetection.
    Propasals can come from the detection files or scoring files

    A detection can match with only one unique ground truth in this matching procedure.

    Note that gallery faces will be excluded from the results when their face_ids are given in the the parameter of 'exclude'.
    """
    assigned_detections = {}
    # store all detection scores [scores..],matched_overlaps -> [(indice_proposal,subject_id),..], misdetections-> [indices..]
    for img, (face_ids, subject_ids, ground_truth_bboxes) in ground_truth.items():

        if img not in proposals:
            continue

        detection_scores, prop_bboxes, _ = proposals[img]
        prop_bboxes = np.array(prop_bboxes)

        iou_pairwise = get_iou_pairwise(ground_truth_bboxes, prop_bboxes)

        # initialize arrays to keep track of matches
        matched_indices = []  # store pairs of matched ground truth and proposal indices
        unmatched_detection_indices = list(range(len(prop_bboxes)))  # initially, all proposals are unmatched

        while True:
            # find the maximum IoU in the current IoU matrix
            max_iou = np.max(iou_pairwise)

            if max_iou < overlap_threshold:
                break  # exit if no IoU above the threshold is found

            # find the indices of the maximum IoU
            gt_index, proposal_index = np.unravel_index(np.argmax(iou_pairwise), iou_pairwise.shape)

            # exclude gallery faces if it is given
            if exclude and face_ids[gt_index] in exclude:
                # if it is matched, dont add the detection to false positives as well as true positives
                unmatched_detection_indices.remove(proposal_index)
                iou_pairwise[gt_index, :] = 0
                iou_pairwise[:, proposal_index] = 0
                continue

            # add the matched pair to the list -> proposal index : subject_id of the ground truth
            matched_indices.append((proposal_index, subject_ids[gt_index]))
            unmatched_detection_indices.remove(proposal_index)

            # set the IoU values for the corresponding ground truth and proposals to 0
            iou_pairwise[gt_index, :] = 0
            iou_pairwise[:, proposal_index] = 0

        # add the image results to the dictionary
        assigned_detections[img] = (detection_scores, matched_indices, unmatched_detection_indices)

    return assigned_detections


def compute_DR_FDPI(all_matched_detections, face_numbers, image_numbers, plot_detection_numbers=False):
    """
    It computes Detection Rate (true positive rate) and False Detection Per Image (FDPI) for plotting.
    """

    positives = []
    negatives = []

    for _, (detection_scores, matched_indices, misdetections) in all_matched_detections.items():
        # detections matched with a ground-truth is assigned as a positive score
        pos = [detection_scores[detection_ind] for detection_ind, _ in matched_indices]
        positives.extend(pos)

        # detections that are not matched with any of ground truth
        neg = [detection_scores[detection_ind] for detection_ind in misdetections]
        negatives.extend(neg)

    cprint(f"In total: {face_numbers} faces, {len(positives)} detected faces and {len(negatives)} false detections",
           'green', attrs=['bold', 'underline'])
    positives = np.array(positives)
    negatives = np.array(negatives)

    # thresholds
    # thresholds = np.linspace(min(negatives), max(negatives), 100)
    thresholds = np.unique(negatives)

    # detection rate and false detection per image
    DR = []
    FDPI = []

    for thr in thresholds:

        dr = np.sum(positives >= thr)

        if not plot_detection_numbers:
            # get the rate
            dr /= face_numbers

        fdpi = np.sum(negatives >= thr) / image_numbers  # the number of images in the probe

        DR.append(dr)
        FDPI.append(fdpi)

    return DR, FDPI


def plot_froc_curve(results, labels, saving_path, face_numbers, linear=False, plot_detection_numbers=False):
    """
    This function plots F-ROC curve (DR-FDPI) based on given results

    results: a list of [DR,FDPI]s
    """
    # create the figure
    figure = plt.figure(figsize=(7, 4.5))
    # set the plotter function based on the linear flag
    plotter = plt.semilogx if not linear else plt.plot

    max_fd = 0
    min_fd = np.inf
    for ix, label in enumerate(labels):
        plotter(results[ix][1], results[ix][0], label=label)

        max_fd = max(max_fd, results[ix][1][0])  # the first index is the biggest because its thr was the smallest
        min_fd = min(min_fd, results[ix][1][-1])

    plt.grid(True, color=(0.6, 0.6, 0.6))

    plt.legend(loc=2 if not linear else 4, prop={'size': 14})
    plt.xlabel('False Detection Per Image')

    if not linear:
        plt.xlim([min_fd, max_fd])
    else:
        plt.xlim([0, max_fd])

    if plot_detection_numbers:
        plt.ylim((0, face_numbers))
        plt.ylabel('Number of Detections')
    else:
        plt.ylim((0, 1))
        plt.ylabel('Detection Rate')

    plt.tight_layout()
    plt.savefig(saving_path)


def main(pred, gt_path):
    # get config arguments

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
        count_images = 0
        # pr_curve = np.zeros((thresh_num, 2)).astype('float')
        pr_curve_list = []
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))

        positives = []  # scores of true positive detections
        negatives = []  # scores of false positive detections

        # for each setting (e.g., Parade)
        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]

            # for each image in the current setting
            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)
                count_images += 1

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

                valid_scores = pred_info[valid_mask, 4]
                valid_recall = pred_recall[valid_mask]

                # Separate positives (TP) and negatives (FP)
                tp_mask = np.zeros_like(valid_recall, dtype=bool)
                tp_mask[0] = valid_recall[0] > 0
                tp_mask[1:] = np.diff(valid_recall) > 0

                positives.extend(valid_scores[tp_mask].tolist())
                negatives.extend(valid_scores[~tp_mask].tolist())

        positives = np.array(positives)
        negatives = np.array(negatives)

        # Use unique negative scores as thresholds
        thresholds = np.unique(negatives) if len(negatives) > 0 else [0]

        # Sort arrays once
        pos_sorted = np.sort(positives) # descending order
        neg_sorted = np.sort(negatives)  # descending order

        pos_counts = len(pos_sorted) - np.searchsorted(pos_sorted, thresholds, side='left')
        neg_counts = len(neg_sorted) - np.searchsorted(neg_sorted, thresholds, side='left')

        DR = pos_counts / float(count_face)
        FDPI = neg_counts / count_images

        aps.append((DR, FDPI))

    for (DR, FDPI), name in zip(aps, settings):
        plt.figure(figsize=(6, 4))
        plt.semilogx(FDPI, DR, linewidth=2)
        plt.grid(True, linestyle='--', alpha=.6)
        plt.xlabel('False Detections per Image')
        plt.ylabel('Detection Rate')
        plt.title(f'F-ROC ({name.capitalize()})')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'froc_{name}.png', dpi=300)
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/save-checkpoints/2025-06-03_NEIGHBOURHOOD/like-mxnet/')
    parser.add_argument('-g', '--gt', default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/widerface_evaluate/ground_truth')
    #parser.add_argument('-i', '--iter', default='140')
    #parser.add_argument('-d', '--det_result_txt', default=None)

    args = parser.parse_args()
    main(args.pred, args.gt)
