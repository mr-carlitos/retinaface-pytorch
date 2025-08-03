##### CARLOS CODE FILE ########
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import sys

from evaluation_mogface_final import get_preds, norm_score, get_gt_boxes, np_round, image_eval, voc_ap

def evaluate_model(pred_dir: str, facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list):
    pred = get_preds(pred_dir)
    pred = norm_score(pred)

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

        recall = tp_cum / float(count_face)
        precision = tp_cum / (tp_cum + fp_cum)  # will be envelope-corrected in voc_ap
        ap, _, _ = voc_ap(recall, precision)

        aps.append((recall, precision, ap))
    return aps


def main(checkpoints_root: str, gt_path: str, out_dir: str):
    # 1) Collect all model folders
    model_dirs = sorted(
        d for d in (os.path.join(checkpoints_root, f) for f in os.listdir(checkpoints_root))
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, 'like-mxnet'))
    )
    if not model_dirs:
        raise RuntimeError('No <model>/like-mxnet folders found under {}'.format(checkpoints_root))

    settings = ['easy', 'medium', 'hard']

    curves_per_setting = {s: {} for s in settings}

    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)

    # 2) Evaluate every model
    for model_path in model_dirs:
        model_name = os.path.basename(model_path)
        try:
            curves = evaluate_model(os.path.join(model_path, 'like-mxnet'), facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list)
        except Exception as e:
            print(f'✗ Skipping {model_name}: {e}', file=sys.stderr)
            continue
        for s, (recall, precision, ap) in zip(settings, curves):
            curves_per_setting[s][model_name] = (recall, precision, ap)

    # 3) Plot one figure per setting
    os.makedirs(out_dir, exist_ok=True)
    colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, setting in enumerate(settings):
        plt.figure(figsize=(7, 5))
        for jdx, (model_name, (recall, precision, ap)) in enumerate(curves_per_setting[setting].items()):
            colour = colour_cycle[jdx % len(colour_cycle)]
            plt.plot(recall, precision, label=model_name + " with AP " + "%.4f" % ap, linewidth=2, color=colour)

        plt.grid(True, linestyle='--', alpha=.6)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0, 1)
        plt.legend(loc='lower left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'precision_recall_{setting}.png'), dpi=300)
        plt.close()
        print(f'✓ saved {setting} plot → {out_dir}/precision_recall_{setting}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare all RetinaFace checkpoints.')
    parser.add_argument('--checkpoints', '-c',
                        default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/save-checkpoints/deconv_pool',
                        help='Folder that contains individual checkpoint dirs & Where to store the generated PNGs.')
    parser.add_argument('--gt', '-g',
                        default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/widerface_evaluate/ground_truth',
                        help='Path to WiderFace ground-truth directory.')

    args = parser.parse_args()
    main(args.checkpoints, args.gt, args.checkpoints)
