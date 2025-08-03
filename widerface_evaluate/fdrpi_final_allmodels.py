##### CARLOS CODE FILE ########
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import sys

from evaluation_mogface_final import get_preds, norm_score, get_gt_boxes, np_round, image_eval

def evaluate_model(pred_dir: str, facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list):
    """Return three (DR, FDPI) tuples – Easy, Medium, Hard – for one model."""
    pred = norm_score(get_preds(pred_dir))

    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]

    curves = []  # [(DR, FDPI), …]

    for setting_id in range(3):
        iou_th = 0.5
        gt_list = setting_gts[setting_id]
        count_face = 0
        count_images = 0
        positives, negatives = [], []

        pbar = tqdm.tqdm(range(len(event_list)), leave=False, desc=f'[{os.path.basename(pred_dir)}] {settings[setting_id].capitalize():6}')

        for i in pbar:
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]

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

                pred_info = np_round(pred_info, 1)[np.argsort(pred_info[:, 4])][::-1]
                gt_boxes = np_round(gt_boxes)
                ignore = np_round(ignore)
                pred_recall, prop_l = image_eval(pred_info, gt_boxes, ignore, iou_th)

                valid_mask = (prop_l == 1)
                if not np.any(valid_mask):
                    continue

                scores = pred_info[valid_mask, 4]
                recalls = pred_recall[valid_mask]

                tp_mask = np.zeros_like(recalls, dtype=bool)
                tp_mask[0] = recalls[0] > 0
                tp_mask[1:] = np.diff(recalls) > 0

                positives.extend(scores[tp_mask])
                negatives.extend(scores[~tp_mask])

        positives = np.asarray(positives)
        negatives = np.asarray(negatives)
        thr = np.unique(negatives) if negatives.size else np.array([0])

        # Sort arrays once
        pos_sorted = np.sort(positives)  # descending order
        neg_sorted = np.sort(negatives)  # descending order

        pos_counts = len(pos_sorted) - np.searchsorted(pos_sorted, thr, side='left')
        neg_counts = len(neg_sorted) - np.searchsorted(neg_sorted, thr, side='left')

        DR = pos_counts / float(max(count_face, 1))
        FDPI = neg_counts / max(count_images, 1)

        curves.append((DR, FDPI))
    return curves


def main(checkpoints_root: str, gt_path: str, out_dir: str):
    # 1) Collect all model folders
    model_dirs = sorted(
        d for d in (os.path.join(checkpoints_root, f) for f in os.listdir(checkpoints_root))
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, 'like-mxnet'))
    )
    if not model_dirs:
        raise RuntimeError('No <model>/like-mxnet folders found under {}'.format(checkpoints_root))

    settings = ['easy', 'medium', 'hard']
    # store curves per setting: dict['easy'] = {'model_name': (DR, FDPI), ...}
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
        for s, (DR, FDPI) in zip(settings, curves):
            curves_per_setting[s][model_name] = (DR, FDPI)

    # 3) Plot one figure per setting
    os.makedirs(out_dir, exist_ok=True)
    colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, setting in enumerate(settings):
        plt.figure(figsize=(7, 5))
        for jdx, (model_name, (DR, FDPI)) in enumerate(curves_per_setting[setting].items()):
            colour = colour_cycle[jdx % len(colour_cycle)]
            plt.semilogx(FDPI, DR, label=model_name, linewidth=2, color=colour)

        plt.grid(True, linestyle='--', alpha=.6)
        plt.xlabel('False Positive Detections Per Image')
        plt.ylabel('True Positive Detection Rate')
        plt.ylim(0, 1)
        plt.legend(loc='lower left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fpdpi-tpdr_{setting}.png'), dpi=300)
        plt.close()
        print(f'✓ saved {setting} plot → {out_dir}/fpdpi-tpdr_{setting}.png')


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
