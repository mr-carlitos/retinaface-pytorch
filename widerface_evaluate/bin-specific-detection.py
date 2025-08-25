##### CARLOS CODE FILE ########
"""
Compute and plot per-bin Hard-GT recall on WIDER-Face validation at 0.90 / 0.95 confidence.
"""

import os, argparse, numpy as np, matplotlib.pyplot as plt
from scipy.io import loadmat
from bbox import bbox_overlaps


def canonical_key(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name))[0]

def clean_matlab_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode('utf-8')
    x = str(x)
    if x.startswith("b'") and x.endswith("'"):
        return x[2:-1]
    return x

def _to_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    if a.size == 0:
        return a.reshape(0, 4)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a

def _valid_idx(idx, n):
    idx = np.asarray(idx, int).ravel()
    return idx[(idx >= 0) & (idx < n)]

def get_gt(gt_root):
    m  = loadmat(os.path.join(gt_root, 'wider_face_val.mat'))
    hm = loadmat(os.path.join(gt_root, 'wider_hard_val.mat'))
    return m['face_bbx_list'], m['event_list'], m['file_list'], hm['gt_list']

def read_pred(txt_path):
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    key  = canonical_key(lines[0])
    rows = [list(map(float, l.split())) for l in lines[2:]]
    return key, np.asarray(rows, np.float32)

def get_preds(pred_root):
    pred = {}
    for ev in os.listdir(pred_root):
        ev_path = os.path.join(pred_root, ev)
        if not os.path.isdir(ev_path):
            continue
        p_dict = {}
        for txt in (f for f in os.listdir(ev_path) if f.endswith('.txt')):
            k, b = read_pred(os.path.join(ev_path, txt))
            p_dict[k] = b
        pred[ev] = p_dict
    return pred

def lookup_in_event(ev_dict: dict, gt_key: str) -> np.ndarray:
    if gt_key in ev_dict:
        return ev_dict[gt_key]
    for pk in ev_dict:
        if pk.endswith(gt_key):
            return ev_dict[pk]
    return np.empty((0, 5), dtype=np.float32)

def main(checkpoints_root, gt_root, out_dir, debug=False):
    os.makedirs(out_dir, exist_ok=True)

    face, events, file_list, hard = get_gt(gt_root)
    n_events = len(events)

    bins       = [(8,16), (16,32), (32,64), (64,128), (128, np.inf)]
    bin_labels = ['8–16', '16–32', '32–64', '64–128', '128+']
    thresholds = [0.3, 0.90, 0.95]

    model_dirs = sorted(
        os.path.join(checkpoints_root, n)
        for n in os.listdir(checkpoints_root)
        if os.path.isdir(os.path.join(checkpoints_root, n))
    )
    if not model_dirs:
        raise RuntimeError(f'No subfolders in {checkpoints_root}')

    # pre-compute total Hard-GT per bin
    total_gt = np.zeros(len(bins), int)
    for ei in range(n_events):
        fboxes, hidxs = face[ei][0], hard[ei][0]
        for im_i in range(len(fboxes)):
            gt_wh = _to_2d(fboxes[im_i][0])
            hidx  = _valid_idx(hidxs[im_i][0]-1, gt_wh.shape[0])
            if hidx.size == 0:
                continue
            sizes = np.sqrt(np.clip(gt_wh[hidx,2]*gt_wh[hidx,3], 0, None))
            for bi,(lo,hi) in enumerate(bins):
                total_gt[bi] += ((sizes > lo) & (sizes <= hi)).sum()

    print('Total GT per bin:', dict(zip(bin_labels, total_gt)), '\n')

    # run models
    results = {thr:{} for thr in thresholds}
    for mdir in model_dirs:
        mname     = os.path.basename(mdir)
        preds     = get_preds(os.path.join(mdir, 'like-mxnet'))
        print(f'▶ MODEL {mname}: predictions for {len(preds)} events.')

        for thr in thresholds:
            hits = np.zeros(len(bins), int)
            for ei in range(n_events):
                ev_name = clean_matlab_str(events[ei][0][0])
                p_event = preds.get(ev_name, {})
                fboxes, hidxs = face[ei][0], hard[ei][0]

                for im_i in range(len(fboxes)):
                    gt_key = clean_matlab_str(file_list[ei][0][im_i][0][0])
                    pb     = lookup_in_event(p_event, gt_key)
                    keep   = pb[pb[:,4] >= thr]
                    if keep.size == 0:
                        continue

                    kxyxy = np.column_stack([
                        keep[:,0],
                        keep[:,1],
                        keep[:,0]+keep[:,2],
                        keep[:,1]+keep[:,3],
                    ]).astype(np.float64)

                    gt_wh = _to_2d(fboxes[im_i][0])
                    hidx  = _valid_idx(hidxs[im_i][0]-1, gt_wh.shape[0])
                    if hidx.size == 0:
                        continue
                    gwh = gt_wh[hidx]
                    gxyxy = np.column_stack([
                        gwh[:,0],
                        gwh[:,1],
                        gwh[:,0]+gwh[:,2],
                        gwh[:,1]+gwh[:,3],
                    ]).astype(np.float64)

                    ov = bbox_overlaps(kxyxy, gxyxy)
                    max_iou = ov.max(axis=0) if ov.size else np.zeros(gxyxy.shape[0])

                    sizes = np.sqrt(np.clip(gwh[:,2]*gwh[:,3],0,None))
                    for bi,(lo,hi) in enumerate(bins):
                        idx = np.where((sizes > lo) & (sizes <= hi))[0]
                        hits[bi] += np.sum(max_iou[idx] >= 0.5)

            results[thr][mname] = hits

    # —— now plot per-bin recall instead of raw hits ——
    for thr in thresholds:
        plt.figure(figsize=(10,6))
        model_names = list(results[thr].keys())
        n           = len(model_names)
        x           = np.arange(len(bins))
        width       = 0.8 / max(n,1)

        for i,m in enumerate(model_names):
            recall = results[thr][m] / total_gt
            plt.bar(x + i*width, recall, width, label=m)

        plt.xticks(x + width*(n-1)/2, bin_labels)
        plt.ylabel('Recall')
        plt.xlabel('Face size bins in pixels (square root of width * height)')
        plt.ylim(0, 1)
        plt.title(f'Hard subset recall, conf ≥ {thr:.2f}')
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.legend(fontsize='small', ncol=2)
        plt.tight_layout()
        fn = os.path.join(out_dir, f'recall_per_bin_thr{int(thr*100):02d}.png')
        plt.savefig(fn, dpi=300)
        plt.close()
        print(f'✓ saved {fn}')

    print('\n' + '='*60 + '\nSUMMARY STATISTICS\n' + '='*60)
    print('Total Hard-GT faces:', dict(zip(bin_labels, total_gt)))
    for thr in thresholds:
        print(f'\nThreshold {thr:.2f}')
        for m,h in results[thr].items():
            rec_str = ', '.join(
                f'{l}:{r:.3f}'
                for l,r in zip(bin_labels, h/np.maximum(total_gt,1))
            )
            print(f'  {m:<20} {rec_str}')

# ───────────────────── CLI wrapper (unchanged paths) ────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Count per-bin GT hits (Hard subset) at 0.90 & 0.95 conf."
    )
    p.add_argument('-c','--checkpoints',
                   default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/save-checkpoints/deconv_pool',
                   help="root folder containing each model subfolder")
    p.add_argument('-g','--gt',
                   default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/widerface_evaluate/ground_truth',
                   help="path to WIDERFACE ground-truth dir")
    p.add_argument('-o','--out',
                   default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/widerface_evaluate',
                   help="where to write the PNG bar-charts")
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    main(args.checkpoints, args.gt, args.out, debug=args.debug)
