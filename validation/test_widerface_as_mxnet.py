##### CARLOS CODE FILE ########
## Based on mxnet implementation of RetinaFace
# retinaface.py: https://github.com/deepinsight/insightface/blob/master/detection/retinaface/retinaface.py#L573
# test_widerface.py: https://github.com/deepinsight/insightface/blob/master/detection/retinaface/test_widerface.py
from __future__ import print_function
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.box_voting import bbox_vote
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, clip_boxes
from utils.timer import Timer


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/save-checkpoints/2025-06-02_CROSSATTENTION_ONLYUPPER_PYRAMIDIAL_POSITIONALENC/CROSSATTENTION_FROMUPPER_PYRAMIDIAL_final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/save-checkpoints/2025-06-02_CROSSATTENTION_ONLYUPPER_PYRAMIDIAL_POSITIONALENC/like-mxnet/', type=str, help='Dir to save txt results')
parser.add_argument('--origin_size', default=False, type=bool, help='Whether use origin image size to evaluate')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--images_folder', default='/local/scratch/datasets/WiderFace/WIDER_val/images/', type=str, help='image dataset path')
parser.add_argument('--input_val_txt', default='/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/data/widerface/val/wider_val.txt', type=str, help='val txt path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.35, type=float, help='visualization_threshold')
parser.add_argument('--flipping', default=True, type=bool, help='if we do flipping during evaluation pipeline')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = False

    torch.cuda.set_device(4)
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # Multi-scale and flip settings.
    # If not evaluating at original size, use multi-scale.
    # These numbers represent target values for the shorter side of the image
    TEST_SCALES = [500, 800, 1100, 1400, 1700] if not args.origin_size else None
    do_flip = args.flipping  # Enable horizontal flip.
    max_size = 2150  # Maximum allowed size for the longer side.

    # Testing dataset.
    testset_folder = args.images_folder
    testset_list = args.input_val_txt
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'all': Timer()}
    start_idx = 0
    # Loop over images.
    for i, img_name in enumerate(test_dataset):
        if i < start_idx:
            continue
        print("Processing image {}: {}".format(i, img_name))
        image_path = os.path.join(testset_folder, img_name)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_raw is None:
            print("Failed to load image:", image_path)
            continue
        original_width = img_raw.shape[1]

        im_shape = img_raw.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        aggregated_dets = []  # Accumulate detections from different scales/flips.
        _t['all'].tic()

        if TEST_SCALES is not None:
            scales_to_test = TEST_SCALES
        else:
            scales_to_test = [None]  # Only one scale (original size).

        # Loop over test scales.
        for scale_val in scales_to_test:
            if scale_val is None:
                resize = 1.0
            else:
                # Use pyramid branch parameters from MXNet
                target_size_pyr = 800
                max_size_pyr = 1200
                im_scale0 = float(target_size_pyr) / float(im_size_min)
                if np.round(im_scale0 * im_size_max) > max_size_pyr:
                    im_scale0 = float(max_size_pyr) / float(im_size_max)
                # Scale factor is computed relative to the pyramid target size (800)
                resize = float(scale_val) / float(target_size_pyr) * im_scale0

            # Loop over flip states.
            for flip in [False, True] if do_flip else [False]:
                # Prepare image: flip if required.
                if flip:
                    img_proc = cv2.flip(img_raw, 1)
                else:
                    img_proc = img_raw.copy()
                # Resize.
                if resize != 1.0:
                    img_resized = cv2.resize(img_proc, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                else:
                    img_resized = img_proc.copy()

                im_height, im_width, _ = img_resized.shape
                # Scale tensor for converting network outputs back.
                scale_tensor = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)

                # Preprocessing: subtract mean (BGR) and transpose.
                img_input = np.float32(img_resized)
                img_input -= (104, 117, 123)
                img_input = img_input.transpose(2, 0, 1)  # Convert to C x H x W.
                img_input = torch.from_numpy(img_input).unsqueeze(0)
                img_input = img_input.to(device)

                # Forward pass.
                with torch.no_grad():
                    loc, conf = net(img_input)
                # Generate prior boxes and decode detections.
                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                with torch.no_grad():
                    priors = priorbox.vectorized_forward().float()
                    priors = priors.to(device)
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                # Convert boxes to original image coordinates.
                boxes = boxes * scale_tensor / resize
                boxes = boxes.cpu().numpy()

                boxes = clip_boxes(boxes, img_raw.shape[:2])

                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

                # Filter low confidence detections.
                inds = np.where(scores > args.confidence_threshold)[0]
                if len(inds) == 0:
                    continue
                boxes = boxes[inds]
                scores_selected = scores[inds]

                # Unflip boxes if the image was flipped.
                if flip:
                    x1 = boxes[:, 0].copy()
                    x2 = boxes[:, 2].copy()
                    boxes[:, 0] = original_width - x2 - 1
                    boxes[:, 2] = original_width - x1 - 1

                # Stack boxes and scores.
                dets = np.hstack((boxes, scores_selected[:, np.newaxis])).astype(np.float32, copy=False)
                aggregated_dets.append(dets)

        if len(aggregated_dets) == 0:
            print("No detections for image {}".format(img_name))
            continue

        # Combine detections from all scales and flip passes.
        all_dets = np.vstack(aggregated_dets)

        # Apply box voting.
        final_dets = bbox_vote(all_dets, args.nms_threshold, args.keep_top_k)

        _t['all'].toc()

        # Save detection results to a text file.
        save_name = os.path.join(args.save_folder, img_name[:-4] + ".txt")
        save_dir = os.path.dirname(save_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(save_name, "w") as fd:
            file_basename = os.path.basename(save_name)[:-4] + "\n"
            fd.write(file_basename)
            fd.write(str(final_dets.shape[0]) + "\n")
            for box in final_dets:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = "{} {} {} {} {} \n".format(x, y, w, h, confidence)
                fd.write(line)

        print("im_detect: {}/{} total_time: {:.4f}s".format(i + 1, num_images, _t['all'].average_time))

        if args.save_image:
            for b in final_dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)