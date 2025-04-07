# config.py

cfg_re50 = {
    'name': 'RetinaFace_baseline_withFPN',
    'backbone_name': 'Resnet50-11k',
    'featuremaps_at_end_of_stage': True,
    #This means that for each feature map (corresponding to a particular scale), two square anchors are generated—one with a smaller size and one with a larger size.
    'min_sizes': [
        [16, int(16*2**(1/3)), int(16*2**(2/3))],   # For P2
        [32, int(32*2**(1/3)), int(32*2**(2/3))],   # For P3
        [64, int(64*2**(1/3)), int(64*2**(2/3))],   # For P4
        [128, int(128*2**(1/3)), int(128*2**(2/3))], # For P5
        [256, int(256*2**(1/3)), int(256*2**(2/3))]  # For P6
    ],
    #TODO: Optional, Check if we need this "steps" attribute, since we already regulate the feature maps attribute and I guess one can infer the step size from the feature maps.
    'steps': [4, 8, 16, 32, 64],
    # The variance list of parameters is in fact an abuse of terms. Instead it indicates and contains precomputed standard deviations of the landmark and bounding box targets. What the code is doing is some sort of ad-hoc normalization of the targets.
    'variance': [0.1, 0.2],
    'clip': False,
    'gpu_train': True,
    'batch_size': 8,
    #For gradient accumulation
    'accumulation_steps' : 2,
    'ngpu': 1,
    'apply_FPN': True,
    'epoch': 80,
    'decay1': 55,
    'decay2': 68,
    'image_size': 640,
    'pretrain': True,
    #For P6 to work, the 'return_layers' attribute must contain '3' in the list.
    'introduce_P6': True,
    #0 == C2, 1== C3, 2 == C4, 3 == C5
    'return_layers': [0, 1, 2, 3],
    'in_channel': 256,
    'out_channel': 256,
    'anchor_num': 3,
    'iou_threshold_background' : 0.3,
    'iou_threshold_foreground': 0.5,
    'neg_pos_ratio': 3,
    'focal_gamma' : 2.0,
    'focal_alpha': 0.25
}