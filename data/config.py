# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50-11k',
    #This means that for each feature map (corresponding to a particular scale), two square anchors are generated—one with a smaller size and one with a larger size.
    'min_sizes': [
        [16, int(16*2**(1/3)), int(16*2**(2/3))],   # For P2
        [32, int(32*2**(1/3)), int(32*2**(2/3))],   # For P3
        [64, int(64*2**(1/3)), int(64*2**(2/3))],   # For P4
        [128, int(128*2**(1/3)), int(128*2**(2/3))], # For P5
        [256, int(256*2**(1/3)), int(256*2**(2/3))]  # For P6
    ],
    #TODO: Check if we need this "steps" attribute, since we already regulate the feature maps attribute and I guess one can infer the step size from the feature maps.
    'steps': [4, 8, 16, 32, 64],
    # The variance list of parameters is in fact an abuse of terms. Instead it indicates and contains precomputed standard deviations of the landmark and bounding box targets. What the code is doing is some sort of ad-hoc normalization of the targets.
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 2,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 640,
    'pretrain': True,
    #For P6 to work, the 'return_layers' attribute must contain '3' in the list.
    'introduce_P6': True,
    #layer1 == C2, layer2 == C3, layer3 == C4, layer4 == C5
    #'return_layers': {'layer1' : 1, 'layer2': 2, 'layer3': 3, 'layer4': 4},
    #0 == C2, 1== C3, 2 == C4, 3 == C5
    'return_layers': [0, 1, 2, 3],
    'in_channel': 256,
    'out_channel': 256,
    'anchor_num': 3
}

