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
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
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
    'introduce_P6': True,
    #layer1 == C2, layer2 == C3, layer3 == C4, layer4 == C5
    'return_layers': {'layer1' : 1, 'layer2': 2, 'layer3': 3, 'layer4': 4},
    'in_channel': 256,
    'out_channel': 256
}

