# CARLOS CODE FILE: Modified the original config
# config.py
from data.config_transform import NeckMode, PositionalMode

cfg_re50 = {
    'name': 'CROSSATTENTION_FROMUPPERANDLOWER_PYRAMIDIAL',
    'backbone_name': 'Resnet50-11k',
    'featuremaps_at_end_of_stage': False,
    #This means that for each feature map (corresponding to a particular scale), three square anchors are generated—one with a smaller size and one with a larger size.
    'min_sizes': [
        [16, int(16*2**(1/3)), int(16*2**(2/3))],   # For P2
        [32, int(32*2**(1/3)), int(32*2**(2/3))],   # For P3
        [64, int(64*2**(1/3)), int(64*2**(2/3))],   # For P4
        [128, int(128*2**(1/3)), int(128*2**(2/3))], # For P5
        [256, int(256*2**(1/3)), int(256*2**(2/3))]  # For P6
    ],
    'steps': [4, 8, 16, 32, 64],
    # The variance list of parameters is in fact an abuse of terms. Instead it indicates and contains precomputed standard deviations of the landmark and bounding box targets. What the code is doing is some sort of ad-hoc normalization of the targets.
    'variance': [0.1, 0.2],
    'neck_mode': NeckMode.CROSSATTENTION,
    'shared_ssh': False,

    #-----FOR CROSS ATTENTION------
    'residualconn' : False,
    'groupnorm' : False,
    'query_focused_residualconn': False,
    'pos_embedding': PositionalMode.POS_ENCODING,
    'apply_convblock': True,
    'attention_heads': 4,
    'upperandlower': True,  # True = UPPERANDLOWER, False = ONLYUPPER
    'pyramidial': True,     #True = Pyramidial, False= Horizontal
    #------ END ---------

    #For P6 to work, the 'return_layers' attribute must contain '3' in the list.
    'introduce_P6': True,
    #0 == C2, 1== C3, 2 == C4, 3 == C5
    'return_layers': [0, 1, 2, 3],
    'in_channel': 256,
    'out_channel': 256,
    # Make sure this aligns with the variable min_sizes (above) -> amount of sizes per point
    'anchor_num': 3,
    'iou_threshold_background' : 0.35,
    #'iou_threshold_foreground': 0.5, -> we would use this for OHEM. But since we have implemented Focal Loss, we won't use OHEM & this attribute
    # 'neg_pos_ratio': 3, -> we would use this for OHEM. But since we have implemented Focal Loss, we won't use OHEM & this attribute
    'focal_gamma' : 2.0,
    'focal_alpha': 0.25
}