# CARLOS CODE FILE: Modified the original config
# config.py


'''
This is the main config file to setup a new model variant. Hence, this is a very important piece of code. A short overview of the settings:
- name: Feel free to choose the name you want
- backbone_name: Can be either Resnet50-11k' or 'Resnet50-1k', depending if you want the ImageNet-1k or ImageNet-11k backbone
- 'featuremaps_at_end_of_stage': Remember that we get 5 feature maps C1...C5 from the backbone. This comes from the fact that the backbone is devided in 5 stages.
    This setting decies if we use the feature maps as the end of each stage, or if we use intermediate feature maps, as in the original RetinaFace implementation
- 'min_sizes': Defines the anchor sizes for each anchor position and feature map level i
- 'steps': Defines the step size / stride between anchors, and this depends of course on which feature map level we work.
- 'variance': The variance list of parameters is in fact an abuse of terms. Instead it indicates and contains precomputed standard deviations of the landmark and bounding box targets. What the code is doing is some sort of ad-hoc normalization of the targets.
- 'neck_mode': Since this master thesis is about providing alternative neck architectures, we have defined inside config_transform.py a series of model variants which we reference here. Set here the model variant that you want to use
- 'shared_losshead': Defines if we use shared loss heads over P2...P6 or if each feature map level gets its own loss heads (one head for classification and one for bounding box regression)
- [CROSS-ATTENTION] 'query_focused_residualconn': If true, we apply the query-focused residual connection
- [CROSS-ATTENTION] 'pos_embedding': Different modes of how to define the positional embeddings. PositionalMode.NOTHING means no positional embeddings, PositionalMode.POS_EMBEDDING_QKV means we use learned positional embeddings and PositionalMode.POS_ENCODING_QKV means we apply fixed sinusoidal positional encoding (similar to Vaswani paper, but we have 2D instead of 1D encoding)
- [CROSS-ATTENTION] 'attention_heads': Number of attention heads
- [CROSS-ATTENTION] 'upperandlower': True = UPPERANDLOWER -> we use i+1, i and i-1 , False = ONLYUPPER -> we use i+1, i but not i-1
- [CROSS-ATTENTION] 'pyramidial': True = Pyramidial (meaning we go top-down and reuse P_i+1 when we compute P_i), False= Horizontal (meaning we go top-down and don't use P_i+1, but work with C_i+1 (from backbone) when we compute P_i)
- [CROSS-ATTENTION] 'increase_receptive_field': Defines the patch sizes. If false, we have 1x1 -> 2x2 -> 4x4 as patch sizes for i+1, i, i-1. If true, we have 2x2 -> 4x4 -> 8x8 as patch sizes (true was used for the final results in my thesis)
- 'introduce_P6': If we introduce P6 or not
- 'return_layers': Tells the neck architecture which of the 4 backbone feature maps (0 == C2, 1== C3, 2 == C4, 3 == C5) should be used to calculate the final outputs P_i. Per default, we want to use all of them, hence 'return_layers': [0, 1, 2, 3]
- 'in_channel': Input channel parameter, needed for some detail in the FPN implementation (not very important)
- 'out_channel': Channel dimensionality parameter, will be used for all feature map levels. Per default, we say that the FPN outputs a channel dimensionality of 256 (context module then works on feature maps that all have channel dimensionality set to 256)
- 'anchor_num': How many anchors we use per anchor location. Per default, it's 3 anchors and the sizes are defined in 'min_sizes'
- 'iou_threshold_background' : IoU threshold value we use during training
- 'focal_gamma': Focal loss gamma
- 'focal_alpha': Focal loss alpha

Important reference code files:
1. biubug6's config: https://github.com/biubug6/Pytorch_Retinaface/blob/master/data/config.py
'''

from data.config_transform import NeckMode, PositionalMode

cfg_re50 = {
    'name': 'FINAL_LARGEGROUPS_24July',
    'backbone_name': 'Resnet50-11k',
    'featuremaps_at_end_of_stage': False,
    'min_sizes': [ #This means that for each feature map (corresponding to a particular scale), three square anchors are generated—one with a smaller size, medium size and one with a larger size.
        [16, int(16*2**(1/3)), int(16*2**(2/3))],   # For P2
        [32, int(32*2**(1/3)), int(32*2**(2/3))],   # For P3
        [64, int(64*2**(1/3)), int(64*2**(2/3))],   # For P4
        [128, int(128*2**(1/3)), int(128*2**(2/3))], # For P5
        [256, int(256*2**(1/3)), int(256*2**(2/3))]  # For P6
    ],
    'steps': [4, 8, 16, 32, 64], # P2...P6
    'variance': [0.1, 0.2],
    'neck_mode': NeckMode.CROSSATTENTION,

    'shared_losshead': False,

    # -----START: FOR CROSS ATTENTION------
    'query_focused_residualconn': True,
    'pos_embedding': PositionalMode.NOTHING,
    'attention_heads': 4,
    'upperandlower': True,
    'pyramidial': True,
    'increase_receptive_field': True,
    # ------ END: FOR CROSS ATTENTION ---------

    'introduce_P6': True,
    'return_layers': [0, 1, 2, 3], #For P6 to work, the 'return_layers' attribute must contain '3' in the list.
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