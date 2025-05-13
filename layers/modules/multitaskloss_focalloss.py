##### CARLOS CODE FILE ########

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_utils import match_focal_loss, log_sum_exp
from data import cfg_re50
from torchvision.ops import sigmoid_focal_loss

class MultiTaskLossWithFocalLoss(nn.Module):
    """Weighted Loss Function (inspired from SSD and Focal Loss)
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Apply Focal Loss to all anchors (also all negative ones)
    Objective Loss:
        L(x,c,l,g) = (Lfocal(x, c) + αLloc(x,l,g)) / N
        Where, Lfocal is the Focal Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf and https://arxiv.org/pdf/1708.02002 for more details.
    """

    def __init__(self, num_classes, iou_threshold_background, variance, focal_gamma=2.0, focal_alpha=0.25, gpu_train=True):
        super(MultiTaskLossWithFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold_background = iou_threshold_background
        self.variance = variance
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.gpu_train = gpu_train

    def forward(self, predictions, priors, targets):
        """
        Arguments:
            predictions (tuple): A tuple containing:
                loc_data: predicted boxes, shape (batch_size, num_priors, 4)
                conf_data: predicted class scores, shape (batch_size, num_priors, num_classes)
            priors (tensor): Prior boxes, shape (num_priors, 4)
            targets (list of tensors): Ground truth boxes and labels for each image,
                each of shape (num_objs, 5) (last element is the label)
        Returns:
            loss_l: Localization loss (Smooth L1)
            loss_c: Classification loss (Focal Loss)
        """
        #conf_data is (batch_size, anchors, 2) -> so the innerst element has size 2 -> but its not that these two elements sum to 1
        loc_data, conf_data = predictions
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        # torch.Tensor() -> This creates a tensor with uninitialized memory. The values in the tensor are essentially garbage/random values that were in memory at that location
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            defaults = priors.data

            # match’s role is to assign each of the many predefined anchor (prior) boxes to a ground truth box (or mark it as background) based on the
            # intersection over union (IoU) overlap.
            # It then encodes the targets (box offsets and landmark offsets) that the network will learn to predict.
            # Everything is saved in the loc_t, conf_t and landm_t tensors
            match_focal_loss(self.threshold_background, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.gpu_train:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # ----- Localization Loss -----
        # Positive mask: anchors with label > 0 are considered positives
        zeros = torch.tensor(0).cuda()
        # pos holds the indices of the positives
        pos = conf_t > zeros

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # The predicted box offsets (loc_data) and the target offsets (loc_t) for the POSITIVE anchors are gathered and reshaped into matrices with 4 columns.
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # ----- Classification Loss using Focal Loss -----

        conf_data = conf_data.view(-1, self.num_classes)
        conf_t = conf_t.view(-1)
        targets_one_hot = (conf_t.unsqueeze(1) == torch.arange(self.num_classes, device=conf_t.device).unsqueeze(0)).float()

        # Compute focal loss using logits directly.
        loss_c = sigmoid_focal_loss(
            inputs=conf_data,
            targets=targets_one_hot,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum"
        )

        # Normalize losses by the number of positive anchors
        N = max(pos.data.long().sum().float(), 1)

        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
