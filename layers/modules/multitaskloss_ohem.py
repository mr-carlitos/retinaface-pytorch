import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import match_ohem, log_sum_exp
class MultiTaskLossWithOHEM(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.

    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, iou_threshold_background, iou_threshold_foreground, neg_pos_ratio, variance, gpu_train):
        super(MultiTaskLossWithOHEM, self).__init__()
        self.num_classes = num_classes
        self.threshold_background = iou_threshold_background
        self.threshold_foreground = iou_threshold_foreground
        self.negpos_ratio = neg_pos_ratio
        self.variance = variance
        self.gpu_train = gpu_train

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        #conf_data is (batch_size, anchors, 2) -> so the innerst element has size 2 -> but its not that these two elements sum to 1
        loc_data, conf_data = predictions
        priors = priors
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
            match_ohem(self.threshold_background, self.threshold_foreground, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.gpu_train:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        zeros = torch.tensor(0).cuda()

        ignored_tensor = torch.tensor(-1).cuda()
        ignored = conf_t == ignored_tensor
        #variable for debugging, not used by code
        ignored_sum = ignored.long().sum(1, keepdim = True)

        # pos holds the indices of the positives
        pos = conf_t > zeros
        # variable for debugging, not used by code
        pos_sum = pos.long().sum(1, keepdim=True)

        #needed for the gather() function later. We will temporarily set the conf values of the ignored entries to 1.
        pos_and_ignored = conf_t != zeros
        conf_t[pos_and_ignored] = 1

        background = conf_t == zeros
        # variable for debugging, not used by code
        background_sum = background.long().sum(1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # The predicted box offsets (loc_data) and the target offsets (loc_t) for the POSITIVE anchors are gathered and reshaped into matrices with 4 columns.
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        # self.num_classes = 2, conf_data is model output
        batch_conf = conf_data.view(-1, self.num_classes)
        #The log_sum_exp function computes the logarithm of the sum of exponentials of a tensor’s elements in a numerically stable way.
        # batch_conf.gather(1, conf_t.view(-1, 1)) selects, for each anchor (each row in batch_conf), the predicted confidence value corresponding to its target class as given in conf_t.
        # loss_c computes, for each anchor, the negative log-likelihood of the target class. This is done by subtracting the logit for the true class from the log-sum-exp of all logits.
        # get a per-anchor “loss” (which is then used to rank negatives for hard negative mining)
        # But it's actually is mathematically equivalent to what F.cross_entropy() (This is equivalent to computing the negative log-likelihood for the true class when the softmax function is applied.)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        # Hard negative mining selects the negatives that the network is getting most wrong—that is, the negatives with very high loss values.
        pos_view = pos.view(-1, 1)
        ignored_view = ignored.view(-1, 1)
        loss_c[pos_view] = 0 # filter out pos boxes for now
        loss_c[ignored_view] = 0 # filter out ignored boxes
        loss_c = loss_c.view(num, -1)
        test_loss_values_negatives_sorted, loss_idx = loss_c.sort(1, descending=True)
        # This second sort doesn't sort the losses again; instead, it tells you the rank (position) of each anchor in the descending order.
        test_idx_rank_values, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # .gt(0) (greater than zero) -> creates a tensor that is nonzero (1 or 2) wherever an anchor is either positive or selected as a negative.
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        # loss_c is ultimately computing the softmax cross-entropy loss for a binary classification problem.
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
