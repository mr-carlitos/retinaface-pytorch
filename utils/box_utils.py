import torch
import numpy as np

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

#This function computes the “intersection over foreground”
#In data augmentation, you might want to ensure that a random crop (ROI) fully covers a ground-truth box. If the IOF is 1, it means the entire box in a lies within the box in b.
def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(threshold_background, threshold_foreground, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold_background: (float) The overlap threshold used when matching boxes to background
        threshold_foreground: (float) The overlap threshold used when matching boxes to foreground
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """

    #1. Compute IoU Overlaps
    # jaccard index
    # we basically compare EACH of the truths with EACH of the 102'300 priors. First, we convert the priors from cx, cy, s_kx, s_ky to (xmin, ymin, xmax, ymax)
    # overlaps has Shape: [box_a.size(0), box_b.size(0)] which means [AmountOfTruthBoxes, 102'300]
    overlaps = jaccard(
        truths,
        point_form(priors)
    )

    # 2. For Each Ground Truth, Find Its Best-Matching Prior
    #best_prior_overlap holds the highest IoU for each ground-truth box, best_prior_idx contains the index of the prior that gives that maximum overlap
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    ##### CARLOS CODE STARTS HERE #######################
    # Check for duplicates in best_prior_idx
    unique_vals, counts = best_prior_idx.unique(return_counts=True)
    duplicate_vals = unique_vals[counts > 1]
    if duplicate_vals.numel() > 0:
        print(f"WARNING: Duplicate indices found in best_prior_idx: {duplicate_vals.tolist()}")
    ##### CARLOS CODE ENDS HERE #######################


    # 3. Filter Out Ground Truth Boxes with Insufficient Overlap
    # ignore hard gt
    # The function only keeps ground-truth boxes that have at least one prior with an overlap of 0.2 or higher.
    # I guess this 0.2 indexing is some kind of pre-filtering
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        # sets every element in that access matrix (loc_t[idx]) to 0.
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # 4. For Each Prior, Find Its Best-Matching Ground Truth
    # [1, num_priors] best ground truth for each prior
    # best_truth_overlap (shape [n_priors]) contains the highest IoU that each prior has with any ground-truth box.
    # best_truth_idx (shape [n_priors]) contains the index of the ground-truth box that best matches each prior.
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # 5. Force Each Ground Truth Box to Match With Its Best Prior
    # Docs: Tensor.index_fill_(dim, index, value)
    # Fills the elements of the self tensor with value by selecting the indices in the order given in index.
    # index_fill_ is used to force the best matching prior (from the bipartite matching step) to have a very high overlap
    #   (set to 2, which is above any possible IoU value) so that it will definitely be selected.
    # Purpose: This guarantees that each ground truth box is “assigned” at least to one prior regardless of the standard threshold.
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)
    # TODO refactor: index  best_prior_idx with long tensor
    # The loop forces the assignment so that the best prior for each ground truth is linked with that specific ground truth
    # Force Each Ground-Truth Box to be Matched with Its Best Prior
    # Purpose: This step ensures that every ground truth is represented in the target assignment
    # The initial overlaps.max(0, keepdim=True) finds the highest-overlap ground truth for each prior, but it doesn’t guarantee that EVERY ground truth is assigned to its best prior.
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # 6. Generate Final Target Tensors
    # a. Localization Targets
    matches = truths[best_truth_idx]            # Shape: [num_priors,4]
    # Encode Localization: convert the ground-truth box coordinates (in matches) and the corresponding priors into regression targets.
    # this means computing offsets from the prior’s center, plus log-scale differences for width and height.
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn

    # b. Confidence Targets
    conf = labels[best_truth_idx]              # Shape: [num_priors]
    # Be aware! This statement does not mean that all entries in conf are >= 0. Only those who have a very low IoU will be 0. But there are still those which have a high IoU and are marked as -1.
    conf[best_truth_overlap < threshold_background] = 0    # label as background

    #ignore_mask = (best_truth_overlap >= threshold_background) & (best_truth_overlap <= threshold_foreground)
    #true_count = ignore_mask.sum()
    #print(true_count)
    #conf[ignore_mask] = -2

    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    """The purpose of the encode() function is to convert the ground-truth bounding box coordinates (matched to each prior/anchor) into a normalized offset representation 
    that the network will predict. 
    In other words, it transforms the absolute coordinates into a relative “delta” with respect to the anchor’s center and size.
    
    The encode() function takes the ground-truth box and the corresponding prior (anchor) and computes:
        - The normalized offset of the centers,
        - The log-scale difference of the sizes,
    and concatenates these into a single tensor of shape [num_priors, 4] that serves as the regression target.
    """

    # 1. Compute the Ground-Truth Box Center Offset
    # g_cxcy = ground_truth_center − prior_center. Results in raw offset (in pixels) between the ground truth and the prior.
    # Remember that priors remained / are in (cx, cy, s_kx, s_ky) format.

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]

    # 2. Normalize the Center Offset: This normalization step scales the offsets by the size of the anchor
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])

    # 3. Compute the Ground-Truth Box Size Ratio
    # For each matched box, compute its width and height: gt_wh = (x2 - x1, y2 - y1)
    # Then, divide these dimensions by the corresponding width and height of the prior: gt_wh / prior_wh
    # This gives a ratio that indicates how much larger or smaller the ground-truth box is compared to the anchor.

    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]

    # 4. Convert the Size Ratio into Log-Space
    g_wh = torch.log(g_wh) / variances[1]

    # Each row of the output now represents the regression target for a prior in the NORMALIZED (relative to the prior / anchor box) form: [delta-x, delta-y, delta-width, delta-height]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    # boxes[:, :2] ist esentially == (cx, cy) + (Delta_cx, Delta_cy) * (w, h) -> wir haben es also mit einem NORMALISIERTEN Delta zu tun, wir können also nicht einfach e.g., für x = cx + Delta_cx machen

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2 # Shift from center to top-left
    boxes[:, 2:] += boxes[:, :2] # Compute bottom-right from top-left and dimensions
    return boxes

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max