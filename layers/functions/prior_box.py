import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        # feature_maps is created by iterating through the list steps which has INCREASING values
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    """
    The forward() method generates a set of prior (or anchor) boxes in a normalized coordinate space (values between 0 and 1). Each anchor is represented in the format [cx,cy,w,h] where:
    
        cx,cy: Normalized center coordinates of the box.
        s_kx, s_ky: Normalized width and height of the box.
    
    These anchors are generated for each feature map cell at multiple scales (defined by min_sizes) across different feature map resolutions (derived from steps).
    
    """

    def forward(self):
        anchors = []
        #k-th feature map, where f is the feature map
        # feature_maps is created by iterating through the list steps which has INCREASING values
        for index, feature_map in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[index]
            # i,j are feature map indices
            for row_index, column_index in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    cx = (column_index + 0.5) * self.steps[index] / self.image_size[1]
                    cy = (row_index + 0.5) * self.steps[index] / self.image_size[0]
                    anchors += [cx, cy, s_kx, s_ky]

        # back to torch land

        # So far, anchors is just a very long 1D list. But we want a matrix of shape (N, 4).
        # The -1 tells PyTorch to automatically calculate the number of rows so that the total number of elements remains the same.
        output = torch.Tensor(anchors).view(-1, 4)
        # If self.clip is set to True, the anchor values are clamped to be within [0,1]: This ensures that no coordinate goes outside the normalized image range.
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
