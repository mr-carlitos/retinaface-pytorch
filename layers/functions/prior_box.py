import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
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
        return output

#CARLOS CODE: Implemented a vectorized version.
    def vectorized_forward(self):
        anchors_all = []
        # Loop over each feature map scale.
        for k, f in enumerate(self.feature_maps):
            step = self.steps[k]
            fea_h, fea_w = f  # height and width of the feature map
            # Create grid shifts along x and y (normalized to image width/height).
            shift_x = (np.arange(fea_w) + 0.5) * step / self.image_size[1]  # shape (fea_w,)
            shift_y = (np.arange(fea_h) + 0.5) * step / self.image_size[0]  # shape (fea_h,)
            # Get all combinations via meshgrid.
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            # Flatten the grid.
            shift_x = shift_x.reshape(-1)  # shape (N,)
            shift_y = shift_y.reshape(-1)  # shape (N,)

            # Convert min_sizes[k] into a NumPy array.
            min_sizes = np.array(self.min_sizes[k], dtype=np.float32)  # shape (M,)
            # Compute normalized widths and heights.
            s_kx = min_sizes / self.image_size[1]  # shape (M,)
            s_ky = min_sizes / self.image_size[0]  # shape (M,)

            # For each grid location, assign all the min_sizes.
            N = shift_x.shape[0]  # number of grid positions
            M = min_sizes.shape[0]  # number of sizes per location

            # Repeat the center coordinates for each min_size.
            cx = np.repeat(shift_x, M)  # shape (N*M,)
            cy = np.repeat(shift_y, M)  # shape (N*M,)
            # Tile the sizes (each grid location gets M sizes)
            s_kx = np.tile(s_kx, N)  # shape (N*M,)
            s_ky = np.tile(s_ky, N)  # shape (N*M,)

            # Stack into anchors of shape (N*M, 4): [cx, cy, s_kx, s_ky]
            anchors = np.stack([cx, cy, s_kx, s_ky], axis=1)
            anchors_all.append(anchors)

        # Concatenate anchors from all feature map levels.
        output = np.concatenate(anchors_all, axis=0)

        return torch.from_numpy(output)
