import cv2
import numpy as np
import random
from utils.box_utils import matrix_iof


def _crop(image, boxes, labels, img_dim):
    #boxes is a np array which holds the x1,y1,x2,y2 coordinates of the ground truth face boxes
    #image is one image (which may contain multiple faces), opened witz OpenCV
    height, width, _ = image.shape
    pad_image_flag = True

    # If none of the 250 attempts produces a valid crop, the original image and annotations are returned along with the flag (still True)
    # to indicate that no cropping occurred.
    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        #x1, y1, x2, y2 -> square box
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        #computes how much each ground-truth box overlaps with the ROI. Only if the ROI sufficiently covers a box (overlap value ≥ 1) does it proceed.
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2

        #It verifies that the center of each face box lies inside the ROI. Boxes whose centers are outside are discarded.
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)

        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()

        if boxes_t.shape[0] == 0:
            continue

        #CROP: The image is cropped to the ROI, using array slicing (note that image slicing uses [y1:y2, x1:x2]).
        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        #Clipping: For each selected box, the top-left coordinates are clipped to be no less than the ROI’s top-left.
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim

        # ensures that only boxes with a minimum size greater than 0 (i.e., valid boxes) are kept.
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag

#apply random color distortions to an image
def _distort(image):

    # _convert performs a linear transformation on the pixel values:
    #     Multiplication by alpha: Scales the intensity (affecting contrast).
    #     Addition of beta: Shifts the intensity (affecting brightness).

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta

        #ensures all pixel values are within the valid range [0,255]
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t

# This function randomly flips an image horizontally—and if it does,
# it also adjusts the associated bounding boxes and facial landmarks so that they remain consistent with the flipped image.
def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        #To reverse only the RGB channels for each pixel -> image_reversed = image[:, :, ::-1]
        #horizontal flip -> image = image[:, ::-1] -> You go into the 2nd dimension (the columns) and make a horizontal flip (reverse order)
        image = image[:, ::-1]
        boxes = boxes.copy()
        #The first colon : means "for all rows (all boxes)",
        #The slice 0::2 means "start at index 0 and take every second element."
        #The slice 2::-2 means "start at index 2 and go backwards in steps of 2."
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    #The entire square is filled with the provided rgb_mean value. This acts as the padding color so that the added borders blend with the average color of the dataset.
    image_t[:, :] = rgb_mean
    #The original image is copied into the top-left corner of the square. The rest of the area (if any) remains filled with the mean color.
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    #Choosing an Interpolation Method
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]

    #Resize to size, defined in config.py (or cfg variable)
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    #remove mean. Subtracting the mean centers the pixel values around zero. This normalization step speeds up training and improves convergence.
    image -= rgb_mean
    return image.transpose(2, 0, 1)


class preproc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        # for image, we know: image = cv2.imread(self.imgs_path[index]). So image is the opencv encoded picture in original format
        # for targets, we know: targets has dimension (#amountOfFacesOnPicture, 15)
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        #labels: Are either 1 or -1, as preprocessed by __getitem__(self, index) in WiderFaceDetection(data.Dataset)
        labels = targets[:, -1].copy()

        image_t, boxes_t, labels_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)

        image_t = _distort(image_t)
        #If image_t is not square yet: This function pads an image with a specified mean color to form a square image when needed, preserving the original content in the top-left corner.
        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)

        image_t, boxes_t = _mirror(image_t, boxes_t)

        height, width, _ = image_t.shape
        # In _resize_subtract_mean, the image gets its 640 x 640 format.
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        # Convert the absolute pixel coordinates of bounding boxes and landmarks into relative coordinates (i.e., values between 0 and 1).
        # Normalized coordinates allow the model to work independently of the absolute image size
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        #The normalized boxes and landmarks are combined with labels to form the final target tensor.
        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t
