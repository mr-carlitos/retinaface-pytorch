##### CARLOS CODE FILE ########
from layers.functions.prior_box import PriorBox
import torch
import cv2
import random
from data import cfg_re50

# Choose which feature map to work with:
# 0: P2, 1: P3, 2: P4, 3: P5, 4: P6 (default is P6)
chosen_feature_map_index = 4

# Set the drawing mode: choose from 'border', 'center', or 'both'
draw_mode = 'both'

# --- Utility: Decode anchors from center-size to (xmin, ymin, xmax, ymax) ---
def point_form(boxes):
    """
    Converts anchors from [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


# --- Script: Draw one anchor per feature map ---
def draw_anchors_on_image(image_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image from", image_path)
        return
    img_h, img_w = image.shape[:2]

    # Use a random 640x640 crop if possible; otherwise, resize.
    target_size = cfg_re50['image_size']
    if img_w >= target_size and img_h >= target_size:
        max_x = img_w - target_size
        max_y = img_h - target_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        image = image[y:y + target_size, x:x + target_size]
        img_h, img_w = image.shape[:2]
    else:
        # If the image is smaller than 640x640, fallback to resizing.
        image = cv2.resize(image, (target_size, target_size))
        img_h, img_w = image.shape[:2]

    # Create PriorBox and generate anchors (in normalized coordinates)
    priorbox = PriorBox(cfg_re50, image_size=(target_size, target_size))
    anchors = priorbox.vectorized_forward()  # shape: [num_anchors, 4]

    # Convert anchors to corner coordinates (still normalized)
    anchors_corners = point_form(anchors)
    # Scale normalized anchors to image pixel coordinates
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    anchors_corners = anchors_corners * scale
    anchors_corners = anchors_corners.numpy()

    # Define colors (BGR) for each feature map:
    # P2 (index 0): blue, P3: green, P4: yellow, P5: orange, P6: red
    colors = {
        0: (255, 0, 0),      # blue for P2
        1: (0, 255, 0),      # green for P3
        2: (0, 255, 255),    # yellow for P4
        3: (0, 165, 255),    # orange for P5
        4: (0, 0, 255)       # red for P6
    }

    # Get the chosen feature map dimensions and compute number of anchors for that map
    p_chosen_fm = priorbox.feature_maps[chosen_feature_map_index]
    num_anchors_chosen = p_chosen_fm[0] * p_chosen_fm[1] * len(cfg_re50['min_sizes'][chosen_feature_map_index])

    # Compute start index by summing all anchors from previous feature maps
    start_index = sum([fm[0] * fm[1] * len(cfg_re50['min_sizes'][i])
                       for i, fm in enumerate(priorbox.feature_maps[:chosen_feature_map_index])])
    end_index = start_index + num_anchors_chosen

    # Define a list of 10 colors (BGR format)
    colors_list = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (0, 255, 255),    # Yellow
        (0, 165, 255),    # Orange
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (128, 0, 128),    # Purple
        (0, 128, 255),    # Light Blue
        (128, 128, 0)     # Olive
    ]

    for i, anchor_idx in enumerate(range(start_index, end_index)):
        if random.random() < 1.1:  # Draw each anchor with 40% probability
            box = anchors_corners[anchor_idx]  # [xmin, ymin, xmax, ymax]
            box = box.astype(int)
            color = colors_list[i % len(colors_list)]
            if draw_mode in ['border', 'both']:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 1)
            if draw_mode in ['center', 'both']:
                # Compute the center coordinate of the anchor
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2
                cv2.circle(image, (cx, cy), 2, color, -1)  # Draw a filled circle (radius 2)

    # Save the result
    cv2.imwrite(output_path, image)
    print("Output saved to:", output_path)

if __name__ == '__main__':

    input_image_path = "./curve/test.jpg"         # Input image file (should exist)
    output_image_path = "output_anchors.jpg"  # Output file with drawn anchors
    draw_anchors_on_image(input_image_path, output_image_path)