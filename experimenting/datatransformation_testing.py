##### CARLOS CODE FILE ########
import torch

# Let's assume we have three anchors and two classes.
# conf_data represents the raw logits (predictions) from your network.
# Shape: (3, 2)
conf_data = torch.tensor([
    [0.2, -0.1],   # Anchor 1: predicted logits for [background, face]
    [1.5, 0.3],    # Anchor 2: predicted logits for [background, face]
    [-0.5, 2.0]    # Anchor 3: predicted logits for [background, face]
])

# conf_t is the ground truth class indices for each anchor.
# Shape: (3,)
# For example, let's say:
# Anchor 1 is background (class 0), Anchor 2 is face (class 1), Anchor 3 is face (class 1)
conf_t = torch.tensor([0, 1, 1])

# To use torchvision's sigmoid_focal_loss, targets must be in one-hot format.
# We convert conf_t into one-hot vectors.
num_classes = 2
conf_t_unsqueezed = conf_t.unsqueeze(1)
num_classes_arranged = torch.arange(num_classes)
arranged_unsqueezed = num_classes_arranged.unsqueeze(0)
targets_one_hot = (conf_t.unsqueeze(1) == torch.arange(num_classes).unsqueeze(0))

print("conf_data (logits):")
print(conf_data)
print("\nconf_t (class indices):")
print(conf_t)
print("\ntargets_one_hot (one-hot encoded):")
print(targets_one_hot)
