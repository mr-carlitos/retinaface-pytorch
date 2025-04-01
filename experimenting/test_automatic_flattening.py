##### CARLOS CODE FILE ########
import torch

# Let's create a small example
batch_size = 2
num_priors = 4

# Target class indices: shape [batch_size, num_priors]
conf_t = torch.tensor([
    [0, 1, 0, 0],  # First image in batch
    [0, 0, 1, 0]   # Second image in batch
])

# Positive anchors: shape [batch_size, num_priors]
pos = torch.tensor([
    [False, True, False, False],  # First image has one positive at index 1
    [False, False, True, False]   # Second image has one positive at index 2
])

# Selected negative anchors: shape [batch_size, num_priors]
neg = torch.tensor([
    [True, False, False, False],  # First image selects one negative at index 0
    [True, False, False, True]    # Second image selects two negatives at indices 0 and 3
])

# Combined mask for positive and negative anchors
pos_plus_neg = pos + neg
print("Combined mask:")
print(pos_plus_neg)

# Selecting elements from conf_t using the mask
targets_weighted = conf_t[pos_plus_neg]
print("\nSelected targets (flattened):")
print(targets_weighted)
print("Shape:", targets_weighted.shape)