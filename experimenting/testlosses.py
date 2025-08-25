##### CARLOS CODE FILE ########
import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

# Number of anchors
N = 5000
alpha=0.25
gamma=2


scores = torch.randn(N, 2)

# One-hot targets as float
class_inds = torch.randint(0, 2, (N,))
targets = torch.zeros(N, 2, dtype=torch.float)
targets.scatter_(1, class_inds.unsqueeze(1), 1)

# Sigmoid focal loss
loss_sig = sigmoid_focal_loss(
    inputs=scores,
    targets=targets,
    alpha=alpha,
    gamma=gamma,
    reduction="sum"
)

#Softmax focal loss

log_probs = F.log_softmax(scores, dim=1)
log_pt = log_probs[torch.arange(N), class_inds]
pt     = log_pt.exp()
focal_weight = (1 - pt).pow(gamma)
alpha_factor = torch.where(class_inds == 1,
                           alpha,
                           1.0 - alpha).to(pt.dtype)
loss_sm = -(alpha_factor * focal_weight * log_pt).sum()

print(f"Sigmoid focal loss: {loss_sig.item():.4f}")
print(f"Softmax focal loss: {loss_sm.item():.4f}")
