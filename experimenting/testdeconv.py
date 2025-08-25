##### CARLOS CODE FILE ########
import torch
import torch.nn as nn
import torch.nn.functional as F

# With square kernels and equal stride
m = nn.ConvTranspose2d(16, 33, 3, stride=2, padding=1, output_padding=1)
# non-square kernels and unequal stride and with padding
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output)


# exact output size can be also specified as an argument
input = torch.randn(1, 16, 12, 12)
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
h = downsample(input)
h.size()
output = upsample(h, output_size=input.size())
output.size()
