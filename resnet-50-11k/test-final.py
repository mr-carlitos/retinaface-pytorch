##### CARLOS CODE FILE ########
## This is a file where I experimented around with the imported ImageNet-11k backbone, you can ignore this
import numpy as np
import torch
from PIL import Image
import importlib.util
import sys

torch.cuda.set_device(5)

# Load the module using importlib.util
spec = importlib.util.spec_from_file_location("MainModel", "./resnet-50-ImageNet11k-final.py")
MainModel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MainModel)

# Register the module in sys.modules with the expected name
sys.modules["MainModel"] = MainModel

model = torch.load('./resnet-50-ImageNet11k-final.pth').cuda()
model.eval()

np.random.seed(42)
dummy_data = np.random.rand(1, 3, 640, 640)

# Convert to torch.Tensor
x = torch.from_numpy(dummy_data).float().cuda()
output = model(x)
with torch.no_grad():
    featuremaps = model.extract_features_after_stage(x)
    featuremaps_new = model.extract_features_after_stage_v2(x)

listed = [torch.sum(torch.abs(featuremaps_new[i] - featuremaps[i])) for i in range(len(featuremaps))]
tensored = torch.stack(listed)

difference = torch.abs(tensored)
summed = torch.sum(difference)

print(difference)
print(summed)