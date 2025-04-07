##### CARLOS CODE FILE ########

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
featuremaps = model.extract_features_as_mxnet(x)
print(output.shape)
print(output)
print(type(featuremaps))