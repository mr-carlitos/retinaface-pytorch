##### CARLOS CODE FILE ########

import mxnet as mx
import numpy as np
import torch
from PIL import Image
import importlib.util
import sys

torch.cuda.set_device(5)

def get_torch_kitmodel():
    # Load the module using importlib.util
    spec = importlib.util.spec_from_file_location("MainModel", "./resnet-50-ImageNet11k-final.py")
    MainModel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MainModel)

    # Register the module in sys.modules with the expected name
    sys.modules["MainModel"] = MainModel

    model = torch.load('./resnet-50-ImageNet11k-final.pth').cuda()
    model.eval()
    return model

def get_mxnet_originalmodel():
    symbol_file = './mxnet-files/resnet-50-symbol.json'
    params_file = './mxnet-files/resnet-50-0000.params'

    # Load the network symbol
    sym = mx.sym.load(symbol_file)

    # Load parameters; they are stored with keys like "arg:layer_name" and "aux:layer_name"
    save_dict = mx.nd.load(params_file)
    arg_params = {}
    aux_params = {}
    for key, value in save_dict.items():
        tp, name = key.split(":", 1)
        if tp == "arg":
            arg_params[name] = value
        elif tp == "aux":
            aux_params[name] = value

    # Create an MXNet Module from the symbol
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())

    # Bind the module to an input shape
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 640, 640))])

    # Set the parameters (weights) into the module
    mod.set_params(arg_params, aux_params)

    # Save the symbol as an attribute for later use.
    mod._sym = sym

    return mod


def get_intermediate_output(model, data, layer_name):
    """
    Extract intermediate feature maps from a bound MXNet module

    Parameters:
    -----------
    model: mx.mod.Module
        The bound MXNet module
    data: mx.nd.array
        Input data
    layer_name: str
        Name of the internal layer to extract

    Returns:
    --------
    feature_map: numpy.ndarray
        The feature map of the specified layer
    """
    # Get the original model's internal symbols
    all_layers = model._sym.get_internals()

    # Select the desired layer
    sym = all_layers[layer_name + '_output']

    # Create a new module with the modified symbol
    new_mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=model.data_names)

    # Set the same shape and parameters as the original model
    new_mod.bind(for_training=False, data_shapes=model.data_shapes)

    # Share parameters with the original model
    arg_params, aux_params = model.get_params()
    new_mod.set_params(arg_params, aux_params)

    # Forward pass with the provided data
    new_mod.forward(mx.io.DataBatch([data]))

    # Get the output
    outputs = new_mod.get_outputs()
    return outputs[0].asnumpy()

layer_list = ["stage2_unit1_relu2", "stage3_unit1_relu2", "stage4_unit1_relu2", "relu1"]


mod = get_mxnet_originalmodel()
model = get_torch_kitmodel()

np.random.seed(44)
dummy_data = np.random.rand(1, 3, 640, 640)

# Convert to torch.Tensor
x = torch.from_numpy(dummy_data).float().cuda()
output = model(x)
#featuremaps = model.extract_features(x)
print(output.shape)
print(output)


dummy_data = mx.nd.array(dummy_data)

mod.forward(mx.io.DataBatch([dummy_data]))

outputs = mod.get_outputs()
output_data = outputs[0]  # Get the first output tensor

print(output_data.shape)
print(output_data)

# After getting both outputs, add these lines:

# Convert the PyTorch output to NumPy on CPU
torch_output = output.detach().cpu().numpy()

# Convert the MXNet output to NumPy
mxnet_output = output_data.asnumpy()

# Ensure shapes match
print(f"PyTorch output shape: {torch_output.shape}")
print(f"MXNet output shape: {mxnet_output.shape}")

# Calculate absolute difference
abs_diff = np.abs(torch_output - mxnet_output)

# Calculate the sum of all absolute differences
total_abs_diff = np.sum(abs_diff)
print(f"Total absolute difference: {total_abs_diff}")

# Print statistics
print(f"Mean absolute difference: {np.mean(abs_diff)}")
print(f"Max absolute difference: {np.max(abs_diff)}")
print(f"Min absolute difference: {np.min(abs_diff)}")
print(f"Standard deviation of differences: {np.std(abs_diff)}")


feature_maps_mx = [get_intermediate_output(mod, dummy_data, i) for i in layer_list]
#extract_features() returns a tuple, not a list
feature_maps_torch = model.extract_features(x)
feature_maps_torch = [feature_maps_torch[i] for i in range(len(feature_maps_torch))]


for i, (fm_mx, fm_torch) in enumerate(zip(feature_maps_mx, feature_maps_torch)):
    # Convert Torch tensor to NumPy array
    fm_torch_np = fm_torch.detach().cpu().numpy()
    if fm_mx.shape == fm_torch_np.shape:
        abs_diff = np.abs(fm_mx - fm_torch_np)
        print(f"Layer {layer_list[i]} - Total absolute difference: {np.sum(abs_diff)}")
    else:
        print(f"Layer {layer_list[i]} shapes differ: MXNet {fm_mx.shape}, Torch {fm_torch_np.shape}")

print("done")