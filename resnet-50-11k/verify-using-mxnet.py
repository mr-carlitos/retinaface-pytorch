##### CARLOS CODE FILE ########

import mxnet as mx
import numpy as np

# Paths to your model files
symbol_file = 'resnet-50-symbol.json'
params_file = 'resnet-50-0000.params'

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

# Verify by doing a forward pass with dummy data:
np.random.seed(42)
dummy_data = mx.nd.array(np.random.rand(1, 3, 640, 640))

mod.forward(mx.io.DataBatch([dummy_data]))

outputs = mod.get_outputs()
output_data = outputs[0]  # Get the first output tensor

print("Output shape:", output_data.shape)
print(output_data)



## This section below was used to find out how the original RetinaFace authors compute the feature maps

#F1 = 0
#F2 = 0
#
#def get_sym_conv(sym):
#    all_layers = sym.get_internals()
#    isize = 640
#    _, out_shape, _ = all_layers.infer_shape(data=(1, 3, isize, isize))
#    outputs = all_layers.list_outputs()
#    count = len(outputs)
#    stride2name = {}
#    stride2layer = {}
#    stride2shape = {}
#    for i in range(count):
#        name = outputs[i]
#        shape = out_shape[i]
#        print(i, name, count, shape)
#        if not name.endswith('_output'):
#            continue
#        if len(shape) != 4:
#            continue
#        assert isize % shape[2] == 0
#        if shape[1] > 9999:
#            break
#        stride = isize // shape[2]
#        stride2name[stride] = name
#        stride2layer[stride] = all_layers[name]
#        stride2shape[stride] = shape
#    strides = sorted(stride2name.keys())
#    for stride in strides:
#        print('stride', stride, stride2name[stride], stride2shape[stride])



#get_sym_conv(sym)