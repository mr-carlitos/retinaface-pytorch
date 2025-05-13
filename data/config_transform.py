##### CARLOS CODE FILE ########
from enum import Enum, auto

# FPN is called "neck"
class NeckMode(Enum):
    # Uses FPN, just as introduced in RetinaFace
    BASELINE_FPN = auto()

    # Instead of using a FPN, apply only a 1x1conv on each backbone feature map
    NO_FPN = auto()

    # Uses a new neck mechanism with pooling, where we first have a similar mechanism like FPN (but with Deconvolution instead of NN upsampling), and then lower levels inform
    # upper layers via pooling and element-wise addition
    DECONV_POOLING = auto()

    # TODO
    POOLING_DECONV = auto()

    # TODO
    NEIGHBOURHOOD_POOLING_DECONV = auto()

    # TODO
    CROSSATTENTION_FROMUPPER = auto()

    # TODO
    CROSSATTENTION_FROMUPPERANDLOWER = auto()


def transform_layer_config(return_layers):
    layer_dict = dict()
    layer_list = return_layers.copy()

    for layer in layer_list:
        if layer == 0:
            layer_dict["layer1"] = 0

        elif layer == 1:
            layer_dict["layer2"] = 1
        
        elif layer == 2:
            layer_dict["layer3"] = 2

        elif layer == 3:
            layer_dict["layer4"] = 3

        else:
            raise Exception("Invalid key detected in config-transform.py!")
    return layer_dict
