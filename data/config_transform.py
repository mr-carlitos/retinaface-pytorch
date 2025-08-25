##### CARLOS CODE FILE ########
from enum import Enum, auto

# FPN is called "neck"
class NeckMode(Enum):
    # Uses FPN, just as introduced in RetinaFace
    BASELINE_FPN = auto()

    # Instead of using a FPN, apply only a 1x1conv on each backbone feature map
    NO_FPN = auto()

    # RQ2 a.)
    ONLY_DECONV = auto()

    # RQ2 b.)
    ONlY_POOL = auto()

    # RQ2 c.)
    DECONV_POOLING = auto()

    # RQ2 d.)
    NEIGHBOURHOOD_DECONV_POOLING = auto()

    # RQ3
    CROSSATTENTION = auto()

    # IGNORED: Alternative cross-attention module, results from this variant were ignored when writing the thesis, since the results were not very good.
    DIRECT_CROSSATTENTION = auto()


class PositionalMode(Enum):
    # Learned positional embeddings
    POS_EMBEDDING_QKV = auto()

    # 2D sinus/cosinus positional encoding for Q and K maps
    POS_ENCODING_QKV = auto()

    # No positional information
    NOTHING = auto()

# helper function
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
