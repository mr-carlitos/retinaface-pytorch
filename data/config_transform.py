##### CARLOS CODE FILE ########

def transform_layer_config(return_layers):
    layer_list = sorted(return_layers)
    layer_dict = dict()

    for layer in layer_list:
        if layer == 0:
            layer_dict["layer1"] = 1

        elif layer == 1:
            layer_dict["layer2"] = 2
        
        elif layer == 2:
            layer_dict["layer3"] = 3

        elif layer == 3:
            layer_dict["layer4"] = 4

        else:
            raise Exception("Invalid key detected in config-transform.py!")
    return layer_dict
