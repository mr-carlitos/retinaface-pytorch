##### CARLOS CODE FILE ########

def transform_layer_config(return_layers):
    layers_listed = list(return_layers.keys())

    layer_indices = list()

    for layer in layers_listed:
        if layer == "layer1":
            layer_indices.append(0)

        elif layer == "layer2":
            layer_indices.append(1)
        
        elif layer == "layer3":
            layer_indices.append(2)

        elif layer == "layer4":
            layer_indices.append(3)

        else:
            print("Invalid key detected in config-transform.py!")
    return layer_indices
