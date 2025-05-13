##### CARLOS CODE FILE ########

from models.net import conv_bn1X1, conv_bn, conv_bn_no_relu
import torch
import torch.nn as nn
import torch.nn.functional as F

def deconv_bn_relu(inp, oup, leaky = 0):
    num_groups = max(1, oup // 32) #For GroupNorm

    return nn.Sequential(
        #ConvTranspose2d = Deconv
        nn.ConvTranspose2d(inp, oup, 3, stride=2, padding=1, output_padding=1),
        nn.GroupNorm(num_groups, oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
##TODO: Pooling == max pooling?
class PoolingArchitecture(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(PoolingArchitecture,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        # Usually, in_channels_list = [128,256,512,2048] -> we need [2048,512,256,128]
        in_channels_list = list(reversed(in_channels_list))

        self.convlist1x1 = nn.ModuleList()
        for in_channel in in_channels_list:
            self.convlist1x1.append(conv_bn1X1(in_channel, out_channels, stride=1, leaky=leaky))

        self.deconvlist = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.deconvlist.append(deconv_bn_relu(out_channels, out_channels, leaky=leaky))

        self.merge_3x3conv_list_secondpart = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.merge_3x3conv_list_secondpart.append(conv_bn(out_channels, out_channels, leaky=leaky))

        self.merge_3x3conv_list_thirdpart = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.merge_3x3conv_list_thirdpart.append(conv_bn(out_channels, out_channels, leaky=leaky))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        # input = OrderedDict, 128,256,512,2048
        input = list(input.values())
        # input = List, 128,256,512,2048
        input = list(reversed(input))
        # input = List, 2048,512,256,128

        #First part: Let's apply 1x1conv to all our feature maps in order to obtain for all maps channel_size = 256
        output_list = list()
        for inp, layer in zip(input, self.convlist1x1):
            output_list.append(layer(inp))

        #Second part: Do the FPN (but with Deconvolution instead Nearest Neighbour Upsampling)
        last_layer_output = output_list[0]
        intermediate_outputs = list()

        for idx in range(1, len(input)):
            if idx == 1:
                output_variable = last_layer_output
            else:
                output_variable = intermediate_outputs[idx - 2]
            # size(2): Height
            # size(3): Width
            up = self.deconvlist[idx -1](output_variable) # Deconv usage here
            addition = output_list[idx] + up
            merged = self.merge_3x3conv_list_secondpart[idx - 1](addition)
            intermediate_outputs.append(merged)
        intermediate_outputs = list(reversed(intermediate_outputs))
        intermediate_outputs.append(last_layer_output) # in here, we have the structure [P2, P3, P4, P5]


        #Third part: apply Pooling so that the upper levels get informed by the lower levels
        final_outputs = list()
        first_layer_intermediate = intermediate_outputs[0]

        for idx in range(len(intermediate_outputs)-1):
            if idx == 0:
                intermediate_variable = first_layer_intermediate
            else:
                intermediate_variable = final_outputs[idx-1]
            pooled = self.pool(intermediate_variable)
            pool_applied = intermediate_outputs[idx+1] + pooled
            merged_final = self.merge_3x3conv_list_thirdpart[idx](pool_applied)
            final_outputs.append(merged_final)
        final_outputs = [first_layer_intermediate] + final_outputs
        return final_outputs