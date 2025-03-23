import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)

        #TODO: Is this relu here necessary?
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        in_channels_list = list(reversed(in_channels_list))

        ##### CARLOS CODE STARTS HERE #######################
        self.fpn_list = nn.ModuleList()
        for in_channel in in_channels_list:
            self.fpn_list.append(conv_bn1X1(in_channel, out_channels, stride = 1, leaky = leaky))

        self.merge_list = nn.ModuleList()
        for _ in range(len(in_channels_list)-1):
            self.merge_list.append(conv_bn(out_channels, out_channels, leaky = leaky))

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())
        input = list(reversed(input))

        output_list = list()
        for inp, layer in zip(input, self.fpn_list):
            output_list.append(layer(inp))

        last_layer_output = output_list[0]
        final_outputs = list()

        for idx in range(1, len(input)):
            if idx == 1:
                output_variable = last_layer_output
            else:
                output_variable = final_outputs[idx-2]
            # TODO: Is "mode=nearest" correct??
            # size(2): Height
            # size(3): Width
            up = F.interpolate(output_variable, size=[output_list[idx].size(2), output_list[idx].size(3)], mode="nearest")
            addition = output_list[idx] + up
            # TODO: Find out why we use a conv layer with activation here, shouldn't it be one without, just as the guys in the FPN paper say?
            merged = self.merge_list[idx-1](addition)
            final_outputs.append(merged)
        final_outputs = list(reversed(final_outputs))
        final_outputs.append(last_layer_output)
        return final_outputs