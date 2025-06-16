import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inp, oup, stride = 1, leaky = 0):
    num_groups = max(1, oup // 32) #For GroupNorm

    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        #nn.BatchNorm2d(oup),
        nn.GroupNorm(num_groups, oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    num_groups = max(1, oup // 32) #For GroupNorm

    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.GroupNorm(num_groups, oup),
        #nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    num_groups = max(1, oup // 32) #For GroupNorm

    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        #nn.BatchNorm2d(oup),
        nn.GroupNorm(num_groups, oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        #setting the leak parameter to 0 makes leaky ReLU equivalent to the standard ReLU
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, x):
        conv3X3 = self.conv3X3(x)

        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        x = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)

        x = F.relu(x)
        # Apply DCN if enabled
        #if self.use_dcn:
        #    out = self.dcn(out)
        return x

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        # Usually, in_channels_list = [128,256,512,2048] -> we need [2048,512,256,128]
        in_channels_list = list(reversed(in_channels_list))

        ##### CARLOS CODE STARTS HERE #######################
        self.fpn_list = nn.ModuleList()
        for in_channel in in_channels_list:
            self.fpn_list.append(conv_bn1X1(in_channel, out_channels, stride = 1, leaky = leaky))

        self.merge_list = nn.ModuleList()
        for _ in range(len(in_channels_list)-1):
            self.merge_list.append(conv_bn(out_channels, out_channels, leaky = leaky))

    def forward(self, input):
        # input = OrderedDict, 128,256,512,2048
        input = list(input.values())
        # input = List, 128,256,512,2048
        input = list(reversed(input))
        # input = List, 2048,512,256,128

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
            # size(2): Height
            # size(3): Width
            up = F.interpolate(output_variable, size=[output_list[idx].size(2), output_list[idx].size(3)], mode="nearest")
            addition = output_list[idx] + up
            merged = self.merge_list[idx-1](addition)
            final_outputs.append(merged)
        final_outputs = list(reversed(final_outputs))
        final_outputs.append(last_layer_output)
        return final_outputs