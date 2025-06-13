##### CARLOS CODE FILE ########
from data import NeckMode
from models.net import conv_bn1X1, conv_bn
import torch.nn as nn

def deconv_bn_relu(inp, oup, leaky = 0):
    num_groups = max(1, oup // 32) #For GroupNorm

    return nn.Sequential(
        #ConvTranspose2d = Deconv
        nn.ConvTranspose2d(inp, oup, 3, stride=2, padding=1, output_padding=1),
        nn.GroupNorm(num_groups, oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

class PoolingArchitecture(nn.Module):
    def __init__(self, in_channels_list, out_channels, mode):
        super(PoolingArchitecture,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        self.mode = mode

        # Usually, in_channels_list = [128,256,512,2048] -> we need [2048,512,256,128]
        in_channels_list = list(reversed(in_channels_list))

        self.convlist1x1 = nn.ModuleList()
        for in_channel in in_channels_list:
            self.convlist1x1.append(conv_bn1X1(in_channel, out_channels, stride=1, leaky=leaky))

        if mode == NeckMode.DECONV_POOLING or mode == NeckMode.NEIGHBOURHOOD_DECONV_POOLING or mode == NeckMode.ONLY_DECONV:
            self.deconvlist = nn.ModuleList()
            for _ in range(len(in_channels_list) - 1):
                self.deconvlist.append(deconv_bn_relu(out_channels, out_channels, leaky=leaky))

        self.merge_3x3conv_list_secondpart = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.merge_3x3conv_list_secondpart.append(conv_bn(out_channels, out_channels, leaky=leaky))

        if mode == NeckMode.DECONV_POOLING:
            self.merge_3x3conv_list_thirdpart = nn.ModuleList()
            for _ in range(len(in_channels_list) - 1):
                self.merge_3x3conv_list_thirdpart.append(conv_bn(out_channels, out_channels, leaky=leaky))

        if mode == NeckMode.DECONV_POOLING or mode == NeckMode.NEIGHBOURHOOD_DECONV_POOLING or mode == NeckMode.ONlY_POOL:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def preprocessing(self, inp):
        # input = OrderedDict, 128,256,512,2048
        inp = list(inp.values())
        # input = List, 128,256,512,2048
        inp = list(reversed(inp))
        # input = List, 2048,512,256,128

        # First part: Let's apply 1x1conv to all our feature maps in order to obtain for all maps channel_size = 256
        output_list = list()

        for i, layer in zip(inp, self.convlist1x1):
            output_list.append(layer(i))
        return output_list, inp

    def forward(self, inp):
        if self.mode == NeckMode.ONlY_POOL:
            return self.forward_onlypooling(inp)
        elif self.mode == NeckMode.NEIGHBOURHOOD_DECONV_POOLING:
            return self.forward_neighbourhood(inp)
        else:
            return self.forward_not_neighbourhood(inp)

    def forward_onlypooling(self, inp):
        output_list, inp = self.preprocessing(inp)
        #List, 128,256,512,2048
        output_list = list(reversed(output_list))

        #We get C2
        first_layer_output = output_list[0]
        intermediate_outputs = list()

        for idx in range(1, len(inp)):
            if idx == 1:
                x = first_layer_output
            else:
                x = intermediate_outputs[-1]

            pooled = self.pool(x)
            x = output_list[idx] + pooled
            x = self.merge_3x3conv_list_secondpart[idx - 1](x)
            intermediate_outputs.append(x)

        return [first_layer_output] + intermediate_outputs

    def forward_neighbourhood(self, inp):
        output_list, inp = self.preprocessing(inp)

        #Second part: Do the FPN (but with Deconvolution instead Nearest Neighbour Upsampling)
        last_layer_output = output_list[0]
        intermediate_outputs = list()

        for idx in range(1, len(inp)):
            if idx == 1:
                x = last_layer_output
            else:
                x = intermediate_outputs[idx - 2]
            # size(2): Height
            # size(3): Width
            #x = self.deconvlist[idx -1](x, output_size=output_list[idx].shape[2:]) # Commented out because of dimensionality errors at inference / evaluation time, see next comment

            # Comment: I wrote the following lines of code, because at inference / evaluation time, our feature maps aren't necessarily divisible by 2
            deconv_block = self.deconvlist[idx - 1]  # this is nn.Sequential([ConvT, GN, ReLU])
            target_size = output_list[idx].shape[2:]  # (H, W) of the lateral feature

            # 1) call the ConvTranspose2d with output_size
            convT = deconv_block[0]  # nn.ConvTranspose2d
            x = convT(x, output_size=target_size)  # now x.shape == shape of next layer

            # 2) run through the rest of the sequential
            for m in list(deconv_block.children())[1:]:
                x = m(x)

            x = output_list[idx] + x
            if idx + 1 < len(output_list):
                pooled = self.pool(output_list[idx + 1])
                x = x + pooled
            x = self.merge_3x3conv_list_secondpart[idx - 1](x)
            intermediate_outputs.append(x)
        intermediate_outputs = list(reversed(intermediate_outputs))
        intermediate_outputs.append(last_layer_output) # in here, we have the structure [P2, P3, P4, P5]

        return intermediate_outputs

    def forward_not_neighbourhood(self, inp):
        output_list, inp = self.preprocessing(inp)

        #Second part: Do the FPN (but with Deconvolution instead Nearest Neighbour Upsampling)
        last_layer_output = output_list[0]
        intermediate_outputs = list()

        for idx in range(1, len(inp)):
            if idx == 1:
                x = last_layer_output
            else:
                x = intermediate_outputs[idx - 2]
            # size(2): Height
            # size(3): Width
            #x = self.deconvlist[idx -1](x, output_size=output_list[idx].shape[2:]) # Commented out because of dimensionality errors at inference / evaluation time, see next comment

            # Comment: I wrote the following lines of code, because at inference / evaluation time, our feature maps aren't necessarily divisible by 2
            deconv_block = self.deconvlist[idx - 1]  # this is nn.Sequential([ConvT, GN, ReLU])
            target_size = output_list[idx].shape[2:]  # (H, W) of the lateral feature

            # 1) call the ConvTranspose2d with output_size
            convT = deconv_block[0]  # nn.ConvTranspose2d
            x = convT(x, output_size=target_size)  # now x.shape == shape of next layer

            # 2) run through the rest of the sequential
            for m in list(deconv_block.children())[1:]:
                x = m(x)

            x = output_list[idx] + x
            x = self.merge_3x3conv_list_secondpart[idx - 1](x)
            intermediate_outputs.append(x)
        intermediate_outputs = list(reversed(intermediate_outputs))
        intermediate_outputs.append(last_layer_output) # in here, we have the structure [P2, P3, P4, P5]

        if self.mode == NeckMode.DECONV_POOLING:
            #Third part: apply Pooling so that the upper levels get informed by the lower levels
            final_outputs = list()
            first_layer_intermediate = intermediate_outputs[0]

            for idx in range(len(intermediate_outputs)-1):
                if idx == 0:
                    x = first_layer_intermediate
                else:
                    x = final_outputs[idx-1]
                x = self.pool(x)
                x = intermediate_outputs[idx+1] + x
                x = self.merge_3x3conv_list_thirdpart[idx](x)
                final_outputs.append(x)
            final_outputs = [first_layer_intermediate] + final_outputs
            return final_outputs
        else:
            return intermediate_outputs