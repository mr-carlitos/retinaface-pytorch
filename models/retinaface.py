import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from data.config_transform import transform_layer_config

from models.net import FPN as FPN
from models.net import SSH as SSH

class ClassHead(nn.Module):
    def __init__(self, inchannels, num_anchors):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors*2, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self, inchannels, num_anchors):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.cfg = cfg
        self.phase = phase
        # in_channel = 256
        in_channels_stage = cfg['in_channel']

        ##### CARLOS CODE STARTS HERE #######################
        number_of_featuremaps = len(self.cfg['return_layers'])
        return_layers_sorted = sorted(self.cfg['return_layers'])

        if self.cfg['introduce_P6'] and 3 in self.cfg['return_layers']:
            self.P6 = self.create_P6()
            number_of_featuremaps += 1

        self.backbone = None
        in_channels_list = list()

        if cfg['name'] == 'Resnet50-11k':
            import importlib.util
            import sys

            # Load the module using importlib.util
            spec = importlib.util.spec_from_file_location("MainModel", "./resnet-50-11k/resnet-50-ImageNet11k-final.py")
            MainModel = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(MainModel)

            # Register the module in sys.modules with the expected name
            sys.modules["MainModel"] = MainModel

            self.backbone = torch.load('./resnet-50-11k/resnet-50-ImageNet11k-final.pth')
            self.backbone.train()

            # For FPN
            for layer_idx in return_layers_sorted:
                if layer_idx == 3:
                    in_channels_list.append(int(in_channels_stage * (2**layer_idx)))
                else:
                    in_channels_list.append(int(in_channels_stage * (2**(layer_idx-1))))

        elif cfg['name'] == 'Resnet50-1k':
            import torchvision.models as models
            resnet_pytorched_backbone = models.resnet50(pretrained=cfg['pretrain'])
            print("Loaded ResNet50-1k as backbone :)")
            #self.body -> First layer type where the inputs get fed through. It uses the backbone. 'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}.
            layer_dict = transform_layer_config(return_layers_sorted)
            self.backbone = _utils.IntermediateLayerGetter(resnet_pytorched_backbone, layer_dict)

            # For FPN
            for layer_idx in return_layers_sorted:
                in_channels_list.append(in_channels_stage * (2**layer_idx))
        
        else:
            self.backbone = None
            raise Exception("Invalid 'name' parameter in config. No valid backbone!")
        # out_channels_fpn is 256
        out_channels_fpn = cfg['out_channel']

        self.fpn = FPN(in_channels_list, out_channels_fpn)

        self.context_modules_list = nn.ModuleList()

        for num_featuremap in range(number_of_featuremaps):
            self.context_modules_list.append(SSH(out_channels_fpn, out_channels_fpn))

        anchor_num = cfg['anchor_num']

        self.ClassHead = self._make_class_head(fpn_num=number_of_featuremaps, inchannels=out_channels_fpn, anchor_num=anchor_num)
        self.BboxHead = self._make_bbox_head(fpn_num=number_of_featuremaps, inchannels=out_channels_fpn, anchor_num=anchor_num)
        ##### CARLOS CODE ENDS HERE #######################

    def _make_class_head(self, fpn_num, inchannels, anchor_num):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead
    
    def _make_bbox_head(self, fpn_num, inchannels, anchor_num):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def forward(self,inputs):
        ##### CARLOS CODE STARTS HERE #######################
        #TODO: Move these lines of code to a function so that RetinaFace object is clean
        if self.cfg['name'] == 'Resnet50-11k':
            out_raw = self.backbone(inputs)
            indices = sorted(self.cfg['return_layers'].copy())
            out_filtered = list(out_raw[i] for i in indices)

            out = OrderedDict()
            for idx, key in enumerate(indices):
                out[key] = out_filtered[idx]

        else:
            out = self.backbone(inputs)

        # FPN
        fpn = self.fpn(out)

        if self.cfg['introduce_P6'] and 3 in self.cfg['return_layers']:
            #Remember that out is a dict, not a list. So we need to do out[3]
            feature_P6 = self.P6(out[3])
            fpn.append(feature_P6)

        # Context Module
        i = 0
        features = list()
        for context_module in self.context_modules_list:
            features.append(context_module(fpn[i]))
            i += 1

        ##### CARLOS CODE ENDS HERE #######################
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1))
        return output

    ##### CARLOS CODE STARTS HERE #######################
    def create_P6(self):
        # Assuming C5 features have in_channels = cfg['in_channel'] * 8 = 2048
        in_channels_C5 = self.cfg['in_channel'] * 8
        out_channels = self.cfg['out_channel']
        P6 = nn.Conv2d(in_channels_C5, out_channels, kernel_size=3, stride=2, padding=1)
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(P6.weight)
        if P6.bias is not None:
            nn.init.constant_(P6.bias, 0)
        return P6
    ##### CARLOS CODE ENDS HERE #######################
