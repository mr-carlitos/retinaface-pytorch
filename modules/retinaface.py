import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from data import NeckMode
from data.config_transform import transform_layer_config
from modules.net import conv_bn1X1
from modules.net import FPN as FPN
from modules.net import SSH as SSH
from modules.poolingarchitecture import PoolingArchitecture
from modules.attentionarchitecture import AttentionArchitecture

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
        self.shared_ssh = cfg['shared_ssh']
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

        if cfg['backbone_name'] == 'Resnet50-11k':
            import importlib.util
            import sys

            # Load the module using importlib.util
            spec = importlib.util.spec_from_file_location("MainModel", "/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/resnet-50-11k/resnet-50-ImageNet11k-final.py")
            MainModel = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(MainModel)

            # Register the module in sys.modules with the expected name
            sys.modules["MainModel"] = MainModel

            self.backbone = torch.load('/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/resnet-50-11k/resnet-50-ImageNet11k-final.pth', weights_only=False)
            self.backbone.train()

            if cfg['featuremaps_at_end_of_stage']:
                # For FPN
                for layer_idx in return_layers_sorted:
                    in_channels_list.append(int(in_channels_stage * (2 ** layer_idx)))
            else:
                # For FPN
                for layer_idx in return_layers_sorted:
                    if layer_idx == 3:
                        in_channels_list.append(int(in_channels_stage * (2**layer_idx)))
                    else:
                        in_channels_list.append(int(in_channels_stage * (2**(layer_idx-1))))

        elif cfg['backbone_name'] == 'Resnet50-1k':
            import torchvision.models as models
            resnet_pytorched_backbone = models.resnet50(pretrained=True)
            print("Loaded ResNet50-1k as backbone :)")
            #self.body -> First layer type where the inputs get fed through. It uses the backbone. 'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}.
            layer_dict = transform_layer_config(return_layers_sorted)
            self.backbone = _utils.IntermediateLayerGetter(resnet_pytorched_backbone, layer_dict)

            # For FPN
            for layer_idx in return_layers_sorted:
                in_channels_list.append(in_channels_stage * (2**layer_idx))
        
        else:
            self.backbone = None
            raise Exception("Invalid 'backbone-name' parameter in config. No valid backbone!")

        # out_channels_fpn is 256
        out_channels_fpn = cfg['out_channel']
        self.p6_in_neck = False

        if cfg['neck_mode'] == NeckMode.BASELINE_FPN:
            self.neck = FPN(in_channels_list, out_channels_fpn)

        elif cfg['neck_mode'] == NeckMode.ONLY_DECONV:
            self.neck = PoolingArchitecture(in_channels_list, out_channels_fpn, NeckMode.ONLY_DECONV)

        elif cfg['neck_mode'] == NeckMode.DECONV_POOLING:
            self.neck = PoolingArchitecture(in_channels_list, out_channels_fpn, NeckMode.DECONV_POOLING)

        elif cfg['neck_mode'] == NeckMode.ONlY_POOL:
            self.neck = PoolingArchitecture(in_channels_list, out_channels_fpn, NeckMode.ONlY_POOL)

        elif cfg['neck_mode'] == NeckMode.NEIGHBOURHOOD_DECONV_POOLING:
            self.neck = PoolingArchitecture(in_channels_list, out_channels_fpn, NeckMode.NEIGHBOURHOOD_DECONV_POOLING)

        elif cfg['neck_mode'] == NeckMode.CROSSATTENTION:
            in_channels_list.append(in_channels_stage)
            self.p6_in_neck = True
            self.neck = AttentionArchitecture(in_channels_list, out_channels_fpn, self.phase, cfg)

        else:
            self.conv1x1list = nn.ModuleList()
            for layer_index in range(len(self.cfg['return_layers'])):
                self.conv1x1list.append(conv_bn1X1(in_channels_list[layer_index], out_channels_fpn, stride = 1))

        if self.shared_ssh:
            self.shared_ssh_module = SSH(out_channels_fpn, out_channels_fpn)

        else:
            self.context_modules_list = nn.ModuleList()

            for num_featuremap in range(number_of_featuremaps):
                self.context_modules_list.append(SSH(out_channels_fpn, out_channels_fpn))

        anchor_num = cfg['anchor_num']

        self.ClassHead = self._make_class_head(fpn_num=number_of_featuremaps, inchannels=out_channels_fpn, anchor_num=anchor_num)
        self.BboxHead = self._make_bbox_head(fpn_num=number_of_featuremaps, inchannels=out_channels_fpn, anchor_num=anchor_num)

        self.feature_P6 = None
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

    def forward(self,x):
        ##### CARLOS CODE STARTS HERE #######################
        if self.cfg['backbone_name'] == 'Resnet50-11k':
            out_raw = self.backbone.orchestrate(x, self.cfg['featuremaps_at_end_of_stage'])
            indices = sorted(self.cfg['return_layers'].copy())
            out_filtered = list(out_raw[i] for i in indices)

            x = OrderedDict()
            for idx, key in enumerate(indices):
                x[key] = out_filtered[idx]

        else:
            x = self.backbone(x)

        if self.cfg['introduce_P6'] and 3 in self.cfg['return_layers']:
            #Remember that x is a dict, not a list. So we need to do x[3]
            self.feature_P6 = self.P6(x[3])

        if self.cfg['neck_mode'] != NeckMode.NO_FPN:
            # Apply the neck (standard FPN / Pooling Module / Attention Module)
            if self.p6_in_neck and (self.feature_P6 is not None):
                x[4] = self.feature_P6

            intermediate = self.neck(x)
        else:
            intermediate = list()
            j = 0
            for conv1x1 in self.conv1x1list:
                intermediate.append(conv1x1(x[j]))
                j += 1

        if self.feature_P6 is not None:
            intermediate.append(self.feature_P6)

        # Context Module (SSH): Can be shared or not shared
        x = list()

        if self.shared_ssh:
            for ii in intermediate:
                x.append(self.shared_ssh_module(ii))
        else:
            i = 0
            for context_module in self.context_modules_list:
                x.append(context_module(intermediate[i]))
                i += 1

        ##### CARLOS CODE ENDS HERE #######################
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(x)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(x)],dim=1)

        if self.phase == 'train':
            x = (bbox_regressions, classifications)
        else:
            x = (bbox_regressions, F.softmax(classifications, dim=-1))

        self.feature_P6 = None
        return x

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
