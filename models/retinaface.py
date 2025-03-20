import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from data.config_transform import transform_layer_config

from models.net import FPN as FPN
from models.net import SSH as SSH

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

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
        in_channels_stage2 = cfg['in_channel']


        ##### CARLOS CODE STARTS HERE #######################
        number_of_featuremaps = len(self.cfg['return_layers'])
        in_channels_list = None
        self.backbone = None

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

            for param in self.backbone.parameters():
                print(param)
                print(param.requires_grad)

            # For FPN
            in_channels_list = [
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 8,
            ]

        elif cfg['name'] == 'Resnet50-1k':
            import torchvision.models as models
            resnet_pytorched_backbone = models.resnet50(pretrained=cfg['pretrain'])
            print("Loaded ResNet50-1k as backbone :)")
            #self.body -> First layer type where the inputs get fed through. It uses the backbone. 'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}.
            layer_dict = transform_layer_config(cfg['return_layers'])
            self.backbone = _utils.IntermediateLayerGetter(resnet_pytorched_backbone, layer_dict)

            # For FPN
            in_channels_list = [
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]
        
        else:
            self.backbone = None
            raise Exception("Invalid 'name' parameter in config. No valid backbone!")

        if self.cfg['introduce_P6'] and 3 in self.cfg['return_layers']:
            self.P6 = self.create_P6()
            number_of_featuremaps += 1

        ##### CARLOS CODE ENDS HERE #######################

        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)

        #TODO: Create a list of ssh's, depending on if P6 is true and length of cfg.return_layers
        self.ssh_list = nn.ModuleList()

        for num_featuremap in range(number_of_featuremaps):
            self.ssh_list.append(SSH(out_channels, out_channels))

        self.ClassHead = self._make_class_head(fpn_num=number_of_featuremaps, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=number_of_featuremaps, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=number_of_featuremaps, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        ##### CARLOS CODE STARTS HERE #######################
        #TODO: Move these lines of code to a function so that RetinaFace object is clean
        if self.cfg['name'] == 'Resnet50-11k':
            out_raw = self.backbone.extract_features(inputs)
            indices = self.cfg['return_layers'].copy().sort()
            out_filtered = list(out_raw[i] for i in indices)

            out = OrderedDict()
            for idx, key in enumerate(indices):
                out[key] = out_filtered[idx]

        else:
            out = self.backbone(inputs)

        ##### CARLOS CODE ENDS HERE #######################

        # FPN
        out_copy = out.copy()
        fpn = self.fpn(out_copy)

        if hasattr(self, 'P6'):
            feature_P6 = self.P6(out[3])
            #TODO: Incorporate P6 into computation
            fpn.append(feature_P6)

        # SSH
        # TODO: Fix this code, as it does not work right now
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

    ##### CARLOS CODE STARTS HERE #######################
    def create_P6(self):
        # Assuming C5 features have in_channels = cfg['in_channel'] * 8
        in_channels_C5 = self.cfg['in_channel'] * 8
        out_channels = self.cfg['out_channel']
        P6 = nn.Conv2d(in_channels_C5, out_channels, kernel_size=3, stride=2, padding=1)
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(P6.weight)
        if P6.bias is not None:
            nn.init.constant_(P6.bias, 0)
        return P6
    ##### CARLOS CODE ENDS HERE #######################
