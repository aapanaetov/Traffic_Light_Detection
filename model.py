import os
import sys
import yaml
import numpy as np
import torch
import torch.utils.data
import PIL
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_model(name):
    if name == 'res50_FPN':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif name == 'mobilenet_v3_FPN':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    elif name == 'mobilenet_v2_001':
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                        output_size=7,
                                                        sampling_ratio=2)
        model = FasterRCNN(backbone,
                           num_classes=5,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)

    return model