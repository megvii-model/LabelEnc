"""
Implements the Feature Pyramid Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import build_backbone, build_custom_basemodel_backbone
from ..backbone.label_encoding_function import LabelEncodingFunction
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

from fcos_core.layers.ground_truth_encode import box_encode
from fcos_core.structures.image_list import to_image_list


class LabelEncStep2Network(nn.Module):
    def __init__(self, cfg):
        super(LabelEncStep2Network, self).__init__()
        self.backbone = build_backbone(cfg)

        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        label_encoding_function = LabelEncodingFunction(self.num_classes - 1)
        self.label_encoding_function = build_custom_basemodel_backbone(cfg, label_encoding_function)
        for p in self.label_encoding_function.parameters():
            p.requires_grad = False

        out_channels = self.backbone.out_channels
        self.rpn = build_rpn(cfg, out_channels)
        self.roi_heads = build_roi_heads(cfg, out_channels)

        self.feature_adapt = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.layernorm = nn.GroupNorm(num_groups=1, num_channels=out_channels, affine=False)

    def forward(self, images, targets=None, auxiliary_loss_weight=1.0):
        images = to_image_list(images)

        if self.training:
            h, w = images.tensors.size()[-2:]
            gts = box_encode(targets, h, w, self.num_classes - 1, aug=False)
            label_encodings = self.label_encoding_function(gts)

        features = self.backbone(images.tensors)

        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)

            # calculate auxiliary loss
            with torch.no_grad():
                n = features[0].size(0)
                label_encodings = [self.layernorm(i) for i in label_encodings]
                label_encodings = torch.cat([i.view(n, self.backbone.out_channels, -1) for i in label_encodings], dim=2)

            features = [self.layernorm(self.feature_adapt(i)) for i in features]
            features = torch.cat([i.view(n, self.backbone.out_channels, -1) for i in features], dim=2)

            losses.update({"auxiliary loss": F.mse_loss(label_encodings, features) * auxiliary_loss_weight})

            return losses

        else:
            return result

