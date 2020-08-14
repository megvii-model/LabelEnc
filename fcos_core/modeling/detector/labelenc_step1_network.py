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


class LabelEncStep1Network(nn.Module):
    def __init__(self, cfg):
        super(LabelEncStep1Network, self).__init__()
        self.backbone = build_backbone(cfg)

        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        label_encoding_function = LabelEncodingFunction(self.num_classes - 1)
        self.label_encoding_function = build_custom_basemodel_backbone(cfg, label_encoding_function)

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

    def forward(self, images, targets=None, distance_loss_weight=1.0):
        images = to_image_list(images)

        if self.training:
            h, w = images.tensors.size()[-2:]
            gts = box_encode(targets, h, w, self.num_classes - 1, aug=True)
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

            ae_proposals, ae_proposal_losses = self.rpn(images, label_encodings, targets)
            if self.roi_heads:
                ae_x, ae_results, ae_detector_losses = self.roi_heads(label_encodings, ae_proposals, targets)
            else:
                # RPN-only models don't have roi_heads
                ae_x = label_encodings
                ae_result = ae_proposals
                ae_detector_losses = {}

            for k, v in ae_proposal_losses.items():
                losses['ae_' + k] = v
            for k, v in ae_detector_losses.items():
                losses['ae_' + k] = v
            losses.update(proposal_losses)
            losses.update(detector_losses)

            # calculate distance loss
            with torch.no_grad():
                n = features[0].size(0)
                label_encodings = [self.layernorm(i) for i in label_encodings]
                label_encodings = torch.cat([i.view(n, self.backbone.out_channels, -1) for i in label_encodings], dim=2)
                label_encodings.detach_()

            features = [self.layernorm(self.feature_adapt(i)) for i in features]
            features = torch.cat([i.view(n, self.backbone.out_channels, -1) for i in features], dim=2)

            losses.update({"distance loss": F.mse_loss(label_encodings, features) * distance_loss_weight})

            return losses

        else:
            return result

