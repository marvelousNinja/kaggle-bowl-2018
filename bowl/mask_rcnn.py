from functools import partial
from collections import namedtuple

import torch
import numpy as np
from torch.nn.functional import softmax
from torch.nn.functional import sigmoid

from bowl.mask_heads import MaskHead
from bowl.rpns import RPN
from bowl.roi_heads import RoIHead
from bowl.utils import construct_boxes
from bowl.utils import from_numpy
from bowl.utils import to_numpy
from bowl.utils import non_max_suppression
from bowl.utils import generate_anchors
from roi_align.crop_and_resize import CropAndResizeFunction

MaskRCNNOutputs = namedtuple('MaskRCNNOutputs', [
    'rpn_logits', 'rpn_deltas', 'rpn_proposals', 'anchors', 'rcnn_logits', 'rcnn_deltas', 'rcnn_masks', 'rcnn_detections', 'rcnn_detection_masks', 'rcnn_detection_scores', 'image_shape'
])

class MaskRCNN(torch.nn.Module):
    def __init__(self, backbone, scales, ratios):
        super(MaskRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = RPN(input_channels=self.backbone.output_channels(), anchors_per_location=len(scales) * len(ratios))
        self.roi_shape = (self.backbone.output_channels(), 7, 7)
        self.rcnn = RoIHead(self.roi_shape, num_classes=1)
        self.generate_anchors = partial(generate_anchors, self.backbone.stride(), scales, ratios)
        self.mask_head = MaskHead(self.backbone.output_channels())

    def forward(self, x):
        image_shape = x.shape[2:]
        anchors = self.generate_anchors(image_shape)
        x = self.backbone(x)
        rpn_logits, rpn_deltas = self.rpn(x)

        # NMS block
        # 1. Convert to numpy and to bboxes
        scores = softmax(rpn_logits[0], dim=1)[:, [1]]
        boxes = construct_boxes(to_numpy(rpn_deltas[0]), anchors)

        # 2. Clip bboxes to image shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_shape[1] - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_shape[0] - 1)

        # 3. Convert to pytorch and perform actual NMS
        keep_indicies = non_max_suppression(from_numpy(boxes), scores, iou_threshold=0.5)[:2000]
        rpn_proposals = boxes[keep_indicies]

        # RoI Heads block
        # 1. Normalize boxes to 0..1 coordinates
        normalized_boxes = rpn_proposals.copy()
        normalized_boxes[:, [0, 2]] /= (image_shape[1] - 1)
        normalized_boxes[:, [1, 3]] /= (image_shape[0] - 1)

        # 2. Reorder to y1, x1, y2, x2
        normalized_boxes = from_numpy(normalized_boxes[:, [1, 0, 3, 2]]).detach().contiguous()

        # 3. Extract feature map crops
        image_ids = from_numpy(np.repeat(0, len(normalized_boxes)), dtype=np.int32).detach().contiguous()
        cropper = CropAndResizeFunction(self.roi_shape[1] * 2, self.roi_shape[2] * 2, 0)
        crops = cropper(x, normalized_boxes, image_ids)
        crops = torch.nn.functional.max_pool2d(crops, kernel_size=2)
        rcnn_logits, rcnn_deltas = self.rcnn(crops)

        # 4. Extract masks
        rcnn_masks = self.mask_head(crops)

        # Second NMS block
        # 1. Convert to numpy and to bboxes
        rcnn_detection_scores = softmax(rcnn_logits, dim=1)[:, [1]]
        rcnn_detections = construct_boxes(to_numpy(rcnn_deltas), rpn_proposals)

        # 2. Clip bboxes to image shape
        rcnn_detections[:, [0, 2]] = np.clip(rcnn_detections[:, [0, 2]], 0, image_shape[1] - 1)
        rcnn_detections[:, [1, 3]] = np.clip(rcnn_detections[:, [1, 3]], 0, image_shape[0] - 1)

        # 3. Convert to pytorch and perform actual NMS
        keep_indicies = non_max_suppression(from_numpy(rcnn_detections), rcnn_detection_scores, iou_threshold=0.3)
        rcnn_detections = rcnn_detections[keep_indicies]
        rcnn_detection_scores = rcnn_detection_scores[keep_indicies]
        rcnn_detection_masks = to_numpy(sigmoid(rcnn_masks[keep_indicies][:, 0]))

        return MaskRCNNOutputs(
            rpn_logits, rpn_deltas, rpn_proposals,
            anchors, rcnn_logits, rcnn_deltas, rcnn_masks,
            rcnn_detections, rcnn_detection_masks, rcnn_detection_scores,
            image_shape
        )
