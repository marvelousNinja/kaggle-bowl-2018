from functools import partial

from fire import Fire
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.nn.functional import smooth_l1_loss
from torch.nn.functional import softmax
from tqdm import tqdm

from bowl.backbones import ResnetBackbone
from bowl.rpns import RPN
from bowl.roi_heads import RoIHead
from bowl.toy_shapes import generate_segmentation_batch
from bowl.utils import iou
from bowl.utils import construct_deltas
from bowl.utils import construct_boxes
from bowl.utils import from_numpy
from bowl.utils import to_numpy
from bowl.utils import non_max_suppression
from bowl.utils import generate_anchors
from roi_align.crop_and_resize import CropAndResizeFunction

class FasterRCNN(torch.nn.Module):
    def __init__(self, backbone, scales, ratios):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = RPN(input_channels=self.backbone.output_channels(), anchors_per_location=len(scales) * len(ratios))
        self.roi_shape = (self.backbone.output_channels(), 7, 7)
        self.rcnn = RoIHead(self.roi_shape, num_classes=1)
        self.generate_anchors = partial(generate_anchors, self.backbone.stride(), scales, ratios)

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
        cropper = CropAndResizeFunction(self.roi_shape[1], self.roi_shape[2], -1)
        crops = cropper(x, normalized_boxes, image_ids)
        rcnn_logits, rcnn_deltas = self.rcnn(crops)

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

        return rpn_logits, rpn_deltas, rpn_proposals, anchors, rcnn_logits, rcnn_deltas, rcnn_detections, rcnn_detection_scores

def fit():
    backbone = ResnetBackbone()
    model = FasterRCNN(backbone, scales=[16, 32, 64, 96], ratios=[1.0])
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=0.001)
    step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10])

    num_epochs = 100
    num_batches = 10

    val_images, val_gt_boxes, _ = generate_segmentation_batch(1, (448, 448))
    image_shape = val_images[0].shape[1:]
    val_anchors = model.generate_anchors(image_shape)
    val_true_labels, val_label_weights, val_true_deltas = generate_rpn_targets(val_anchors, val_gt_boxes[0], image_shape)

    for _ in tqdm(range(num_epochs)):
        for _ in tqdm(range(num_batches)):
            optimizer.zero_grad()
            images, gt_boxes, _ = generate_segmentation_batch(1, (448, 448))
            pred_labels, pred_deltas, pred_rpn_proposals, anchors, rcnn_logits, rcnn_deltas, _, _ = model(from_numpy(images))

            image_shape = images[0].shape[1:]
            true_labels, label_weights, true_deltas = generate_rpn_targets(anchors, gt_boxes[0], image_shape)

            cls_loss = cross_entropy(pred_labels[0], from_numpy(true_labels, dtype=np.int64), weight=from_numpy(label_weights), ignore_index=-1)
            reg_loss = smooth_l1_loss(pred_deltas[0][[np.where(true_labels == 1)[0]]], from_numpy(true_deltas[np.where(true_labels == 1)[0]]))

            rcnn_true_labels, rcnn_label_weights, rcnn_true_deltas = generate_rcnn_targets(pred_rpn_proposals, gt_boxes[0])
            rcnn_cls_loss = cross_entropy(rcnn_logits, from_numpy(rcnn_true_labels, dtype=np.int64), weight=from_numpy(rcnn_label_weights), ignore_index=-1)

            if len(np.where(rcnn_true_labels == 1)[0]) > 0:
                rcnn_reg_loss = smooth_l1_loss(rcnn_deltas[[np.where(rcnn_true_labels == 1)[0]]], from_numpy(rcnn_true_deltas[np.where(rcnn_true_labels == 1)[0]]))
            else:
                rcnn_reg_loss = from_numpy(np.array([0]))

            # TODO AS: Accumulate
            # tqdm.write(f'rpn cls {cls_loss.data[0]:.5f} - rpn reg {reg_loss.data[0]:.5f} - rcnn cls {rcnn_cls_loss.data[0]:.5f} - rcnn reg {rcnn_reg_loss.data[0]:.5f}')
            loss = cls_loss + reg_loss + rcnn_cls_loss + rcnn_reg_loss
            loss.backward()
            optimizer.step()
        step_lr.step()

        pred_labels, pred_deltas, _, anchors, _, _, detections, detection_scores = model(from_numpy(val_images))
        cls_loss = cross_entropy(pred_labels[0], from_numpy(val_true_labels, dtype=np.int64), weight=from_numpy(val_label_weights), ignore_index=-1)
        reg_loss = smooth_l1_loss(pred_deltas[0][[np.where(val_true_labels == 1)[0]]], from_numpy(val_true_deltas[np.where(val_true_labels == 1)[0]]))
        tqdm.write(f'val cls {cls_loss.data[0]:.5f} - val reg {reg_loss.data[0]:.5f}')
        display_boxes(detections, to_numpy(detection_scores), np.moveaxis(val_images[0], 0, 2))

def generate_rpn_targets(anchors, gt_boxes, image_shape, batch_size=256, balanced=False, positive_threshold=0.5, negative_threshold=0.3):
    ious = iou(anchors, gt_boxes)
    # For each anchor row, find maximum along columns
    gt_indicies = np.argmax(ious, axis=1)

    # For each anchor row, actual values of maximum IoU
    max_iou_per_anchor = ious[range(len(anchors)), gt_indicies]

    # For each gt box, anchor with the highest IoU (including ties)
    max_iou_per_gt_box = np.max(ious, axis=0)
    anchors_with_max_iou = np.where(ious == max_iou_per_gt_box)[0]

    # Anchors what cross image boundary
    outside_image = np.where(~(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] < image_shape[1]) &
        (anchors[:, 3] < image_shape[0])
    ))[0]

    # Negative: 0, Positive: 1, Neutral: -1
    neutral = -1,
    positive = 1
    negative = 0

    labels = np.repeat(neutral, len(anchors))
    labels[max_iou_per_anchor < negative_threshold] = negative
    labels[anchors_with_max_iou] = positive
    labels[max_iou_per_anchor >= positive_threshold] = positive
    labels[outside_image] = neutral

    deltas = construct_deltas(gt_boxes[gt_indicies], anchors)
    labels, label_weights = sample_batch(labels, batch_size, balanced)
    return labels, label_weights, deltas

def sample_batch(labels, batch_size, balanced):
    positives = np.argwhere(labels == 1).reshape(-1)
    negatives = np.argwhere(labels == 0).reshape(-1)

    allowed_positives = batch_size // 2
    actual_positives = len(positives)
    extra_positives = max(0, actual_positives - allowed_positives)

    allowed_negatives = batch_size - (actual_positives - extra_positives)
    actual_negatives = len(negatives)
    extra_negatives = max(0, actual_negatives - allowed_negatives)

    if extra_positives > 0:
        labels[np.random.choice(positives, extra_positives, replace=False)] = -1

    if extra_negatives > 0:
        labels[np.random.choice(negatives, extra_negatives, replace=False)] = -1

    if balanced:
        label_weights = np.array([
            batch_size / len(np.argwhere(labels == 1).reshape(-1)),
            batch_size / len(np.argwhere(labels == 0).reshape(-1))
        ])
    else:
        label_weights = np.array([1.0, 1.0])

    return labels, label_weights

def generate_rcnn_targets(boxes, gt_boxes, batch_size=128, balanced=False, positive_threshold=0.5, min_negative_threshold=0.0):
    ious = iou(boxes, gt_boxes)
    # For each box row, find maximum along columns
    gt_indicies = np.argmax(ious, axis=1)

    # For each box row, actual values of maximum IoU
    max_iou_per_anchor = ious[range(len(boxes)), gt_indicies]

    # Negative: 0, Positive: 1, Neutral: -1
    neutral = -1,
    positive = 1
    negative = 0

    labels = np.repeat(neutral, len(boxes))
    labels[(max_iou_per_anchor < positive_threshold) & (max_iou_per_anchor >= min_negative_threshold)] = negative
    labels[max_iou_per_anchor >= positive_threshold] = positive

    deltas = construct_deltas(gt_boxes[gt_indicies], boxes)
    labels, label_weights = sample_rcnn_batch(labels, batch_size, balanced)
    return labels, label_weights, deltas

def sample_rcnn_batch(labels, batch_size, balanced):
    positives = np.argwhere(labels == 1).reshape(-1)
    negatives = np.argwhere(labels == 0).reshape(-1)

    allowed_positives = batch_size // 4
    actual_positives = len(positives)
    extra_positives = max(0, actual_positives - allowed_positives)

    allowed_negatives = batch_size - (actual_positives - extra_positives)
    actual_negatives = len(negatives)
    extra_negatives = max(0, actual_negatives - allowed_negatives)

    if extra_positives > 0:
        labels[np.random.choice(positives, extra_positives, replace=False)] = -1

    if extra_negatives > 0:
        labels[np.random.choice(negatives, extra_negatives, replace=False)] = -1

    if balanced:
        label_weights = np.array([
            batch_size / (len(np.argwhere(labels == 1).reshape(-1)) + 1),
            batch_size / len(np.argwhere(labels == 0).reshape(-1))
        ])
    else:
        label_weights = np.array([1.0, 1.0])

    return labels, label_weights

import matplotlib.pyplot as plt
from matplotlib import patches
def display_boxes(boxes, scores, bg):
    positives = np.where(scores > 0.5)[0]
    boxes = boxes[positives]
    boxes = boxes.astype(np.int32)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, bg.shape[1] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, bg.shape[0] - 1)
    plt.cla()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    plt.imshow((bg * std + mean) * 255)
    _, ax = plt.gcf(), plt.gca()
    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.pause(1e-7)

if __name__ == '__main__':
    Fire()
