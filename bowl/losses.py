import numpy as np
from torch.nn.functional import l1_loss
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy

from bowl.utils import from_numpy
from bowl.utils import construct_deltas
from bowl.utils import construct_boxes
from bowl.utils import iou
from roi_align.crop_and_resize import CropAndResizeFunction

def compute_loss(outputs, gt):
    rpn_logits, rpn_deltas, rpn_proposals, anchors, rcnn_logits, rcnn_deltas, rcnn_masks, _, _, _, image_shape = outputs
    gt_boxes, gt_masks = gt

    rpn_true_labels, rpn_label_weights, rpn_true_deltas = generate_rpn_targets(anchors, gt_boxes[0], image_shape)
    rpn_cls_loss = cross_entropy(rpn_logits[0], rpn_true_labels, rpn_label_weights, ignore_index=-1)
    positive = (rpn_true_labels == 1).nonzero().view(-1)
    if len(positive) > 0:
        rpn_reg_loss = l1_loss(rpn_deltas[0][positive], rpn_true_deltas[positive])
    else:
        rpn_reg_loss = from_numpy(np.array([0]))

    rcnn_true_labels, rcnn_label_weights, rcnn_true_deltas = generate_rcnn_targets(rpn_proposals, gt_boxes[0])
    rcnn_cls_loss = cross_entropy(rcnn_logits, rcnn_true_labels, rcnn_label_weights, ignore_index=-1)
    positive = (rcnn_true_labels == 1).nonzero().view(-1)
    if len(positive) > 0:
        rcnn_reg_loss = l1_loss(rcnn_deltas[positive], rcnn_true_deltas[positive])
    else:
        rcnn_reg_loss = from_numpy(np.array([0]))

    rcnn_true_masks = generate_mask_targets(image_shape, rpn_proposals, gt_boxes, gt_masks)
    if len(positive) > 0:
        rcnn_mask_loss = binary_cross_entropy_with_logits(rcnn_masks[positive], rcnn_true_masks[positive])
    else:
        rcnn_mask_loss = from_numpy(np.array([0]))

    return sum((rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, rcnn_mask_loss))

def generate_rpn_targets(anchors, gt_boxes, image_shape, batch_size=256, balanced=False, positive_threshold=0.5, negative_threshold=0.3):
    ious = iou(anchors, gt_boxes)
    # For each anchor row, find maximum along columns
    gt_indicies = np.argmax(ious, axis=1)

    # For each anchor row, actual values of maximum IoU
    max_iou_per_anchor = ious[range(len(anchors)), gt_indicies]

    # For each gt box, anchor with the highest IoU (including ties)
    max_iou_per_gt_box = np.max(ious, axis=0)
    anchors_with_max_iou, gt_boxes_for_max_anchors = np.where(ious == max_iou_per_gt_box)

    # While anchor has max IoU for some GT box, it may overlap with other GT box better
    anchors_with_max_iou = anchors_with_max_iou[max_iou_per_anchor[anchors_with_max_iou] == max_iou_per_gt_box[gt_boxes_for_max_anchors]]

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
    return from_numpy(labels, dtype=np.int64), from_numpy(label_weights), from_numpy(deltas)

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
            batch_size / (len(np.argwhere(labels == 1).reshape(-1)) + 1),
            batch_size / (len(np.argwhere(labels == 0).reshape(-1)) + 1)
        ])
    else:
        label_weights = np.array([1.0, 1.0])

    return labels, label_weights

def generate_rcnn_targets(boxes, gt_boxes, batch_size=64, balanced=False, positive_threshold=0.5, min_negative_threshold=0.0):
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
    return from_numpy(labels, dtype=np.int64), from_numpy(label_weights), from_numpy(deltas)

def generate_mask_targets(image_shape, rpn_proposals, gt_boxes, gt_masks):
    gt_masks = gt_masks[0]
    gt_boxes = gt_boxes[0]

    ious = iou(rpn_proposals, gt_boxes)
    gt_indicies = np.argmax(ious, axis=1)

    # RoI Heads block
    # 1. Normalize boxes to 0..1 coordinates
    normalized_boxes = rpn_proposals.copy()
    normalized_boxes[:, [0, 2]] /= (image_shape[1] - 1)
    normalized_boxes[:, [1, 3]] /= (image_shape[0] - 1)

    # 2. Reorder to y1, x1, y2, x2
    normalized_boxes = from_numpy(normalized_boxes[:, [1, 0, 3, 2]]).detach().contiguous()

    # 3. Extract target masks
    mask_ids = from_numpy(gt_indicies, dtype=np.int32).detach().contiguous()
    # TODO AS: RoI shape!
    cropper = CropAndResizeFunction(14, 14, 0)
    crops = cropper(from_numpy(gt_masks[:, None]), normalized_boxes, mask_ids)
    return crops

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
            batch_size / (len(np.argwhere(labels == 0).reshape(-1)) + 1)
        ])
    else:
        label_weights = np.array([1.0, 1.0])

    return labels, label_weights
