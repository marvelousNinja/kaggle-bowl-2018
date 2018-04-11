from functools import partial

import cv2
from fire import Fire
import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.nn.functional import cross_entropy
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import smooth_l1_loss
from torch.nn.functional import l1_loss
from torch.nn.functional import softmax
from torch.nn.functional import sigmoid
from tqdm import tqdm

from bowl.backbones import ResnetBackbone
from bowl.backbones import VGGBackbone
from bowl.generators import toy_shapes_generator
from bowl.generators import bowl_train_generator
from bowl.generators import bowl_validation_generator
from bowl.mask_heads import MaskHead
from bowl.rpns import RPN
from bowl.roi_heads import RoIHead
from bowl.model_checkpoint import ModelCheckpoint
from bowl.utils import as_cuda
from bowl.utils import iou
from bowl.utils import construct_deltas
from bowl.utils import construct_boxes
from bowl.utils import from_numpy
from bowl.utils import to_numpy
from bowl.utils import non_max_suppression
from bowl.utils import generate_anchors
from bowl.utils import display_boxes
from roi_align.crop_and_resize import CropAndResizeFunction

class FasterRCNN(torch.nn.Module):
    def __init__(self, backbone, scales, ratios):
        super(FasterRCNN, self).__init__()
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

        return rpn_logits, rpn_deltas, rpn_proposals, anchors, rcnn_logits, rcnn_deltas, rcnn_masks, rcnn_detections, rcnn_detection_masks, rcnn_detection_scores

def fit(
        scales=[32], image_shape=(224, 224), ratios=[1.0],
        trainable_backbone=False, lr=0.001, dataset='toy',
        num_epochs=10, num_batches=10, backbone='resnet',
        visualize=False
        ):

    np.random.seed(1991)

    if backbone == 'resnet':
        backbone = ResnetBackbone(trainable_backbone)
    else:
        backbone = VGGBackbone(trainable_backbone)

    model = as_cuda(FasterRCNN(backbone, scales, ratios))
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr)
    reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)
    model_checkpoint = ModelCheckpoint(model, 'faster-rcnn', logger=tqdm.write)

    if dataset == 'toy':
        train_generator = toy_shapes_generator(image_shape)
        validation_generator = toy_shapes_generator(image_shape)
    else:
        train_generator = bowl_train_generator(image_shape)
        validation_generator = bowl_validation_generator(image_shape)

    val_images, val_gt_boxes, val_gt_masks = next(validation_generator)

    for _ in tqdm(range(num_epochs)):
        for _ in tqdm(range(num_batches)):
            optimizer.zero_grad()
            images, gt_boxes, gt_masks = next(train_generator)
            outputs = model(from_numpy(images))
            losses = compute_loss(*outputs[:-3], images.shape[2:], gt_boxes, gt_masks)
            loss = sum(losses)
            loss.backward()
            optimizer.step()

        outputs = model.eval()(from_numpy(val_images))
        losses = compute_loss(*outputs[:-3], val_images.shape[2:], val_gt_boxes, val_gt_masks)
        print_losses(losses)
        mask_map = mask_mean_average_precision(outputs, val_gt_masks)
        bbox_map = box_mean_average_precision(outputs, val_gt_boxes)
        tqdm.write(f'bbox map {bbox_map:.5f} - mask map {mask_map:.5f}')
        loss = sum(losses)
        #reduce_lr.step(loss.data[0])
        model_checkpoint.step(mask_map)
        if visualize: display_boxes(outputs[-3], outputs[-2], to_numpy(outputs[-1]), np.moveaxis(val_images[0], 0, 2))

def print_losses(losses):
    rpn_cls, rpn_reg, rcnn_cls, rcnn_reg, rcnn_mask = map(lambda loss: loss.data[0], losses)
    total = rpn_cls + rpn_reg + rcnn_cls + rcnn_reg + rcnn_mask
    tqdm.write(f'total {total:.5f} - rpn cls {rpn_cls:.5f} - rpn reg {rpn_reg:.5f} - rcnn cls {rcnn_cls:.5f} - rcnn reg {rcnn_reg:.5f} - rcnn mask {rcnn_mask:.5f}')

def box_mean_average_precision(outputs, gt_boxes):
    detections = outputs[-3]
    scores = to_numpy(outputs[-1].view(-1))
    detections = detections[scores > 0.7]
    gt_boxes = gt_boxes[0]

    ious = iou(detections, gt_boxes)

    num_positives = 0
    num_negatives = 0

    for threshold in np.arange(0.5, 1, 0.05):
        ious = iou(detections, gt_boxes)

        while True:
            if ious.shape[0] == 0:
                num_negatives += ious.shape[1]
                break

            if ious.shape[1] == 0:
                num_negatives += ious.shape[0]
                break

            max_iou_per_detection = np.max(ious, axis=1)
            top_detection = np.argmax(max_iou_per_detection)
            top_box = np.argmax(ious[top_detection])
            max_iou = max_iou_per_detection[top_detection]

            if max_iou > threshold:
                num_positives += 1
            else:
                num_negatives += 1

            ious = np.delete(ious, top_detection, axis=0)
            ious = np.delete(ious, top_box, axis=1)

    value = num_positives / (num_positives + num_negatives)
    return value

def mask_mean_average_precision(outputs, gt_masks):
    pred_masks = outputs[-2]
    pred_boxes = outputs[-3]
    scores = to_numpy(outputs[-1].view(-1))
    pred_masks = pred_masks[scores > 0.7]
    pred_boxes = pred_boxes[scores > 0.7].astype(np.int)
    gt_masks = gt_masks[0]

    if len(pred_masks) == 0:
        return 0

    expanded_masks = np.zeros((len(pred_masks), gt_masks.shape[1], gt_masks.shape[2]))
    for i, (pred_mask, (x1, y1, x2, y2)) in enumerate(zip(pred_masks, pred_boxes)):
        resized_mask = cv2.resize(pred_mask, (x2 - x1 + 1, y2 - y1 + 1), interpolation=cv2.INTER_CUBIC)
        expanded_masks[i, y1:y2 + 1, x1:x2 + 1] = resized_mask.round()

    gt_masks = mask_util.encode(np.asfortranarray(np.moveaxis(gt_masks, 0, 2).astype(np.uint8)))
    pred_masks = mask_util.encode(np.asfortranarray(np.moveaxis(expanded_masks, 0, 2).astype(np.uint8)))

    num_positives = 0
    num_negatives = 0

    for threshold in np.arange(0.5, 1, 0.05):
        ious = mask_util.iou(pred_masks, gt_masks, np.zeros(len(pred_masks)))

        while True:
            if ious.shape[0] == 0:
                num_negatives += ious.shape[1]
                break

            if ious.shape[1] == 0:
                num_negatives += ious.shape[0]
                break

            max_iou_per_detection = np.max(ious, axis=1)
            top_detection = np.argmax(max_iou_per_detection)
            top_box = np.argmax(ious[top_detection])
            max_iou = max_iou_per_detection[top_detection]

            if max_iou > threshold:
                num_positives += 1
            else:
                num_negatives += 1

            ious = np.delete(ious, top_detection, axis=0)
            ious = np.delete(ious, top_box, axis=1)

    value = num_positives / (num_positives + num_negatives)
    return value

def compute_loss(rpn_logits, rpn_deltas, rpn_proposals, anchors, rcnn_logits, rcnn_deltas, rcnn_masks, image_shape, gt_boxes, gt_masks):
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

    return rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, rcnn_mask_loss

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

def prof():
    import profile
    import pstats
    profile.run('fit()', 'fit.profile')
    stats = pstats.Stats('fit.profile')
    stats.sort_stats('cumulative').print_stats(30)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire()
