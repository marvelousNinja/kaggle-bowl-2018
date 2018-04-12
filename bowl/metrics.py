import cv2
import numpy as np
import pycocotools.mask as mask_util

from bowl.utils import iou
from bowl.utils import to_numpy

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
