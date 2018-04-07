from itertools import product

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.autograd import Variable
from nms.pth_nms import pth_nms

def display_boxes(boxes, scores, bg):
    positives = np.where(scores > 0.5)[0]
    boxes = boxes[positives]
    boxes = boxes.astype(np.int32)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, bg.shape[1] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, bg.shape[0] - 1)
    plt.cla()
    plt.imshow(bg)
    _, ax = plt.gcf(), plt.gca()
    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.pause(1e-7)

def generate_anchors(stride, scales, ratios, image_shape):
    max_y_shift = image_shape[0] // stride
    max_x_shift = image_shape[1] // stride

    anchors = []
    for y_shift, x_shift, scale, ratio in product(range(max_y_shift), range(max_x_shift), scales, ratios):
        x_center = stride / 2 + x_shift * stride - 1
        y_center = stride / 2 + y_shift * stride - 1
        width = scale * ratio
        height = scale / ratio
        anchors.append((
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2
        ))

    return np.array(anchors, dtype=np.float32)

def non_max_suppression(boxes, scores, iou_threshold):
    return pth_nms(torch.cat((boxes, scores), dim=1).data, iou_threshold)

def normalize(image_batch):
    image_batch = image_batch.astype(np.float32)
    image_batch /= 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_batch[:, 0, :, :] -= mean[0]
    image_batch[:, 1, :, :] -= mean[1]
    image_batch[:, 2, :, :] -= mean[2]
    image_batch[:, 0, :, :] /= std[0]
    image_batch[:, 1, :, :] /= std[1]
    image_batch[:, 2, :, :] /= std[2]
    return image_batch

def construct_deltas(gt_boxes, anchors):
    w_a = anchors[:, 2] - anchors[:, 0] + 1
    h_a = anchors[:, 3] - anchors[:, 1] + 1
    w_gt = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
    h_gt = gt_boxes[:, 3] - gt_boxes[:, 1] + 1

    x_center_a = anchors[:, 0] + w_a * 0.5
    y_center_a = anchors[:, 1] + h_a * 0.5
    x_center_gt = gt_boxes[:, 0] + w_gt * 0.5
    y_center_gt = gt_boxes[:, 1] + h_gt * 0.5

    t_x = (x_center_gt - x_center_a) / w_a
    t_y = (y_center_gt - y_center_a) / h_a
    t_w = np.log(w_gt / w_a)
    t_h = np.log(h_gt / h_a)

    return np.column_stack((
        t_x,
        t_y,
        t_w,
        t_h
    )) / [0.1, 0.1, 0.2, 0.2]

def construct_boxes(deltas, anchors):
    deltas = deltas * [0.1, 0.1, 0.2, 0.2]
    t_x = deltas[:, 0]
    t_y = deltas[:, 1]
    t_w = deltas[:, 2]
    t_h = deltas[:, 3]

    w_a = anchors[:, 2] - anchors[:, 0] + 1
    h_a = anchors[:, 3] - anchors[:, 1] + 1
    x_center_a = anchors[:, 0] + w_a * 0.5
    y_center_a = anchors[:, 1] + h_a * 0.5

    w_gt = np.exp(t_w) * w_a
    h_gt = np.exp(t_h) * h_a

    x_center_gt = t_x * w_a + x_center_a
    y_center_gt = t_y * h_a + y_center_a

    x0 = x_center_gt - w_gt * 0.5
    y0 = y_center_gt - h_gt * 0.5
    x1 = x_center_gt + w_gt * 0.5 - 1
    y1 = y_center_gt + h_gt * 0.5 - 1

    return np.column_stack((
        x0,
        y0,
        x1,
        y1
    ))

def generate_anchor_grid(base, scales, ratios, grid_shape):
    anchors = []
    anchors = np.zeros((grid_shape[0], grid_shape[1], len(scales) * len(ratios), 4), dtype=np.float32)
    for y_diff in range(grid_shape[0]):
        for x_diff in range(grid_shape[0]):
            anchor_index = 0
            for scale in scales:
                for ratio in ratios:
                    # TODO AS: Replace with np.sqrt(ratio) to preserve area?
                    width = int(base / ratio * scale)
                    height = int(base * ratio * scale)
                    x_ctr = int(base / 2) + x_diff * base
                    y_ctr = int(base / 2) + y_diff * base
                    x1, y1 = x_ctr - width / 2, y_ctr - height / 2
                    x2, y2 = x_ctr + width / 2, y_ctr + height / 2

                    anchors[y_diff, x_diff, anchor_index] = [x1, y1, x2, y2]
                    anchor_index += 1

    return np.clip(anchors - 1, a_min=0, a_max=None)

def iou(bboxes_a, bboxes_b):
    tl = np.maximum(bboxes_a[:, None, :2], bboxes_b[:, :2])
    br = np.minimum(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], axis=1)
    area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def display_image_and_boxes(image, boxes, masks=None, gt_boxes=None, gt_masks=None):
    boxes = np.clip(boxes, 0, 223).astype(np.int32)

    plt.subplot(131)
    plt.cla()
    _, ax = plt.gcf(), plt.gca()
    ax.imshow(np.moveaxis(image, 0, 2))

    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    if masks is not None:
        plt.subplot(132)
        plt.cla()
        _, ax = plt.gcf(), plt.gca()
        full = np.zeros(shape=image.shape[1:3])
        for (x1, y1, x2, y2), mask in zip(boxes, masks):
            height = y2 - y1
            width = x2 - x1

            if height > 0 and width > 0:
                torch_mask = torch.nn.functional.sigmoid(Variable(torch.from_numpy(mask)))
                predicted_mask = torch.nn.functional.upsample(torch_mask.unsqueeze(dim=0), size=(int(height), int(width)), mode='bilinear')[0][0]
                predicted_mask = predicted_mask.round().data.cpu().numpy()
                full[y1:y2, x1:x2] = predicted_mask

        plt.imshow(full)
        for (x1, y1, x2, y2) in boxes:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    plt.subplot(133)
    plt.cla()
    full = np.zeros(shape=image.shape[1:3])
    _, ax = plt.gcf(), plt.gca()
    if gt_boxes is not None:
        for (x1, y1, x2, y2) in gt_boxes:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='g',facecolor='none')
            ax.add_patch(rect)

    if gt_masks is not None:
        for (x1, y1, x2, y2), mask in zip(gt_boxes, gt_masks):
            full = full + mask

    plt.imshow(full)
    plt.draw()
    plt.pause(1e-17)

def as_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    return variable

def from_numpy(obj, dtype=np.float32):
    variable = Variable(torch.from_numpy(obj.astype(dtype)))
    return as_cuda(variable)

def to_numpy(variable):
    return variable.data.cpu().numpy()
