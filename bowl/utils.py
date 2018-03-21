import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.autograd import Variable

def normalize(image_batch):
    image_batch = image_batch.astype(np.float32)
    image_batch /= 255
    mean = [0.08734627, 0.08734627, 0.08734627]
    std = [0.28179365, 0.28179365, 0.28179365]
    image_batch[:, 0, :, :] -= mean[0]
    image_batch[:, 1, :, :] -= mean[1]
    image_batch[:, 2, :, :] -= mean[2]
    image_batch[:, 0, :, :] /= std[0]
    image_batch[:, 1, :, :] /= std[1]
    image_batch[:, 2, :, :] /= std[2]
    return image_batch

def construct_deltas(gt_boxes, anchors):
    gt_boxes = gt_boxes.astype(np.float32)
    w_gt = gt_boxes[:, 2] - gt_boxes[:, 0]
    h_gt = gt_boxes[:, 3] - gt_boxes[:, 1]
    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    t_x = (gt_boxes[:, 0] - anchors[:, 0]) / w_a
    t_y = (gt_boxes[:, 1] - anchors[:, 1]) / h_a
    t_w = np.log(w_gt / w_a)
    t_h = np.log(h_gt / h_a)
    return np.column_stack((t_x, t_y, t_w, t_h)) / [0.3, 0.3, 0.3, 0.3] #[0.1, 0.1, 0.2, 0.2]

def construct_boxes(deltas, anchors):
    deltas = deltas * [0.3, 0.3, 0.3, 0.3]
    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    x_a = anchors[:, 0]
    y_a = anchors[:, 1]
    return np.column_stack((
        deltas[:, 0] * w_a + x_a,
        deltas[:, 1] * h_a + y_a,
        deltas[:, 0] * w_a + x_a + np.exp(deltas[:, 2]) * w_a,
        deltas[:, 1] * h_a + y_a + np.exp(deltas[:, 3]) * h_a
    ))

def generate_anchor_grid(base, scales, ratios, grid_shape):
    anchors = []
    anchors = np.zeros((grid_shape[0], grid_shape[1], len(scales) * len(ratios), 4), dtype=np.float32)
    for y_diff in range(grid_shape[0]):
        for x_diff in range(grid_shape[0]):
            for scale in scales:
                anchor_index = 0
                for ratio in ratios:
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

def display_image_and_boxes(image, boxes, masks=None):
    boxes = np.clip(boxes, 0, 223).astype(np.int32)

    plt.subplot(121)
    plt.cla()
    _, ax = plt.gcf(), plt.gca()
    ax.imshow(np.moveaxis(image, 0, 2))

    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    if masks is not None:
        plt.subplot(122)
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
