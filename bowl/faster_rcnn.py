from itertools import product

from fire import Fire
import numpy as np
import torch
import torchvision
from torch.nn.functional import cross_entropy
from torch.nn.functional import smooth_l1_loss
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm

from bowl.rpn import RPN
from bowl.utils import iou
from bowl.utils import construct_deltas
from bowl.utils import from_numpy
from bowl.utils import to_numpy
from bowl.toy_shapes import generate_segmentation_batch

class VGGBackbone(torch.nn.Module):
    def __init__(self):
        super(VGGBackbone, self).__init__()
        self.cnn = torchvision.models.vgg16(pretrained=True)
        self.cnn.features = torch.nn.Sequential(*list(self.cnn.features.children())[:-1])
        for param in self.cnn.parameters(): param.requires_grad = False

    def forward(self, x):
        return self.cnn.features(x)

    def output_channels(self):
        return 512

    def stride(self):
        return 16

class ResnetBackbone(torch.nn.Module):
    def __init__(self):
        super(ResnetBackbone, self).__init__()
        self.cnn = torchvision.models.resnet18(pretrained=True)
        for param in self.cnn.parameters(): param.requires_grad = False

    def forward(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        # x = self.cnn.layer4(x)

        return x

    def output_channels(self):
        return 256

    def stride(self):
        return 16

class FasterRCNN(torch.nn.Module):
    def __init__(self, backbone, anchors_per_location):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = RPN(input_channels=self.backbone.output_channels(), anchors_per_location=anchors_per_location)

    def forward(self, x):
        x = self.backbone(x)
        logits, deltas = self.rpn(x)
        return logits, deltas

def fit():
    scales = [32, 64, 128]
    ratios = [1.0]
    backbone = ResnetBackbone()
    stride = backbone.stride()
    model = FasterRCNN(backbone, anchors_per_location=len(ratios) * len(scales))
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=0.001)
    step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14])

    num_epochs = 100
    num_batches = 10

    val_images, val_gt_boxes, _ = generate_segmentation_batch(1, (224, 256))
    image_shape = val_images[0].shape[1:]
    val_anchors = generate_anchors(stride, scales, ratios, image_shape)
    val_true_labels, val_label_weights, _ = generate_rpn_targets(val_anchors, val_gt_boxes[0], image_shape)

    for _ in tqdm(range(num_epochs)):
        for _ in tqdm(range(num_batches)):
            optimizer.zero_grad()
            images, gt_boxes, _ = generate_segmentation_batch(1, (224, 256))
            image_shape = images[0].shape[1:]
            anchors = generate_anchors(stride, scales, ratios, image_shape)
            true_labels, label_weights, _ = generate_rpn_targets(anchors, gt_boxes[0], image_shape)

            pred_labels, _ = model(from_numpy(images))
            loss = cross_entropy(pred_labels[0], from_numpy(true_labels, dtype=np.int64), weight=from_numpy(label_weights), ignore_index=-1)

            loss.backward()
            optimizer.step()
        step_lr.step()

        pred_labels, _ = model(from_numpy(val_images))
        loss = cross_entropy(pred_labels[0], from_numpy(val_true_labels, dtype=np.int64), weight=from_numpy(val_label_weights), ignore_index=-1)
        loss = loss.data[0]
        tqdm.write(f'val loss - {loss:.5f}')

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
    positives = np.argwhere(labels == 1)[0]
    negatives = np.argwhere(labels == 0)[0]

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
            batch_size / len(np.argwhere(labels == 1)[0]),
            batch_size / len(np.argwhere(labels == 0)[0])
        ])
    else:
        label_weights = np.array([1.0, 1.0])

    return labels, label_weights

import matplotlib.pyplot as plt
from matplotlib import patches
def display_boxes(boxes):
    boxes = np.clip(boxes, 0, 223).astype(np.int32)
    plt.cla()
    plt.imshow(np.zeros((224, 224)))
    _, ax = plt.gcf(), plt.gca()
    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.pause(1e-7)

if __name__ == '__main__':
    Fire()
