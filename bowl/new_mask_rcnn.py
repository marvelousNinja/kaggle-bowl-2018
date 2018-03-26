import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
from fire import Fire
from torchvision.models import resnet18
from roi_align.crop_and_resize import CropAndResizeFunction

from bowl.pipelines import pipeline
from bowl.pipelines import get_train_image_ids
from bowl.pipelines import get_validation_image_ids
from bowl.toy_shapes import generate_segmentation_batch
from bowl.utils import construct_boxes
from bowl.utils import construct_deltas
from bowl.utils import generate_anchor_grid
from bowl.utils import iou
from bowl.utils import display_image_and_boxes
from bowl.utils import from_numpy
from bowl.utils import to_numpy
from bowl.utils import as_cuda

def nms(deltas, scores, anchors, score_threshold=0.5, iou_threshold=0.3):
    bbox_indicies = []
    img_indicies = []

    for i in range(deltas.shape[0]):
        image_bboxes = construct_boxes(to_numpy(deltas[i]), anchors)
        image_scores = to_numpy(scores[i])

        order = np.argsort(image_scores).reshape(-1)[::-1]
        keep = []
        ious = iou(image_bboxes, image_bboxes)

        while len(order) > 1:
            top_id = order[0]
            keep.append(top_id)
            under_threshold = np.argwhere(ious[:, top_id] <= iou_threshold).reshape(-1)
            order = order[np.isin(order, under_threshold)]

        positive = np.argwhere(image_scores > score_threshold).reshape(-1)
        keep = list(set(keep) & set(positive))
        bbox_indicies.extend(keep)
        img_indicies.extend([i] * len(keep))

    return np.array(bbox_indicies), np.array(img_indicies)

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.cnn = resnet18(pretrained=True)

    def forward(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        # TODO AS: Seems to work better without it
        # x = self.cnn.layer4(x)
        return x

class MaskRCNN(nn.Module):
    def __init__(self, enable_masks=True):
        super(MaskRCNN, self).__init__()

        self.backbone = as_cuda(Backbone())

        # for param in self.backbone.cnn.parameters():
            # param.requires_grad = False

        self.base = 16
        self.scales = [2]
        self.ratios = [1.0]
        self.anchors_per_location = len(self.scales) * len(self.ratios)
        self.image_height = 224
        self.image_width = 224
        self.anchor_grid_shape = (self.image_height // self.base, self.image_width // self.base)
        self.anchor_grid = generate_anchor_grid(base=self.base, scales=self.scales, ratios=self.ratios, grid_shape=self.anchor_grid_shape)

        # TODO AS: 32 (instead of 512) is enough for toy problem. Need to go higher for more complex tasks
        self.rpn_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.box_classifier = nn.Conv2d(256, 2 * self.anchors_per_location, kernel_size=1, stride=1, padding=0, bias=True)
        self.box_regressor = nn.Conv2d(256, 4 * self.anchors_per_location, kernel_size=1, stride=1, padding=0, bias=True)

    def rpn_forward(self, cnn_map):
        rpn_map = self.rpn_conv(cnn_map)
        box_scores = self.box_classifier(rpn_map)
        box_deltas = self.box_regressor(rpn_map)
        # Since scores were received from 1x1 conv, order is important here
        # Order of anchors and scores should be exactly the same
        # Otherwise, network will never converge
        box_scores = box_scores.permute(0, 2, 3, 1).contiguous().view(box_scores.shape[0], -1, 2)
        box_deltas = box_deltas.permute(0, 2, 3, 1).contiguous().view(box_deltas.shape[0], -1, 4)
        return box_scores, box_deltas

    def forward(self, x):
        outputs = {}
        cnn_map = self.backbone(x)
        box_scores, box_deltas = self.rpn_forward(cnn_map)
        outputs['box_scores'] = box_scores
        outputs['anchors'] = self.anchor_grid.reshape(-1, 4)
        outputs['box_deltas'] =  box_deltas
        keep_bbox_indicies, keep_image_indicies = nms(box_deltas, F.softmax(box_scores, dim=2)[:, :, 1], self.anchor_grid.reshape(-1, 4))
        outputs['keep_bbox_indicies'] = keep_bbox_indicies
        outputs['keep_image_indicies'] = keep_image_indicies
        return outputs

def as_labels_and_gt_indicies(anchors, gt_boxes, threshold=0.7):
    batch_size = 256
    ious = iou(anchors, gt_boxes)
    labels = np.full(ious.shape[0], -1)

    positive = np.unique(np.concatenate([
        np.argwhere(np.any(ious > threshold, axis=1)).reshape(-1),
        np.argmax(ious, axis=0).reshape(-1)
    ]))

    labels[np.random.choice(positive, min(len(positive), int(batch_size / 2)), replace=False)] = 1
    negative = np.argwhere(np.all(ious < 0.3, axis=1) & (labels != 1)).reshape(-1)
    labels[np.random.choice(negative, min(len(negative), batch_size - min(len(positive), int(batch_size / 2))), replace=False)] = 0
    gt_indicies = np.argmax(ious, axis=1)
    return labels, gt_indicies

def rpn_classifier_loss(gt_boxes, box_scores, anchors):
    all_labels = []
    for image_gt_boxes in gt_boxes:
        labels, _ = as_labels_and_gt_indicies(anchors, image_gt_boxes)
        all_labels.extend(labels)
    return F.cross_entropy(box_scores.view(-1, 2), from_numpy(np.array(all_labels), dtype=np.int64), ignore_index=-1)

def rpn_regressor_loss(gt_boxes, box_deltas, anchors):
    all_gt_deltas = []
    all_pr_deltas = []

    for i in range(box_deltas.shape[0]):
        labels, gt_indicies = as_labels_and_gt_indicies(anchors, gt_boxes[i])
        positive = np.where(labels == 1)

        pr_deltas = box_deltas[i][positive]
        gt_deltas = construct_deltas(gt_boxes[i][gt_indicies[positive]], anchors[positive])

        all_gt_deltas.extend(gt_deltas)
        all_pr_deltas.extend(pr_deltas)

    return F.smooth_l1_loss(torch.stack(all_pr_deltas), from_numpy(np.array(all_gt_deltas)))

def fit(train_size=20, validation_size=1, batch_size=1, num_epochs=20, overfit=False):
    net = as_cuda(MaskRCNN())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, weight_decay=0.0005)
    reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    train_ids = get_train_image_ids()[:train_size]

    if overfit:
        validation_ids = train_ids[:validation_size]
    else:
        validation_ids = get_validation_image_ids()[:validation_size]

    validation_images, validation_gt_boxes, validation_gt_masks = map(np.array, zip(*map(pipeline, validation_ids)))

    validation_images = from_numpy(validation_images.astype(np.float32))

    num_batches = len(train_ids) // batch_size

    for epoch in tqdm(range(num_epochs)):
        indicies = np.random.choice(train_ids, len(train_ids), replace=False)

        training_loss = 0.0
        for i in tqdm(range(num_batches)):
            batch_indicies = indicies[i * batch_size:i * batch_size + batch_size]
            image_batch, gt_boxes_batch, gt_masks_batch = map(np.array, zip(*map(pipeline, batch_indicies)))
            image_batch = from_numpy(image_batch.astype(np.float32))

            optimizer.zero_grad()
            outputs = net(image_batch)
            box_scores = outputs['box_scores']
            anchors = outputs['anchors']
            box_deltas = outputs['box_deltas']

            loss = rpn_classifier_loss(gt_boxes_batch, box_scores, anchors)
            loss += rpn_regressor_loss(gt_boxes_batch, box_deltas, anchors)
            loss.backward()
            optimizer.step()
            training_loss += loss.data[0] / num_batches

        validation_outputs = net(validation_images)
        validation_scores = validation_outputs['box_scores']
        validation_anchors = validation_outputs['anchors']
        validation_deltas = validation_outputs['box_deltas']
        validation_keep_bbox_indicies = validation_outputs['keep_bbox_indicies']
        validation_keep_image_indicies = validation_outputs['keep_image_indicies']

        # display_predictions(
        #     validation_images[0],
        #     validation_scores[0],
        #     validation_deltas[0],
        #     validation_keep_bbox_indicies[validation_keep_image_indicies == 0],
        #     validation_anchors,
        #     validation_gt_boxes[0],
        #     validation_gt_masks[0],
        # )

        validation_loss = rpn_classifier_loss(validation_gt_boxes, validation_scores, validation_anchors)
        validation_loss += rpn_regressor_loss(validation_gt_boxes, validation_deltas, validation_anchors)
        validation_loss = validation_loss.data[0]
        reduce_lr.step(validation_loss)

        tqdm.write(f'epoch: {epoch} - val: {validation_loss:.5f} - train: {training_loss:.5f}')

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_predictions(image, box_scores, box_deltas, keep_bbox_indicies, anchors, gt_boxes, gt_masks):
    image = np.clip(((to_numpy(image) + 0.5) * 255).astype(np.int), 0, 255)
    scores = to_numpy(F.softmax(box_scores, dim=1))
    deltas = to_numpy(box_deltas)
    shifted_boxes = construct_boxes(deltas, anchors).astype(np.int)

    # Image
    plt.subplot(231)
    plt.cla()
    _, ax = plt.gcf(), plt.gca()
    ax.imshow(np.moveaxis(image, 0, 2))

    # GT Boxes
    plt.subplot(232)
    plt.cla()
    _, ax = plt.gcf(), plt.gca()
    ax.imshow(np.moveaxis(image, 0, 2))
    for (x1, y1, x2, y2) in gt_boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect)

    # GT Masks
    plt.subplot(233)
    plt.cla()
    _, ax = plt.gcf(), plt.gca()
    full = np.zeros(shape=image.shape[1:3])
    for mask in gt_masks:
        full = full + mask
    ax.imshow(full)

    # Positive anchors
    plt.subplot(234)
    plt.cla()
    _, ax = plt.gcf(), plt.gca()
    ax.imshow(np.moveaxis(image, 0, 2))

    for (x1, y1, x2, y2), score in zip(anchors, scores):
        if score[1] > 0.5:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    plt.pause(1e-17)

    # Positive anchors with shifts
    plt.subplot(235)
    plt.cla()
    _, ax = plt.gcf(), plt.gca()
    ax.imshow(np.moveaxis(image, 0, 2))

    for (x1, y1, x2, y2), score in zip(shifted_boxes, scores):
        if score[1] > 0.5:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    # Anchors after NMS
    plt.subplot(236)
    plt.cla()
    _, ax = plt.gcf(), plt.gca()
    ax.imshow(np.moveaxis(image, 0, 2))

    if len(keep_bbox_indicies) > 0:
        for (x1, y1, x2, y2) in shifted_boxes[keep_bbox_indicies]:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    plt.pause(1e-17)


def prof():
    import profile
    profile.run('fit()')
    import pdb; pdb.set_trace()
if __name__ == '__main__':
    Fire()
