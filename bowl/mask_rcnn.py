import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
from fire import Fire
from torchvision.models import resnet18

from roi_align.crop_and_resize import CropAndResizeFunction
from bowl.toy_shapes import generate_segmentation_batch
from bowl.utils import construct_boxes
from bowl.utils import construct_deltas
from bowl.utils import generate_anchor_grid
from bowl.utils import iou
from bowl.utils import display_image_and_boxes
from bowl.utils import from_numpy
from bowl.utils import to_numpy
from bowl.utils import as_cuda

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
    def __init__(self):
        super(MaskRCNN, self).__init__()

        self.backbone = as_cuda(Backbone())

        for param in self.backbone.cnn.parameters():
            param.requires_grad = False

        self.base = 16
        self.scales = [2]
        self.ratios = [1.0]
        self.anchors_per_location = len(self.scales) * len(self.ratios)
        self.image_height = 224
        self.image_width = 224
        self.anchor_grid_shape = (self.image_height // self.base, self.image_width // self.base)
        self.anchor_grid = generate_anchor_grid(base=self.base, scales=self.scales, ratios=self.ratios, grid_shape=self.anchor_grid_shape)

        # TODO AS: 32 (instead of 512) is enough for toy problem. Need to go higher for more complex tasks
        self.rpn_conv = nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.box_classifier = nn.Conv2d(32, 2 * self.anchors_per_location, kernel_size=1, stride=1, padding=0, bias=True)
        self.box_regressor = nn.Conv2d(32, 4 * self.anchors_per_location, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.crop_and_resize = CropAndResizeFunction(7, 7)

        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid()
        )

    def rpn_forward(self, cnn_map):
        rpn_map = self.rpn_conv(cnn_map)
        rpn_map = self.relu(rpn_map)
        box_scores = self.box_classifier(rpn_map)
        box_deltas = self.box_regressor(rpn_map)
        # Since scores were received from 1x1 conv, order is important here
        # Order of anchors and scores should be exactly the same
        # Otherwise, network will never converge
        box_scores = box_scores.permute(0, 3, 2, 1).contiguous()
        box_scores = box_scores.view(box_scores.shape[0], box_scores.shape[1], box_scores.shape[2], self.anchors_per_location, 2)
        box_scores = box_scores.view(box_scores.shape[0], -1, 2)

        box_deltas = box_deltas.permute(0, 3, 2, 1).contiguous()
        box_deltas = box_deltas.view(box_deltas.shape[0], box_deltas.shape[1], box_deltas.shape[2], self.anchors_per_location, 4)
        box_deltas = box_deltas.view(box_deltas.shape[0], -1, 4)
        return box_scores, box_deltas

    def mask_head_forward(self, x, cnn_map, box_deltas):
        boxes = []
        box_indicies = []
        for i in range(box_deltas.shape[0]):
            image_boxes = construct_boxes(to_numpy(box_deltas.view(box_deltas.shape[0], -1, 4)[i]), self.anchor_grid.reshape(-1, 4))

            new_boxes = np.column_stack([
                image_boxes[:, 1],
                image_boxes[:, 0],
                image_boxes[:, 3],
                image_boxes[:, 2]
            ])

            boxes.extend(new_boxes)
            box_indicies.extend([i] * len(image_boxes))

        boxes = np.clip(np.array(boxes), 0, 223).astype(np.int32) / 223
        boxes = np.array(boxes).astype(np.float32)
        box_indicies = np.array(box_indicies).astype(np.int32)

        boxes = from_numpy(boxes)
        box_indicies = from_numpy(box_indicies, dtype=np.int32)

        crops = self.crop_and_resize(cnn_map, boxes, box_indicies)
        masks = self.mask_head(crops)
        return masks.view(box_deltas.shape[0], box_deltas.shape[1], 2, 14, 14)

    def forward(self, x):
        cnn_map = self.backbone(x)
        box_scores, box_deltas = self.rpn_forward(cnn_map)
        masks = self.mask_head_forward(x, cnn_map, box_deltas)
        return box_scores, box_deltas, self.anchor_grid.reshape(-1, 4), masks

def as_labels_and_gt_indicies(anchors, gt_boxes, include_min=True, threshold=0.7):
    ious = iou(anchors, gt_boxes)
    labels = np.full(ious.shape[0], -1)
    labels[np.any(ious > threshold, axis=1)] = 1
    if include_min:
        labels[np.argmax(ious, axis=0)] = 1
    # Part of negative samples should be discarded
    labels[np.all(ious < 0.3, axis=1) & (labels != 1)] = 2
    all_negatives = np.argwhere(labels == 2).reshape(-1)
    # Critical hyparameter
    # Too few negatives will slow down convergence
    # Too many negatives will dominate loss function
    max_negatives = 25 - len(np.argwhere(labels == 1))
    negative_samples = np.random.choice(all_negatives, min(max_negatives, len(all_negatives)) , replace=False)
    labels[negative_samples] = 0
    labels[labels == 2] = -1
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

def mask_loss(deltas, anchors, masks, gt_boxes, gt_masks):
    crop_and_resize = CropAndResizeFunction(14, 14)
    all_pr_masks = []
    all_gt_masks = []

    for i in range(deltas.shape[0]):
        image_pr_boxes = construct_boxes(to_numpy(deltas[i]), anchors)
        labels, gt_indicies = as_labels_and_gt_indicies(image_pr_boxes, gt_boxes[i], include_min=False, threshold=0.5)
        positive = np.where(labels == 1)

        if len(positive[0]) == 0:
            continue

        image_pr_boxes = image_pr_boxes[positive]
        image_gt_masks = gt_masks[i][gt_indicies[positive]][:, np.newaxis, :, :]
        image_gt_masks = crop_and_resize(
            from_numpy(image_gt_masks),
            from_numpy(np.clip(np.column_stack([
                image_pr_boxes[:, 1],
                image_pr_boxes[:, 0],
                image_pr_boxes[:, 3],
                image_pr_boxes[:, 2]
            ]) / 223, 0, 1)),
            from_numpy(np.arange(len(image_gt_masks)), np.int32)
        )
        image_gt_masks = image_gt_masks[:, 0, :, :]
        image_pr_masks = masks[i][positive][:, 0, :, :]

        all_pr_masks.extend(image_pr_masks)
        all_gt_masks.extend(image_gt_masks)

    if len(all_pr_masks) == 0:
        return from_numpy(np.array([0]))

    return F.binary_cross_entropy_with_logits(torch.cat(all_pr_masks), torch.cat(all_gt_masks))

def fit(train_size=100, validation_size=10, batch_size=8, num_epochs=100):
    net = as_cuda(MaskRCNN())
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0001)
    reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True, threshold=0.0001, cooldown=0, min_lr=1e-06, eps=1e-08
    )

    validation_images, validation_gt_boxes, validation_masks = generate_segmentation_batch(validation_size)
    train_images, train_gt_boxes, train_masks = generate_segmentation_batch(train_size)
    validation_images = from_numpy(validation_images.astype(np.float32))
    num_batches = len(train_images) // batch_size

    for epoch in tqdm(range(num_epochs)):
        indicies = np.random.choice(range(len(train_images)), len(train_images), replace=False)

        training_loss = 0.0
        training_cls_loss = 0.0
        training_reg_loss = 0.0
        training_mask_loss = 0.0
        for i in tqdm(range(num_batches)):
            batch_indicies = indicies[i * batch_size:i * batch_size + batch_size]
            image_batch, gt_boxes_batch, gt_masks_batch = train_images[batch_indicies], train_gt_boxes[batch_indicies], train_masks[batch_indicies]
            image_batch = from_numpy(image_batch.astype(np.float32))

            optimizer.zero_grad()
            box_scores, box_deltas, anchors, predicted_masks = net(image_batch)

            cls_loss = rpn_classifier_loss(gt_boxes_batch, box_scores, anchors)
            reg_loss = rpn_regressor_loss(gt_boxes_batch, box_deltas, anchors)
            _mask_loss = mask_loss(box_deltas, anchors, predicted_masks, gt_boxes_batch, gt_masks_batch)

            combined_loss = cls_loss + reg_loss + _mask_loss
            combined_loss.backward()
            optimizer.step()
            training_cls_loss += cls_loss.data[0] / num_batches
            training_reg_loss += reg_loss.data[0] / num_batches
            training_mask_loss += _mask_loss.data[0] / num_batches
            training_loss += combined_loss.data[0] / num_batches

        validation_scores, validation_deltas, validation_anchors, validation_predicted_masks = net(validation_images)
        fg_scores = to_numpy(validation_scores[0][:, 1])
        top_prediction_indicies = np.argsort(fg_scores)[::-1]
        predicted_boxes = anchors[top_prediction_indicies[:5]]
        predicted_deltas = to_numpy(validation_deltas[0])[top_prediction_indicies[:5]]
        predicted_masks = to_numpy(validation_predicted_masks[0])[top_prediction_indicies[:5]]

        actual_boxes = construct_boxes(predicted_deltas, predicted_boxes)

        img = to_numpy(validation_images[0])
        img = (img - img.min()) / (img.max() - img.min())
        display_image_and_boxes(img, actual_boxes, predicted_masks)

        validation_cls_loss = rpn_classifier_loss(validation_gt_boxes, validation_scores, validation_anchors)
        validation_reg_loss = rpn_regressor_loss(validation_gt_boxes, validation_deltas, validation_anchors)
        validation_mask_loss = mask_loss(validation_deltas, validation_anchors, validation_predicted_masks, validation_gt_boxes, validation_masks)
        total_validation_loss = validation_cls_loss.data[0] + validation_reg_loss.data[0] + validation_mask_loss.data[0]
        reduce_lr.step(total_validation_loss)
        tqdm.write(f'epoch: {epoch} - val reg: {validation_reg_loss.data[0]:.5f} - val cls: {validation_cls_loss.data[0]:.5f} - val mask: {validation_mask_loss.data[0]:.5f} - train reg: {training_reg_loss:.5f} - train cls: {training_cls_loss:.5f} - train mask: {training_mask_loss:.5f}')

def prof():
    import profile
    stats = profile.run('fit()')
    import pdb; pdb.set_trace()
if __name__ == '__main__':
    Fire()
