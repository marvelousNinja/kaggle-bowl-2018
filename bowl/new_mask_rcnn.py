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
            nn.Sigmoid()
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
    total_loss = from_numpy(np.array([0]))
    for image_box_scores, image_gt_boxes in zip(box_scores, gt_boxes):
        labels, _ = as_labels_and_gt_indicies(anchors, image_gt_boxes)
        total_loss += F.cross_entropy(image_box_scores, from_numpy(labels, dtype=np.int64), ignore_index=-1)
    return total_loss / box_scores.shape[0]

def rpn_regressor_loss(gt_boxes, box_deltas, anchors):
    total_loss = from_numpy(np.array([0]))
    for image_box_deltas, image_gt_boxes in zip(box_deltas, gt_boxes):
        labels, indicies = as_labels_and_gt_indicies(anchors, image_gt_boxes)
        positive_samples = np.argwhere(labels == 1).reshape(-1)
        positive_indicies = indicies[positive_samples]
        positive_deltas = image_box_deltas[[positive_samples]]
        positive_gt_boxes = image_gt_boxes[positive_indicies]
        positive_anchors = anchors[positive_samples]
        true_deltas = construct_deltas(positive_gt_boxes, positive_anchors)
        total_loss += F.smooth_l1_loss(positive_deltas, from_numpy(true_deltas))
    return total_loss / box_deltas.shape[0]

def mask_loss(gt_boxes_batch, box_deltas, anchors, predicted_masks, gt_masks):
    total_loss = from_numpy(np.array([0]))
    total_masks_with_loss = 0

    crop_and_resize = CropAndResizeFunction(14, 14)
    crop_and_resize.requires_grad = False

    for image_box_deltas, image_gt_boxes, image_predicted_masks, image_gt_masks in zip(box_deltas, gt_boxes_batch, predicted_masks, gt_masks):
        predicted_boxes = construct_boxes(to_numpy(image_box_deltas), anchors)
        labels, indicies = as_labels_and_gt_indicies(predicted_boxes, image_gt_boxes, include_min=False, threshold=0.5)
        positive_samples = np.argwhere(labels == 1).reshape(-1)

        if len(positive_samples) == 0:
            continue

        positive_indicies = indicies[positive_samples]
        positive_deltas = image_box_deltas[[positive_samples]]
        positive_gt_boxes = image_gt_boxes[positive_indicies]
        positive_gt_masks = image_gt_masks[positive_indicies]
        positive_anchors = anchors[positive_samples]
        predicted_boxes = construct_boxes(to_numpy(positive_deltas), positive_anchors)
        predicted_boxes = np.clip(predicted_boxes, 0, 223).astype(np.int32)
        positive_predicted_masks = image_predicted_masks[[positive_samples]]

        mask_bboxes = []
        mask_indicies = []
        for (x1, y1, x2, y2), one_mask in zip(predicted_boxes, positive_gt_masks):
            mask_bboxes.append([y1, x1, y2, x2])
            mask_indicies.append(len(mask_bboxes) - 1)


        mask_bboxes = (np.array(mask_bboxes) / 223).astype(np.float32)
        mask_bboxes = from_numpy(mask_bboxes)
        mask_indicies = from_numpy(np.array(mask_indicies), dtype=np.int32)
        positive_gt_masks = positive_gt_masks[:, np.newaxis, :, :]
        positive_gt_masks = from_numpy(positive_gt_masks)
        target_masks = crop_and_resize(positive_gt_masks, mask_bboxes, mask_indicies).round()
        target_masks = from_numpy(to_numpy(target_masks.detach()))

        for predicted_mask, target_mask in zip(positive_predicted_masks, target_masks):
            if (target_mask.shape[0] == 0) or (target_mask.shape[1] == 0):
                continue

            predicted_mask = predicted_mask[0]
            target_mask = target_mask[0]
            total_loss += F.binary_cross_entropy(predicted_mask, target_mask)
            total_masks_with_loss += 1

    return total_loss / (total_masks_with_loss + 1.0)

def fit(train_size=100, validation_size=10, batch_size=8, num_epochs=100):
    net = as_cuda(MaskRCNN())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)

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
            _mask_loss = mask_loss(gt_boxes_batch, box_deltas, anchors, predicted_masks, gt_masks_batch)

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
        # display_image_and_boxes(img, actual_boxes, predicted_masks)

        validation_cls_loss = rpn_classifier_loss(validation_gt_boxes, validation_scores, validation_anchors)
        validation_reg_loss = rpn_regressor_loss(validation_gt_boxes, validation_deltas, validation_anchors)
        validation_mask_loss = mask_loss(validation_gt_boxes, validation_deltas, validation_anchors, validation_predicted_masks, validation_masks)
        tqdm.write(f'epoch: {epoch} - val reg: {validation_reg_loss.data[0]:.5f} - val cls: {validation_cls_loss.data[0]:.5f} - val mask: {validation_mask_loss.data[0]:.5f} - train reg: {training_reg_loss:.5f} - train cls: {training_cls_loss:.5f} - train mask: {training_mask_loss:.5f}')

def prof():
    import profile
    stats = profile.run('fit()')
    import pdb; pdb.set_trace()
if __name__ == '__main__':
    Fire()
