import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
from fire import Fire
from torchvision.models import resnet18
import cv2

from roi_align.crop_and_resize import CropAndResizeFunction
from bowl.roi_pooling import roi_pooling
from bowl.utils import generate_segmentation_image

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

class MaskRCNN(nn.Module):
    def __init__(self):
        super(MaskRCNN, self).__init__()

        self.backbone = cuda_pls(Backbone())

        # TODO AS: Extract as a param
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.base = 16
        self.scales = [2] #, 2, 4]
        self.ratios = [1.0]
        self.anchors_per_location = len(self.scales) * len(self.ratios)
        self.image_height = 224
        self.image_width = 224
        self.anchor_grid_shape = (self.image_height // self.base, self.image_width // self.base)
        self.anchor_grid = self.generate_anchor_grid(base=self.base, scales=self.scales, ratios=self.ratios, grid_shape=self.anchor_grid_shape)

        # TODO AS: 32 (instead of 512) is enough for toy problem. Need to go higher for more complex tasks
        self.rpn_conv = nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.box_classifier = nn.Conv2d(32, 2 * self.anchors_per_location, kernel_size=1, stride=1, padding=0, bias=True)
        self.box_regressor = nn.Conv2d(32, 4 * self.anchors_per_location, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.crop_and_resize = CropAndResizeFunction(7, 7)

        self.mask_conv_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask_conv_3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False)
        self.mask_final = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=False)

    def generate_anchor_grid(self, base, scales, ratios, grid_shape):
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

    def forward(self, x):
        cnn_map = self.backbone(x)
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

        anchor_grid = self.anchor_grid.reshape(-1, 4)

        boxes = []
        box_indicies = []
        for i in range(x.shape[0]):
            image_boxes = construct_boxes(box_deltas.view(box_deltas.shape[0], -1, 4)[i].data.cpu().numpy(), self.anchor_grid.reshape(-1, 4))

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

        boxes = Variable(torch.from_numpy(boxes))
        boxes = boxes.detach().contiguous()
        box_indicies = Variable(torch.from_numpy(box_indicies))
        box_indicies = box_indicies.detach()
        crops = self.crop_and_resize(cnn_map, boxes, box_indicies)

        boxes = boxes.data.numpy()
        box_indicies = box_indicies.data.numpy()

        # plt.cla()
        # plt.subplot(121)
        # f, ax = plt.gcf(), plt.gca()
        # plt.imshow(cnn_map[0][0].data.numpy())

        # for (y1, x1, y2, x2) in boxes[:10]: ax.add_patch(patches.Rectangle((x1 * 223, y1 * 223), x2 * 223 - x1 * 223, y2 * 223 - y1 * 223,linewidth=1,edgecolor='r',facecolor='none')) # and ax.add_patch(rect)

        # plt.subplot(885)
        # plt.imshow(crops[0][0].data.numpy())

        # plt.subplot(886)
        # plt.imshow(crops[1][0].data.numpy())

        # plt.subplot(887)
        # plt.imshow(crops[2][0].data.numpy())

        # plt.subplot(888)
        # plt.imshow(crops[3][0].data.numpy())

        # plt.subplot(212)
        # plt.imshow(crops[4][0].data.numpy())



        # plt.pause(1e-7)

        masks = self.mask_conv_1(crops)
        masks = self.relu(masks)
        masks = self.mask_conv_3(masks)
        masks = self.relu(masks)
        masks = self.mask_final(masks)
        masks = F.sigmoid(masks)

        masks = masks.view(box_scores.shape[0], box_scores.shape[1], 2, 14, 14)
        return box_scores, box_deltas, anchor_grid, masks

def iou(bboxes_a, bboxes_b):
    tl = np.maximum(bboxes_a[:, None, :2], bboxes_b[:, :2])
    br = np.minimum(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], axis=1)
    area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

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
    total_loss = cuda_pls(Variable(torch.FloatTensor([0])))
    for image_box_scores, image_gt_boxes in zip(box_scores, gt_boxes):
        labels, _ = as_labels_and_gt_indicies(anchors, image_gt_boxes)
        total_loss += F.cross_entropy(image_box_scores, cuda_pls(Variable(torch.from_numpy(labels.astype(np.int)))), ignore_index=-1)
    return total_loss / box_scores.shape[0]

def construct_deltas(gt_boxes, anchors):
    gt_boxes = gt_boxes.astype(np.float32)

    # TODO AS: I've reformulated it here to shifts from top-left. Double check that
    w_gt = gt_boxes[:, 2] - gt_boxes[:, 0]
    h_gt = gt_boxes[:, 3] - gt_boxes[:, 1]
    # x_centers_gt = gt_boxes[:, 0] + w_gt / 2
    # y_centers_gt = gt_boxes[:, 1] + h_gt / 2

    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    # x_centers_a = anchors[:, 0] + w_a / 2
    # y_centers_a = anchors[:, 1] + h_gt / 2

    t_x = (gt_boxes[:, 0] - anchors[:, 0]) / w_a
    t_y = (gt_boxes[:, 1] - anchors[:, 1]) / h_a
    t_w = np.log(w_gt / w_a)
    t_h = np.log(h_gt / h_a)
    return np.column_stack((t_x, t_y, t_w, t_h))

def construct_boxes(deltas, anchors):
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

def rpn_regressor_loss(gt_boxes, box_deltas, anchors):
    total_loss = cuda_pls(Variable(torch.FloatTensor([0])))
    for image_box_deltas, image_gt_boxes in zip(box_deltas, gt_boxes):
        labels, indicies = as_labels_and_gt_indicies(anchors, image_gt_boxes)
        positive_samples = np.argwhere(labels == 1).reshape(-1)
        positive_indicies = indicies[positive_samples]
        positive_deltas = image_box_deltas[[positive_samples]]
        positive_gt_boxes = image_gt_boxes[positive_indicies]
        positive_anchors = anchors[positive_samples]
        true_deltas = construct_deltas(positive_gt_boxes, positive_anchors)
        total_loss += F.smooth_l1_loss(positive_deltas, cuda_pls(Variable(torch.from_numpy(true_deltas))))
    return total_loss / box_deltas.shape[0]

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_image_and_boxes(image, boxes, masks=None):
    boxes = np.clip(boxes, 0, 223).astype(np.int32)

    plt.subplot(121)
    plt.cla()
    _, ax = plt.gcf(), plt.gca()
    ax.imshow(np.swapaxes(image, 0, 2))

    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    if masks is not None:
        plt.subplot(122)
        full = np.zeros(shape=image.shape[1:3])
        for (x1, y1, x2, y2), mask in zip(boxes, masks):
            height = y2 - y1
            width = x2 - x1

            if height > 0 and width > 0:
                torch_mask = Variable(torch.from_numpy(mask))
                predicted_mask = torch.nn.functional.upsample(torch_mask.unsqueeze(dim=0), size=(int(height), int(width)), mode='bilinear')[0][0]
                predicted_mask = predicted_mask.round().data.cpu().numpy()
                full[y1:y2, x1:x2] = predicted_mask

        plt.imshow(full)

    plt.draw()
    plt.pause(1e-17)

def generate_segmentation_batch(size):
    images = []
    gt_boxes = []
    masks = []

    for _ in range(size):
        image, bboxes, image_masks = generate_segmentation_image((224, 224))
        image = np.swapaxes(image, 0, 2)
        images.append(image)
        gt_boxes.append(bboxes)
        masks.append(image_masks)

    images = np.array(images)
    images = normalize(images)
    gt_boxes = np.array(gt_boxes)
    masks = np.array(masks)

    return images, gt_boxes, masks

def cuda_pls(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    return variable

def mask_loss(gt_boxes_batch, box_scores, box_deltas, anchors, predicted_masks, gt_masks):
    total_loss = cuda_pls(Variable(torch.FloatTensor([0])))

    for image_box_deltas, image_gt_boxes, image_predicted_masks, image_gt_masks in zip(box_deltas, gt_boxes_batch, predicted_masks, gt_masks):
        predicted_boxes = construct_boxes(image_box_deltas.data.numpy(), anchors)
        labels, indicies = as_labels_and_gt_indicies(predicted_boxes, image_gt_boxes, include_min=False, threshold=0.5)
        positive_samples = np.argwhere(labels == 1).reshape(-1)

        if len(positive_samples) == 0:
            continue

        positive_indicies = indicies[positive_samples]
        positive_deltas = image_box_deltas[[positive_samples]]
        positive_gt_boxes = image_gt_boxes[positive_indicies]
        positive_gt_masks = image_gt_masks[positive_indicies]
        positive_anchors = anchors[positive_samples]
        predicted_boxes = construct_boxes(positive_deltas.data.numpy(), positive_anchors)
        predicted_boxes = np.clip(predicted_boxes, 0, 223).astype(np.int32)
        positive_predicted_masks = image_predicted_masks[[positive_samples]]

        mask_bboxes = []
        mask_indicies = []
        for (x1, y1, x2, y2), one_mask in zip(predicted_boxes, positive_gt_masks):
            mask_bboxes.append([y1, x1, y2, x2])
            mask_indicies.append(len(mask_bboxes) - 1)


        mask_bboxes = (np.array(mask_bboxes) / 223).astype(np.float32)
        mask_bboxes = Variable(torch.from_numpy(mask_bboxes))
        mask_indicies = Variable(torch.from_numpy(np.array(mask_indicies).astype(np.int32)))
        positive_gt_masks = positive_gt_masks[:, np.newaxis, :, :]
        positive_gt_masks = Variable(torch.from_numpy(positive_gt_masks.astype(np.float32)))
        cropper = CropAndResizeFunction(14, 14)
        target_masks = cropper(positive_gt_masks, mask_bboxes, mask_indicies).round()

        for predicted_mask, target_mask in zip(positive_predicted_masks, target_masks):
            if (target_mask.shape[0] == 0) or (target_mask.shape[1] == 0):
                continue

            predicted_mask = predicted_mask[0]
            target_mask = target_mask[0]

            plt.cla()
            plt.subplot(121)
            plt.imshow(predicted_mask.data.numpy())
            plt.subplot(122)
            plt.imshow(target_mask.data.numpy())
            plt.draw()
            plt.pause(1e-7)

            total_loss += F.binary_cross_entropy(predicted_mask, target_mask) / len(target_masks)

    return total_loss / box_deltas.shape[0]

def fit(train_size=100, validation_size=10, batch_size=8, num_epochs=100):
    net = cuda_pls(MaskRCNN())
    optimizer = torch.optim.Adam([
        # TODO AS: Extract as a param
        # { 'params': net.backbone.parameters() },
        { 'params': filter(lambda p: p.requires_grad, net.parameters()) }
        # { 'params': net.rpn_conv.parameters(), 'lr': 0.002 },
        # { 'params': net.box_classifier.parameters(), 'lr': 0.002 },
        # { 'params': net.box_regressor.parameters(), 'lr': 0.001 },
    ], lr=0.001) #, momentum=0.9, nesterov=True, weight_decay=0.0005)

    validation_images, validation_gt_boxes, validation_masks = generate_segmentation_batch(validation_size)
    train_images, train_gt_boxes, train_masks = generate_segmentation_batch(train_size)
    validation_images = cuda_pls(Variable(torch.from_numpy(validation_images.astype(np.float32))))
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
            image_batch = cuda_pls(Variable(torch.from_numpy(image_batch.astype(np.float32))))

            optimizer.zero_grad()
            box_scores, box_deltas, anchors, predicted_masks = net(image_batch)

            cls_loss = rpn_classifier_loss(gt_boxes_batch, box_scores, anchors)
            reg_loss = rpn_regressor_loss(gt_boxes_batch, box_deltas, anchors)
            _mask_loss = mask_loss(gt_boxes_batch, box_scores, box_deltas, anchors, predicted_masks, gt_masks_batch)

            combined_loss = cls_loss + reg_loss + _mask_loss
            combined_loss.backward()
            optimizer.step()
            training_cls_loss += cls_loss.data[0] / num_batches
            training_reg_loss += reg_loss.data[0] / num_batches
            training_mask_loss += _mask_loss.data[0] / num_batches
            training_loss += combined_loss.data[0] / num_batches

        validation_scores, validation_deltas, validation_anchors, masks = net(image_batch)
        fg_scores = validation_scores[0][:, 1].data.cpu().numpy()
        top_prediction_indicies = np.argsort(fg_scores)[::-1]
        predicted_boxes = anchors[top_prediction_indicies[:20]]
        predicted_deltas = validation_deltas[0].data.cpu().numpy()[top_prediction_indicies[:20]]
        predicted_masks = masks[0].data.cpu().numpy()[top_prediction_indicies[:20]]

        actual_boxes = construct_boxes(predicted_deltas, predicted_boxes)

        img = image_batch[0].data.cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        display_image_and_boxes(img, actual_boxes, predicted_masks)

        validation_cls_loss = rpn_classifier_loss(validation_gt_boxes, validation_scores, validation_anchors)
        validation_reg_loss = rpn_regressor_loss(validation_gt_boxes, validation_deltas, validation_anchors)
        validation_loss = validation_cls_loss + validation_reg_loss
        tqdm.write(f'epoch: {epoch} - val reg: {validation_reg_loss.data[0]:.5f} - val cls: {validation_cls_loss.data[0]:.5f} - train reg: {training_reg_loss:.5f} - train cls: {training_cls_loss:.5f} - train mask: {training_mask_loss:.5f}')

def prof():
    import profile
    stats = profile.run('fit()')
    import pdb; pdb.set_trace()
if __name__ == '__main__':
    Fire()
