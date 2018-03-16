import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
from fire import Fire
from torchvision.models import resnet18

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
        x = self.cnn.layer4(x)
        return x

def normalize(image_batch):
    image_batch = image_batch.astype(np.float32)
    image_batch /= 255

    # Real images
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # Synth images
    mean = [0.08328683, 0.08328683, 0.08328683]
    std = [0.29126842, 0.29126842, 0.29126842]
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
        self.backbone = Backbone()

        self.base = 32
        self.scales = [1]
        self.ratios = [1.0]
        self.anchors_per_location = len(self.scales) * len(self.ratios)
        self.image_height = 512
        self.image_width = 512
        self.anchor_grid_shape = (self.image_height // self.base, self.image_width // self.base)
        self.anchor_grid = self.generate_anchor_grid(base=self.base, scales=self.scales, ratios=self.ratios, grid_shape=self.anchor_grid_shape)
        self.total_anchors = self.anchor_grid_shape[0] * self.anchor_grid_shape[1] * self.anchors_per_location

        self.anchors = np.repeat(self.anchor_grid[np.newaxis, :], 1, axis=0)
        self.anchors = Variable(torch.from_numpy(self.anchors), requires_grad=False)

        self.rpn_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.box_classifier = nn.Conv2d(512, 2 * self.anchors_per_location, kernel_size=1, stride=1, padding=0, bias=False)
        self.box_regressor = nn.Conv2d(512, 4 * self.anchors_per_location, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)

        # self.conv_1 = nn.Conv2d(512, 256, kernel_size=(15, 1), padding=(7,0), bias=False)
        # self.conv_2 = nn.Conv2d(256, 10*7*7, kernel_size=(1, 15), padding=(0,7), bias=False)

        # self.conv_3 = nn.Conv2d(512, 256, kernel_size=(1, 15), padding=(0,7), bias=False)
        # self.conv_4 = nn.Conv2d(256, 10*7*7, kernel_size=(15, 1), padding=(7,0), bias=False)

        # self.conv_5 = nn.Conv2d(490, 48, kernel_size=1, bias=False)
        # # TODO AS: Hell of elements here
        # self.fc = nn.Linear(24010, 2048)
        # self.classifier_fc = nn.Linear(2048, 2)
        # self.regressor_fc = nn.Linear(2048, 4)

    def generate_anchor_grid(self, base, scales, ratios, grid_shape):
        anchors = []
        x_ctr = int(base / 2)
        y_ctr = int(base / 2)

        for scale in scales:
            for ratio in ratios:
                for x_diff in range(grid_shape[0]):
                    for y_diff in range(grid_shape[0]):
                        width = int(base / ratio * scale)
                        height = int(base * ratio * scale)
                        x_ctr = int(base / 2) + x_diff * base
                        y_ctr = int(base / 2) + y_diff * base
                        x1, y1 = x_ctr - width / 2, y_ctr - height / 2
                        x2, y2 = x_ctr + width / 2, y_ctr + height / 2
                        anchors.append([x1, y1, x2, y2])

        return np.array(anchors, dtype=np.float32)

    def forward(self, x):
        cnn_map = self.backbone(x)
        rpn_map = self.rpn_conv(cnn_map)
        rpn_map = self.relu(rpn_map)

        box_scores = self.box_classifier(rpn_map).view(-1, self.total_anchors, 2)
        box_scores = F.softmax(box_scores, dim=2)
        box_deltas = self.box_regressor(rpn_map).view(-1, self.total_anchors, 4)

        # bboxes, scores, anchors = self.construct_boxes(box_scores, box_deltas, self.reference_anchors)
        # proposals = self.non_max_suppression(bboxes, scores)

        # conv_1 = self.conv_1( cnn_map)
        # conv_1 = self.relu(conv_1)
        # conv_1 = self.conv_2(conv_1)
        # conv_1 = self.relu(conv_1)

        # conv_2 = self.conv_3(cnn_map)
        # conv_2 = self.relu(conv_2)
        # conv_2 = self.conv_4(conv_2)
        # conv_2 = self.relu(conv_2)
        # cnn_map = conv_1.add(conv_2)

        # roi_features = roi_pooling(
        #     cnn_map.squeeze(),
        #     Variable(torch.cat([torch.arange(0, proposals.shape[0]).unsqueeze(1), proposals], 1)),
        #     size=(7, 7),
        #     spatial_scale=0.03125
        # )

        # conv_5 = self.conv_5(roi_features)
        # fc = self.fc(roi_features.view(conv_5.shape[0], -1))

        # final_scores = self.classifier_fc(fc)
        # final_deltas = self.regressor_fc(fc)

        # END HERE

        # final_boxes = self.deltas_to_boxes(final_deltas, proposals)

        # For training RPNs, we assign a binary class label
        # (of being an object or not) to each anchor. We assign
        # a positive label to two kinds of anchors: (i) the
        # anchor/anchors with the highest Intersection-overUnion
        # (IoU) overlap with a ground-truth box, or (ii) an
        # anchor that has an IoU overlap higher than 0.7 with
        # 5
        # any ground-truth box. Note that a single ground-truth
        # box may assign positive labels to multiple anchors.
        # Usually the second condition is sufficient to determine
        # the positive samples; but we still adopt the first
        # condition for the reason that in some rare cases the
        # second condition may find no positive sample. We
        # assign a negative label to a non-positive anchor if its
        # IoU ratio is lower than 0.3 for all ground-truth boxes.
        return box_scores, box_deltas, self.anchors #, final_scores, final_deltas

    # TODO AS: Cleanup transforms and extract params for shapes, number of anchors and etc.
    def construct_boxes(self, scores, deltas, reference_anchors):
        anchors = np.array(reference_anchors).reshape(-1)
        anchors = np.repeat(anchors, 7 * 7)
        anchors = anchors.reshape(1, 48, 7, 7)
        x_diffs = np.arange(0, 7) * 32
        y_diffs = (np.arange(0, 7) * 32).T
        anchors[:, 1::4, :, :] += x_diffs
        anchors[:, ::4, :, :] += y_diffs
        anchors = Variable(torch.from_numpy(anchors.astype(np.float32)))

        bboxes = self.deltas_to_boxes(deltas, anchors)

        bg_scores = scores[:, :12, :, :].view(-1)
        fg_scores = scores[:, 12:, :, :].view(-1)

        scores = torch.stack([
            bg_scores,
            fg_scores
        ], dim=1)

        scores = F.softmax(scores, 1)[:, 1]

        return bboxes, scores, anchors

    def deltas_to_boxes(self, deltas, boxes):
        heights = torch.mul(torch.exp(deltas[:, 2::4, :, :]), boxes[:, 2::4, :, :])
        widths = torch.mul(torch.exp(deltas[:, 3::4, :, :]), boxes[:, 3::4, :, :])
        x_ctrs = torch.mul(deltas[:, ::4, :, :], boxes[:, ::4, :, :]) + boxes[:, ::4, :, :]
        y_ctrs = torch.mul(deltas[:, 1::4, :, :], boxes[:, 1::4, :, :]) + boxes[:, 1::4, :, :]

        bboxes = torch.clamp(torch.stack([
            (x_ctrs - widths / 2).view(-1),
            (y_ctrs - heights / 2).view(-1),
            (x_ctrs + widths / 2).view(-1),
            (y_ctrs + heights / 2).view(-1)
        ], dim=1), 0, 223)

        return bboxes

    # TODO AS: Cleanup area filters
    def non_max_suppression(self, bboxes, scores, threshold=0.5, mode='union'):
        bboxes = bboxes.data
        scores = scores.data

        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]

        areas = (x2-x1+1) * (y2-y1+1)
        larger_than_min = (areas > 10).nonzero().squeeze()
        areas = areas[larger_than_min]
        bboxes = bboxes[larger_than_min]
        scores = scores[larger_than_min]

        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]

        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]]
            yy1 = y1[order[1:]]
            xx2 = x2[order[1:]]
            yy2 = y2[order[1:]]

            w = (xx2-xx1+1).clamp(min=0)
            h = (yy2-yy1+1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]]
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr<=threshold).nonzero().squeeze()

            if ids.numel() == 0:
                break

            order = order[ids+1]

        return bboxes[keep]

def iou(one, another, image):
    x_inter = (torch.min(one[2], another[2]) - torch.max(one[0], another[0])).clamp(0)
    y_inter = (torch.min(one[3], another[3]) - torch.max(one[1], another[1])).clamp(0)
    inter = x_inter * y_inter

    if (inter == 0).all():
        return 0
    else:
        one_area = (one[2] - one[0]) * (one[3] - one[1])
        another_area = (another[2] - another[0]) * (another[3] - another[1])
        val = (inter / (one_area + another_area - inter)).data.numpy()[0]

        # if val > 0:
        #     display_image_and_boxes(image, torch.cat((one, another)).view(-1, 4).data.numpy())
        #     print(val)
        #     input('waiting...')
        return val

def rpn_classifier_loss(gt_boxes, box_scores, anchors, image):
    anchors = anchors.clamp(0, 511)

    ious = []
    for img_id in range(anchors.shape[0]):
        image_ious = []
        ious.append(image_ious)

        for anchor in anchors[img_id]:
            anchor_ious = []
            image_ious.append(anchor_ious)

            for gt_box in gt_boxes[img_id]:
                some_iou = iou(anchor, gt_box, image)
                anchor_ious.append(some_iou)

    ious = np.array(ious)
    labels = np.full(ious.shape[0:2], -1)
    labels[np.any(ious < 0.2, axis=2)] = 0
    labels[np.any(ious > 0.7, axis=2)] = 1
    labels[:, np.argmax(ious, axis=1)] = 1

    negative_examples = np.argwhere(labels == 0)
    positive_examples = np.argwhere(labels == 1)

    # Batch size here is 256...
    # TODO AS: Some balancing
    negative_examples = negative_examples[np.random.choice(len(negative_examples), 10)]
    indicies = np.concatenate([negative_examples, positive_examples])
    # print(len(positive_examples))
    # if len(positive_examples) > 0:
    #     predicted_boxes = np.clip(anchors[0][[positive_examples[:, 1]]].data.numpy(), a_min=0, a_max=None)
    #     display_image_and_boxes(image, predicted_boxes)
    #     # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()

    return F.cross_entropy(box_scores[indicies[:, 0], indicies[:, 1]], Variable(torch.from_numpy((labels[indicies[:, 0], indicies[:, 1]].astype(int)))))

def rpn_regressor_loss():
    pass

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_image_and_boxes(image, boxes):
    # fig, ax = plt.subplots(1)
    plt.cla()
    fig, ax = plt.gcf(), plt.gca()
    ax.imshow(np.swapaxes(image, 0, 2))

    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    plt.draw()
    plt.pause(1e-17)

def fit():
    net = MaskRCNN()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, nesterov=True)

    for _ in range(100):
        mini_batch_loss = 0.0

        optimizer.zero_grad()
        for _ in range(10):

            image_batch = []
            gt_boxes_batch = []

            for _ in range(1):
                image, bboxes, _ = generate_segmentation_image((512, 512))
                image = np.swapaxes(image, 0, 2)
                image_batch.append(image)
                gt_boxes_batch.append(bboxes)

            image_batch = np.array(image_batch)
            image_batch = normalize(image_batch)
            gt_boxes_batch = np.array(gt_boxes_batch)

            image_batch = Variable(torch.from_numpy(image_batch.astype(np.float32)))
            gt_boxes_batch = Variable(torch.from_numpy(gt_boxes_batch.astype(np.float32)))

            box_scores, _, anchors = net(image_batch)
            loss = rpn_classifier_loss(gt_boxes_batch, box_scores, anchors, image) / 10

            mini_batch_loss += loss.data[0]

            num_preds = (box_scores[0][:, 1] > 0.5).detach().nonzero().squeeze().numel()
            if num_preds > 0:
                predicted_boxes = anchors[0][(box_scores[0][:, 1] > 0.5).detach().nonzero().squeeze()].data.numpy()
                display_image_and_boxes(image, predicted_boxes)

            loss.backward()

        optimizer.step()
        print(mini_batch_loss / 10.0)

    # image, bboxes, masks = generate_segmentation_image((224, 224))
    # image = image.transpose(2, 0, 1)

    # image_batch = image[np.newaxis, :]
    # image_batch = normalize(image_batch)
    # gt_boxes_batch = bboxes[np.newaxis, :]

    # image_batch = Variable(torch.from_numpy(image_batch.astype(np.float32)))
    # gt_boxes_batch = Variable(torch.from_numpy(gt_boxes_batch.astype(np.float32)))

    # box_scores, box_deltas, anchors = net(image_batch)
    # rpn_classifier_loss(gt_boxes_batch, box_scores, anchors)

if __name__ == '__main__':
    Fire()
