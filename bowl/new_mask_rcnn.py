import numpy as np
import torch
from tqdm import tqdm
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

        # for param in self.backbone.parameters():
            # param.requires_grad = False

        self.base = 32
        self.scales = [1]
        self.ratios = [1.0]
        self.anchors_per_location = len(self.scales) * len(self.ratios)
        self.image_height = 512
        self.image_width = 512
        self.anchor_grid_shape = (self.image_height // self.base, self.image_width // self.base)
        self.anchor_grid = self.generate_anchor_grid(base=self.base, scales=self.scales, ratios=self.ratios, grid_shape=self.anchor_grid_shape)

        self.rpn_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.box_classifier = nn.Conv2d(512, 2 * self.anchors_per_location, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

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

        return anchors

    def forward(self, x):
        cnn_map = self.backbone(x)
        rpn_map = self.rpn_conv(cnn_map)
        rpn_map = self.relu(rpn_map)

        box_scores = self.box_classifier(rpn_map)
        box_scores = box_scores.permute(0, 2, 3, 1).contiguous()
        box_scores = box_scores.view(box_scores.shape[0], box_scores.shape[1], box_scores.shape[2], self.anchors_per_location, 2)
        box_scores = F.softmax(box_scores, dim=4)
        return box_scores.view(box_scores.shape[0], -1, 2), self.anchor_grid.reshape(-1, 4)

def iou(bboxes_a, bboxes_b):
    tl = np.maximum(bboxes_a[:, None, :2], bboxes_b[:, :2])
    br = np.minimum(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], axis=1)
    area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def rpn_classifier_loss(gt_boxes, box_scores, anchors, images):
    ious = []
    for img_id in range(gt_boxes.shape[0]):
        ious.append(iou(anchors, gt_boxes[img_id]))

    total_loss = cuda_pls(Variable(torch.FloatTensor([0])))

    for image_ious, image_box_scores, gt, image, img_anchors in zip(ious, box_scores, gt_boxes, images, anchors):
        image_ious = np.array(image_ious)
        labels = np.full(image_ious.shape[0], -1)

        # Part of negative samples should be discarded
        labels[np.all(image_ious < 0.3, axis=1)] = 2
        negative_samples = np.argwhere(labels == 2)
        labels[np.random.choice(len(negative_samples), 225)] = 0
        labels[labels == 2] = -1

        labels[np.any(image_ious > 0.7, axis=1)] = 1
        labels[np.argmax(image_ious, axis=0)] = 1
        total_loss += F.cross_entropy(image_box_scores, cuda_pls(Variable(torch.from_numpy(labels.astype(np.float32)))), ignore_index=-1)

    return total_loss

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_image_and_boxes(image, boxes):
    plt.cla()
    _, ax = plt.gcf(), plt.gca()
    ax.imshow(np.swapaxes(image, 0, 2))

    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    plt.draw()
    plt.pause(1e-17)

def generate_segmentation_batch(size):
    images = []
    gt_boxes = []

    for _ in range(size):
        image, bboxes, _ = generate_segmentation_image((512, 512))
        image = np.swapaxes(image, 0, 2)
        images.append(image)
        gt_boxes.append(bboxes)

    images = np.array(images)
    images = normalize(images)
    gt_boxes = np.array(gt_boxes)

    return images, gt_boxes

def cuda_pls(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    return variable

def fit(train_size=100, validation_size=10, batch_size=8, num_epochs=100):
    net = cuda_pls(MaskRCNN())
    optimizer = torch.optim.SGD(
        # filter(lambda p: p.requires_grad, net.parameters()),
        [
            { 'params': net.backbone.parameters() },
            { 'params': net.rpn_conv.parameters(), 'lr': 0.01 },
            { 'params': net.box_classifier.parameters(), 'lr': 0.01 }

        ],
        lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.0005
    )
    validation_images, validation_gt_boxes = generate_segmentation_batch(validation_size)
    train_images, train_gt_boxes = generate_segmentation_batch(train_size)
    validation_images = cuda_pls(Variable(torch.from_numpy(validation_images.astype(np.float32))))
    num_batches = len(train_images) // batch_size

    for epoch in tqdm(range(num_epochs)):
        indicies = np.random.choice(range(len(train_images)), len(train_images))

        training_loss = 0.0
        for i in tqdm(range(num_batches)):
            batch_indicies = indicies[i * batch_size:i * batch_size + batch_size]
            image_batch, gt_boxes_batch = train_images[batch_indicies], train_gt_boxes[batch_indicies]
            image_batch = cuda_pls(Variable(torch.from_numpy(image_batch.astype(np.float32))))

            optimizer.zero_grad()
            box_scores, anchors = net(image_batch)

            loss = rpn_classifier_loss(gt_boxes_batch, box_scores, anchors, image_batch)
            loss.backward()
            optimizer.step()
            training_loss += loss.data[0] / num_batches

        validation_scores, validation_anchors = net(validation_images)
        # fg_scores = validation_scores[0][:, 1].data.cpu().numpy()
        # top_prediction_indicies = np.argsort(fg_scores)[::-1]
        # predicted_boxes = anchors[top_prediction_indicies[:30]]
        # img = validation_images[0].data.cpu().numpy()
        # img = (img - img.min()) / (img.max() - img.min())
        # display_image_and_boxes(img, predicted_boxes)

        validation_loss = rpn_classifier_loss(validation_gt_boxes, validation_scores, validation_anchors, validation_images)
        tqdm.write(f'epoch: {epoch} - val: {validation_loss.data[0]:.5f} - train: {training_loss:.5f}')

def prof():
    import profile
    stats = profile.run('fit()')
    import pdb; pdb.set_trace()
if __name__ == '__main__':
    Fire()
