import torch
import torchvision
import bowl.toy_shapes
import bowl.utils
import bowl.pipelines
import numpy as np

from tqdm import tqdm

class PyramidFeature(torch.nn.Module):
    def __init__(self, c_channels, p_channels, out_channels):
        super(PyramidFeature, self).__init__()
        self.dimension_reducer = torch.nn.Conv2d(c_channels, p_channels, kernel_size=1, padding=0, stride=1)
        self.anti_aliaser = torch.nn.Conv2d(p_channels, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, c, p):
        p = torch.nn.functional.upsample(p, scale_factor=2)
        p = p + self.dimension_reducer(c)
        p = self.anti_aliaser(p)
        return p

class ResNetBackbone(torch.nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.cnn = torchvision.models.resnet50(pretrained=True)

        self.layer_p4 = torch.nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.layer_p3 = PyramidFeature(1024, 256, 256)
        self.layer_p2 = PyramidFeature(512, 256, 256)
        self.layer_p1 = PyramidFeature(256, 256, 256)

    def forward(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        c1 = self.cnn.layer1(x) # torch.Size([1, 256, 56, 56]) 4 ?
        c2 = self.cnn.layer2(c1) # torch.Size([1, 512, 28, 28]) 8 ?
        c3 = self.cnn.layer3(c2) # torch.Size([1, 1024, 14, 14]) 16 ?
        c4 = self.cnn.layer4(c3) # torch.Size([1, 2048, 7, 7]) 32 ?

        p4 = self.layer_p4(c4)
        p3 = self.layer_p3(c3, p4)
        p2 = self.layer_p2(c2, p3)
        p1 = self.layer_p1(c1, p2)

        return p1, p2, p3, p4

    def bases(self):
        return [4, 8, 16, 32]

    def scales(self):
        # {32^2, 64^2, 128^2, 256^2}
        return [32, 64, 128, 256]

    def grid_shape(self):
        return [(56, 56), (28, 28), (14, 14), (7, 7)]

    def generate_anchors(self, bases=[4, 8, 16, 32], scales=[32, 64, 128, 256], grid_shapes=[(56, 56), (28, 28), (14, 14), (7, 7)], ratios=[1.0]):
        anchors = []
        for (base, scale, (y_max, x_max)) in zip(bases, scales, grid_shapes):
            for y in range(y_max):
                for x in range(x_max):
                    for ratio in ratios:
                        height = scale / ratio
                        width = scale * ratio
                        y_ctr = int(base / 2) + y * base
                        x_ctr = int(base / 2) + x * base
                        x1, y1 = x_ctr - width / 2, y_ctr - height / 2
                        x2, y2 = x_ctr + width / 2 - 1, y_ctr + height / 2 - 1
                        anchors.append((x1, y1, x2, y2))
        return np.array(anchors)

class RPN(torch.nn.Module):
    def __init__(self, input_channels, anchors_per_location):
        super(RPN, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.ReLU()
        )

        self.classifier = torch.nn.Conv2d(256, 2 * anchors_per_location, kernel_size=1, stride=1, padding=0)
        self.regressor = torch.nn.Conv2d(256, 4 * anchors_per_location, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        rpn_map = self.conv(x)
        logits = self.classifier(rpn_map).permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 2)
        deltas = self.regressor(rpn_map).permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)
        return logits, deltas

class RCNN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(RCNN, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU()
        )

        self.classifier = torch.nn.Linear(1024, num_classes)
        self.regressor = torch.nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        x = self.layers(x)
        deltas = self.regressor(x)
        logits = self.classifier(x)
        return deltas, logits

class MaskHead(torch.nn.Module):
    def __init__(self, input_channels):
        super(MaskHead, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)

class MaskRCNN(torch.nn.Module):
    def __init__(self, num_scales, anchors_per_location):
        super(MaskRCNN, self).__init__()
        self.backbone = ResNetBackbone()
        self.rpns = [RPN(input_channels=256, anchors_per_location=anchors_per_location) for i in range(num_scales)]
        # self.rpn = RPN()
        # self.rcnn = RCNN(input_size=None, num_classes=None)
        # self.mask_head = MaskHead(input_channels=None)

    def forward(self, x):
        features_per_scale = self.backbone(x)
        rpn_outputs = [rpn(scale_features) for (scale_features, rpn) in zip(features_per_scale, self.rpns)]
        logits, deltas = list(zip(*rpn_outputs))
        logits, deltas = torch.cat(logits, dim=1), torch.cat(deltas, dim=1)
        return logits, deltas

        # rpn_logits, rpn_deltas  = self.rpn(features)
        # TODO AS: Convert logits & deltas to boxes
        # rcnn_crops = None
        # rcnn_logits, rcnn_deltas = self.rcnn(rcnn_crops)

        # TODO AS: Convert logits & deltas to boxes
        # mask_crops = None
        # masks = self.mask_head(mask_crops)

def as_labels_and_gt_indicies(anchors, gt_boxes, threshold=0.7):
    batch_size = 256
    ious = bowl.utils.iou(anchors, gt_boxes)
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
    return torch.nn.functional.cross_entropy(box_scores.view(-1, 2), bowl.utils.from_numpy(np.array(all_labels), dtype=np.int64), ignore_index=-1)

def rpn_regressor_loss(gt_boxes, box_deltas, anchors):
    all_gt_deltas = []
    all_pr_deltas = []

    for i in range(box_deltas.shape[0]):
        labels, gt_indicies = as_labels_and_gt_indicies(anchors, gt_boxes[i])
        positive = np.where(labels == 1)

        pr_deltas = box_deltas[i][positive]
        gt_deltas = bowl.utils.construct_deltas(gt_boxes[i][gt_indicies[positive]], anchors[positive])

        all_gt_deltas.extend(gt_deltas)
        all_pr_deltas.extend(pr_deltas)

    return torch.nn.functional.smooth_l1_loss(torch.stack(all_pr_deltas), bowl.utils.from_numpy(np.array(all_gt_deltas)))

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_predictions(image, box_scores, box_deltas, anchors, gt_boxes, gt_masks):
    image = np.clip(((bowl.utils.to_numpy(image) + 0.5) * 255).astype(np.int), 0, 255)
    scores = bowl.utils.to_numpy(torch.nn.functional.softmax(box_scores, dim=1))
    deltas = bowl.utils.to_numpy(box_deltas)
    shifted_boxes = bowl.utils.construct_boxes(deltas, anchors).astype(np.int)

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
    # plt.subplot(236)
    # plt.cla()
    # _, ax = plt.gcf(), plt.gca()
    # ax.imshow(np.moveaxis(image, 0, 2))

    # if len(keep_bbox_indicies) > 0:
    #     for (x1, y1, x2, y2) in shifted_boxes[keep_bbox_indicies]:
    #         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=1,edgecolor='r',facecolor='none')
    #         ax.add_patch(rect)

    # plt.pause(1e-17)

if __name__ == '__main__':
    # net = MaskRCNN(num_scales=4, anchors_per_location=1)
    # images, _, _ = bowl.toy_shapes.generate_segmentation_batch(1)
    # outputs = net(bowl.utils.bowl.utils.from_numpy(images))
    # import pdb; pdb.set_trace()

    net = bowl.utils.as_cuda(MaskRCNN(num_scales=4, anchors_per_location=1))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, weight_decay=0.0005)
    reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    validation_size = 1
    validation_ids = bowl.pipelines.get_validation_image_ids()[:validation_size]
    validation_images, validation_gt_boxes, validation_gt_masks = map(np.array, zip(*map(bowl.pipelines.pipeline, validation_ids)))

    validation_images = bowl.utils.from_numpy(validation_images.astype(np.float32))

    train_size = 100
    batch_size = 1
    train_ids = bowl.pipelines.get_train_image_ids()[:train_size]
    num_epochs = 100
    num_batches = len(train_ids) // batch_size

    for epoch in tqdm(range(num_epochs)):
        indicies = np.random.choice(train_ids, len(train_ids), replace=False)
        training_loss = 0.0
        for i in tqdm(range(num_batches)):
            batch_indicies = indicies[i * batch_size:i * batch_size + batch_size]
            image_batch, gt_boxes_batch, gt_masks_batch = map(np.array, zip(*map(bowl.pipelines.pipeline, batch_indicies)))
            image_batch = bowl.utils.from_numpy(image_batch.astype(np.float32))

            optimizer.zero_grad()
            logits, deltas = net(image_batch)
            anchors = net.backbone.generate_anchors()

            loss = rpn_classifier_loss(gt_boxes_batch, logits, anchors)
            loss += rpn_regressor_loss(gt_boxes_batch, deltas, anchors)
            loss.backward()
            optimizer.step()
            training_loss += loss.data[0] / num_batches

        logits, deltas = net(validation_images)
        anchors = net.backbone.generate_anchors()

        display_predictions(
            validation_images[0],
            logits[0],
            deltas[0],
            anchors,
            validation_gt_boxes[0],
            validation_gt_masks[0],
        )

        validation_loss = rpn_classifier_loss(validation_gt_boxes, logits, anchors)
        validation_loss += rpn_regressor_loss(validation_gt_boxes, deltas, anchors)
        validation_loss = validation_loss.data[0]
        reduce_lr.step(validation_loss)

        tqdm.write(f'epoch: {epoch} - val: {validation_loss:.5f} - train: {training_loss:.5f}')
