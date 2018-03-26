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
    # TODO AS: Black background images to simplify the training
    whitelist = ['03f583ec5018739f4abb9b3b4a580ac43bd933c4337ad8877aa18b1dfb59fc9a', '0402a81e75262469925ea893b6706183832e85324f7b1e08e634129f5d522cdd', '04acab7636c4cf61d288a5962f15fa456b7bde31a021e5deedfbf51288e4001e', '05040e2e959c3f5632558fc9683fec88f0010026c555b499066346f67fdd0e13', '0532c64c2fd0c4d3188cc751cdfd566b1cfba3d269358717295bab1504c7c275', '05a8f65ebd0b30d3b210f30b4d640c847c2e710d0d135e0aeeaccbe1988e3b6e', '06350c7cc618be442c15706db7a68e91f313758d224de4608f9b960106d4f9ca', '06c779330d6d3447be21df2b9f05d1088f5b3b50dc48724fc130b1fd2896a68c', '072ff14c1d3245bf49ad6f1d4c71cdb18f1cb78a8e06fd2f53767e28f727cb81', '07761fa39f60dc37022dbbe8d8694595fd5b77ceb2af2a2724768c8e524d6770', '077f026f4ab0f0bcc0856644d99cbf639e443ec4f067d7b708bc6cecac609424', '07fb37aafa6626608af90c1e18f6a743f29b6b233d2e427dcd1102df6a916cf5', '08151b19806eebd58e5acec7e138dbfbb1761f41a1ab9620466584ecc7d5fada', '08ae2741df2f5ac815c0f272a8c532b5167ee853be9b939b9b8b7fa93560868a', '094afe36759e7daffe12188ab5987581d405b06720f1d5acf3f2614f404df380', '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9', '0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe', '0b0d577159f0d6c266f360f7b8dfde46e16fa665138bf577ec3c6f9c70c0cd1e', '0b2e702f90aee4fff2bc6e4326308d50cf04701082e718d4f831c8959fbcda93', '0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe', '0bf4b144167694b6846d584cf52c458f34f28fcae75328a2a096c8214e01c0d0', '0c6507d493bf79b2ba248c5cca3d14df8b67328b89efa5f4a32f97a06a88c92c', '0d2bf916cc8de90d02f4cd4c23ea79b227dbc45d845b4124ffea380c92d34c8c', '0ddd8deaf1696db68b00c600601c6a74a0502caaf274222c8367bdc31458ae7e', '0e5edb072788c7b1da8829b02a49ba25668b09f7201cf2b70b111fc3b853d14f', '0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1', '1023509cf8d4c155467800f89508690be9513431992f470594281cd37dbd020d', '10328b822b836e67b547b4144e0b7eb43747c114ce4cacd8b540648892945b00', '10ba6cbee4873b32d5626a118a339832ba2b15d8643f66dddcd7cb2ec80fbc28', '11a0170f44e3ab4a8d669ae8ea9546d3a32ebfe6486d9066e5648d30b4e1cb69', '12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40', '12f89395ad5d21491ab9cec137e247652451d283064773507d7dc362243c5b8e', '139946af9e2c7ef4f0298e622b831dbef5e5c0cd088eb5bc3382f8df9355443d', '13c8ff1f49886e91c98ce795c93648ad8634c782ff57eb928ce29496b0425057', '1400420310c9094361a8a243545187f1d4c2365e081b3bb08c5fa29c7491a55b', '14cc1424c59808274e123db51292e9dbb5b037ef3e7c767a8c45c9ac733b91bf', '150b0ffa318c87b31d78af0e87d60390dbcd84b5f228a8c1fb3225cbe5df3e3f', '1609b1b8480ee52652a644403b3f7d5511410a016750aa3b9a4c8ddb3e893e8e', '16c3d5935ba94b720becc24b7a05741c26149e221e3401924080f41e2f891368', '1740b0a67ca337ea31648b57c81bcfbb841c7bb5cad185199a9f4da596d531b9', '175dbb364bfefc9537931144861c9b6e08934df3992782c669c6fe4234319dfc', '1815cf307859b3e13669041d181aa3b3dbbac1a95aef4c42164b223110c09168', '193ffaa5272d5c421ae02130a64d98ad120ec70e4ed97a72cdcd4801ce93b066', '19f0653c33982a416feed56e5d1ce6849fd83314fd19dfa1c5b23c6b66e9868a', '1a75de9e11303142864efed27e69ea1960dbd82ca910de221a777ed2caf35a6b', '1a75e9f15481d11084fe66bc2a5afac6dc5bec20ed56a7351a6d65ef0fe8762b', '1b2bf5933b0fb82918d278983bee66e9532b53807c3638efd9af66d20a2bae88', '1b518cd2ea84a389c267662840f3d902d0129fab27696215db2488de6d4316c5', '1b6044e4858a9b7cee9b0028d8e54fbc8fb72e6c4424ab5b9f3859bfc72b33c5', '1bd0f2b3000b7c7723f25335fabfcdddcdf4595dd7de1b142d52bb7a186885f0', '1c2f9e121fc207efff79d46390df1a740566b683ff56a96d8cabe830a398dd2e', '1c681dfa5cf7e413305d2e90ee47553a46e29cce4f6ed034c8297e511714f867', '1c8b905c9519061d6d091e702b45274f4485c80dcf7fb1491e6b2723f5002180', '1d4a5e729bb96b08370789cad0791f6e52ce0ffe1fcc97a04046420b43c851dd', '1d5f4717e179a03675a5aac3fc1c862fb442ddc3e373923016fd6b1430da889b', '1db1cddf28e305c9478519cfac144eee2242183fe59061f1f15487e925e8f5b5', '1e61ecf354cb93a62a9561db87a53985fb54e001444f98112ed0fc623fad793e', '1e8408fbb1619e7a0bcdd0bcd21fae57e7cb1f297d4c79787a9d0f5695d77073', '1ee4a111f0e0bb9b001121b94ff98ca736fad03797b25285fe33a47046b3e4b0', '1f0008060150b5b93084ae2e4dabd160ab80a95ce8071a321b80ec4e33b58aca', '1f6b7cead15344593b32d5f2345fc26713dc74d9b31306c824209d67da401fd8', '1f9e429c12f4477221b5b855a5f494fda2ef6d064ff75b061ffaf093e91758c5', '20468e8779c43e089dc0ff30f25e6cf3872d5aa6a0fdad6f8aca382da43e8582', '20c37b1ad2f510ed7396969e855fe93d0d05611738f6e706e8ca1d1aed3ded45', '20e209f6ffa120a72712e1b4c1d3e24d1339227e2936abd4bbd49a636fada423', '212b858a66f0d23768b8e3e1357704fc2f4cf4bbe7eed8cd59b5d01031d553e6', '21408476af0506331e8b5d49b385833e5ef1fbb90815fbf9af9d19b4bb145f76', '2227fd9b01d67c2bcdb407d3205214e6dfeff9fd0725828e3b3651959942ff4a', '2349e95ece2857c89db7e4a8be8c88af0b45f3c4262608120cb3bd6ef51fd241', '237802ac5005f9cf782367156c46c383efd9e05088e5768ca883cbbe24abadb1', '23830d0e51245fc0c9e410efa4c17d2a7d83a0104a3777130119ab892de47a4e', '243443ae303cc09cfbea85bfd22b0c4f026342f3dfc3aa1076f27867910d025b', '245b995878370ef4ea977568b2b67f93d4ecaa9308761b9d3e148e0803780183', '27c30f9011492f234e4587c9a4b53c787037d486f658821196fe354240ac3c47', '2817299fd3b88670e86a9db5651ba24333c299d1d41e5491aabfcd95aee84174', '2869fad54664677e81bacbf00c2256e89a7b90b69d9688c9342e2c736ff5421c', '28d33efef218392e79e385906deb88055d94b65ad217de78c07e85476f80f45a', '295ac4ecf2ee0211c065cf5dbb93b1eb8e61347153447209cd110e9c3e355e81', '29780b28e6a75fac7b96f164a1580666513199794f1b19a5df8587fe0cb59b67']
    train_ids = get_train_image_ids()[:train_size]
    train_images, train_gt_boxes, train_gt_masks = map(np.array, zip(*map(pipeline, train_ids)))

    if overfit:
        validation_images, validation_gt_boxes, validation_gt_masks = train_images, train_gt_boxes, train_gt_masks
    else:
        validation_ids = get_validation_image_ids()[:validation_size]
        # TODO AS: Black background images to simplify the training
        validation_ids = whitelist[-validation_size:]
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
