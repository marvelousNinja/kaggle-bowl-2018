from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
import torch
from fire import Fire
from torchvision.models import resnet18

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

def generate_anchors():
    return []

def construct_boxes():
    return []

class MaskRCNN(nn.Module):
    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.backbone = Backbone()

        # RPN
        self.anchors = generate_anchors()
        self.num_anchors = len(self.anchors)
        self.rpn_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.box_classifier = nn.Conv2d(512, 2 * self.num_anchors, kernel_size=1, stride=1, padding=0, bias=False)
        self.box_regressor = nn.Conv2d(512, 4 * self.num_anchors, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        cnn_map = self.backbone(x)
        rpn_map = self.rpn_conv(cnn_map)
        rpn_map = self.relu(x)
        box_scores = self.box_classifier(rpn_map)
        box_deltas = self.box_regressor(rpn_map)
        proposals = construct_boxes(box_scores, box_deltas, self.anchors)

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
        return x

    def construct_boxes(self, scores, deltas, anchors):
        pass


# def generate_anchors(base=16, scales=[1,2,3]):


def fit():
    net = MaskRCNN()
    input = Variable(torch.randn(2, 3, 224, 224))
    box_scores, box_locations = net(input)

# def fit():
#     net = Net()
#     input = Variable(torch.randn(1, 1, 32, 32))
#     output = net(input)
#     target = Variable(torch.arange(1, 11))
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

#     while True:
#         optimizer.zero_grad()
#         loss = criterion(output, target)
#         optimizer.step()
#         print(net.parameters())
#         import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire()
