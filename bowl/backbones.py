import torch
import torchvision

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
        # x = self.cnn.layer3(x)
        # x = self.cnn.layer4(x)

        return x

    def output_channels(self):
        return 128

    def stride(self):
        return 8
