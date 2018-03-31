import torch

class RPN(torch.nn.Module):
    def __init__(self, input_channels, anchors_per_location):
        super(RPN, self).__init__()

        self.conv = torch.nn.Sequential(
            self.init_layer(torch.nn.Conv2d(input_channels, 512, kernel_size=3, stride=1, padding=1)),
            torch.nn.ReLU()
        )

        self.classifier = self.init_layer(torch.nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1, padding=0))
        self.regressor = self.init_layer(torch.nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        rpn_map = self.conv(x)
        logits = self.classifier(rpn_map).permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 2)
        deltas = self.regressor(rpn_map).permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)
        return logits, deltas

    def init_layer(self, layer, mean=0, std=0.01):
        torch.nn.init.normal(layer.weight.data, mean, std)
        layer.weight.bias = 0
        return layer
