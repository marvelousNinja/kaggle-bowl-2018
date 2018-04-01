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

class MultiRPN(torch.nn.Module):
    def __init__(self, input_channels, anchors_per_location, num_scales):
        super(MultiRPN, self).__init__()
        self.rpns = [RPN(input_channels, anchors_per_location) for i in range(num_scales)]

    def forward(self, x):
        outputs = [rpn(scale_maps) for (scale_maps, rpn) in zip(x, self.rpns)]
        logits, deltas = zip(*outputs)
        return torch.cat(logits, dim=1), torch.cat(deltas, dim=1)
