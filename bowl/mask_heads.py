import torch

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
            torch.nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)
