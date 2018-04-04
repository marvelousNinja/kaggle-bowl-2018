import torch

class RoIHead(torch.nn.Module):
    def __init__(self, roi_shape, num_classes):
        super(RoIHead, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(roi_shape[0] * roi_shape[1] * roi_shape[2], 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
        )

        self.classifier = torch.nn.Linear(1024, num_classes + 1)
        self.regressor = torch.nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        x = self.layers(x.view(x.shape[0], -1))
        logits = self.classifier(x.view(x.shape[0], -1))
        deltas = self.regressor(x.view(x.shape[0], -1))
        return logits, deltas
