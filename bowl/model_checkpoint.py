from datetime import datetime
from functools import partial

import torch

def save_checkpoint(model, path):
    torch.save({'state_dict': model.state_dict()}, path)

def load_checkpoint(model, path):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['state_dict'])

def generate_checkpoint_path(prefix, timestamp, epoch, metrics):
    name = f'{prefix}-{timestamp}-{epoch:02d}-{metrics:.5f}.pt'
    return f'./data/models/{name}'

class ModelCheckpoint():
    def __init__(self, model, prefix, logger=None):
        self.epoch = 0
        self.metrics = 0
        self.model = model
        self.logger = logger

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
        self.generate_checkpoint_path = partial(generate_checkpoint_path, prefix, timestamp)

    def step(self, metrics):
        if metrics > self.metrics:
            checkpoint_path = self.generate_checkpoint_path(self.epoch, metrics)
            save_checkpoint(self.model, checkpoint_path)
            if self.logger: self.logger(f'Checkpoint saved {checkpoint_path}')
        self.epoch += 1
