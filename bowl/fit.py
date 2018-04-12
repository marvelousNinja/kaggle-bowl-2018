from functools import partial

from fire import Fire
import numpy as np
import torch
from torch.nn.functional import softmax
from torch.nn.functional import sigmoid

from bowl.backbones import ResnetBackbone
from bowl.backbones import VGGBackbone
from bowl.faster_rcnn import FasterRCNN
from bowl.generators import toy_shapes_generator
from bowl.generators import bowl_train_generator
from bowl.generators import bowl_validation_generator
from bowl.losses import compute_loss
from bowl.training import fit_model
from bowl.utils import as_cuda
from bowl.utils import display_boxes
from bowl.utils import to_numpy

def fit(
        scales=[32], image_shape=(224, 224), ratios=[1.0],
        trainable_backbone=False, lr=0.001, dataset='toy',
        num_epochs=10, num_batches=10,
        visualize=False
    ):

    np.random.seed(1991)
    backbone = ResnetBackbone(trainable_backbone)
    model = as_cuda(FasterRCNN(backbone, scales, ratios))
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr)

    if dataset == 'toy':
        train_generator = toy_shapes_generator(image_shape)
        validation_generator = toy_shapes_generator(image_shape)
    else:
        train_generator = bowl_train_generator(image_shape)
        validation_generator = bowl_validation_generator(image_shape)

    fit_model(
        model,
        train_generator,
        validation_generator,
        optimizer,
        compute_loss,
        num_epochs,
        num_batches,
        after_validation=display_predictions if visualize else None
    )

def display_predictions(inputs, outputs, gt):
    display_boxes(outputs[-4], outputs[-3], to_numpy(outputs[-2]), np.moveaxis(to_numpy(inputs[0]), 0, 2))

def prof():
    import profile
    import pstats
    profile.run('fit()', 'fit.profile')
    stats = pstats.Stats('fit.profile')
    stats.sort_stats('cumulative').print_stats(30)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire()
