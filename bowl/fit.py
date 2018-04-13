from functools import partial

import numpy as np
import torch
from fire import Fire
from tqdm import tqdm
from torch.nn.functional import softmax
from torch.nn.functional import sigmoid

from bowl.backbones import ResnetBackbone
from bowl.backbones import VGGBackbone
from bowl.generators import toy_shapes_generator
from bowl.generators import bowl_train_generator
from bowl.generators import bowl_validation_generator
from bowl.losses import compute_loss
from bowl.mask_rcnn import MaskRCNN
from bowl.metrics import mask_mean_average_precision
from bowl.model_checkpoint import ModelCheckpoint
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
    model = as_cuda(MaskRCNN(backbone, scales, ratios))
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr)
    model_checkpoint = ModelCheckpoint(model, 'mask-rcnn', logger=tqdm.write)

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
        after_validation=partial(after_validation, visualize, model_checkpoint)
    )

def after_validation(visualize, model_checkpoint, inputs, outputs, gt):
    if visualize:
        display_boxes(
            outputs.rcnn_detections,
            outputs.rcnn_detection_masks,
            to_numpy(outputs.rcnn_detection_scores),
            np.moveaxis(to_numpy(inputs[0]), 0, 2)
        )

    mask_map = mask_mean_average_precision(
        outputs.rcnn_detections,
        outputs.rcnn_detection_masks,
        to_numpy(outputs.rcnn_detection_scores.view(-1)),
        gt[1]
    )

    model_checkpoint.step(mask_map)
    tqdm.write(f'mask mAP {mask_map}')

def prof():
    import profile
    import pstats
    profile.run('fit()', 'fit.profile')
    stats = pstats.Stats('fit.profile')
    stats.sort_stats('cumulative').print_stats(30)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire()
