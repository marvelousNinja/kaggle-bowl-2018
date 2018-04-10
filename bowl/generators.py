from multiprocessing.pool import ThreadPool
from functools import partial

import numpy as np

from bowl.pipelines import get_train_image_ids
from bowl.pipelines import get_validation_image_ids
from bowl.pipelines import pipeline
from bowl.toy_shapes import generate_segmentation_batch

def toy_shapes_generator(image_shape=(224, 224)):
    while True:
        yield generate_segmentation_batch(1, shape=image_shape)

def bowl_train_generator(image_shape=(224, 224)):
    # TODO AS: Disabling multi-threading for now 'cause of OOM on Colab
    # pool = ThreadPool()
    train_ids = get_train_image_ids()

    while True:
        np.random.shuffle(train_ids)
        yield from map(partial(pipeline, image_shape), train_ids)

def bowl_validation_generator(image_shape=(224, 224)):
    validation_ids = get_validation_image_ids()

    while True:
        np.random.shuffle(validation_ids)
        yield from map(partial(pipeline, image_shape), validation_ids)
