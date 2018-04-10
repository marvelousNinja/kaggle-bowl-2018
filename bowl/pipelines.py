import glob
import os
from functools import partial

import cv2
import numpy as np

def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def get_train_image_ids():
    return list_dirs_in('./data/train')[10:]

def get_validation_image_ids():
    return list_dirs_in('./data/train')[:10]

def list_dirs_in(path):
    dirs = [dir.name for dir in os.scandir(path) if dir.is_dir() and not dir.name.startswith('.')]
    return np.sort(dirs)

def list_images_in(path):
    extensions = ['png']
    files = []
    for extension in extensions:
        files.extend(glob.glob(path + f'/*.{extension}'))
    return np.sort(files)

def read_image_by_id(image_id):
    image = read_image(f'./data/train/{image_id}/images/{image_id}.png')
    mask_paths = list_images_in(f'./data/train/{image_id}/masks')
    masks = []
    for mask_path in mask_paths:
        masks.append((read_image(mask_path)))
    return np.array(image), channels_last((np.array(masks)[:, :, :, 1] / 255).astype(np.uint8))

def read_image_by_id_cached(image_id, cache={}):
    if image_id in cache:
        image, masks = cache[image_id]
    else:
        image, masks = read_image_by_id(image_id)
        cache[image_id] = (image, masks)

    return image, masks

def mask_to_bounding_box(mask):
    a = np.where(mask != 0)
    return np.array([np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])])

def resize(x, y, image):
    return cv2.resize(image, (x, y), interpolation=cv2.INTER_CUBIC)

def channels_first(image):
    return np.moveaxis(image, 2, 0)

def channels_last(image):
    return np.moveaxis(image, 0, 2)

def normalize(image):
    image = image.astype(np.float32)
    image /= 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    return image

def non_empty(masks):
    masks = masks[np.nonzero(np.max(masks, axis=(1, 2)))]
    return masks[np.sqrt(np.sum(masks, axis=(1, 2))) > 4]

def crop(top, left, height, width, image):
    return image[top:top + height, left:left+width]

def generate_random_cropper(height, width, image_height, image_width):
    top = np.random.randint(image_height - height)
    left = np.random.randint(image_width - width)
    return partial(crop, top, left, height, width)

def rotate90(times, image):
    return np.rot90(image, times)

def generate_random_rotator():
    times = np.random.randint(4)
    return partial(rotate90, times)

def pipeline(image_shape, image_id):
    image, masks = read_image_by_id_cached(image_id)
    image = resize(image_shape[1] + 16, image_shape[0] + 16, image)
    masks = resize(image_shape[1] + 16, image_shape[0] + 16, masks)

    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    cropper = generate_random_cropper(image_shape[0], image_shape[1], image_shape[0] + 16, image_shape[1] + 16)
    rotator = generate_random_rotator()
    image = cropper(image)
    image = rotator(image)
    image = normalize(image)
    image = channels_first(image)

    masks = cropper(masks)
    masks = rotator(masks)
    masks = channels_first(masks)
    masks = non_empty(masks)
    masks = masks[np.nonzero(np.max(masks, axis=(1, 2)))]
    bboxes = np.array(list(map(mask_to_bounding_box, masks)))

    if len(bboxes) == 0:
        # TODO AS: Oh so dumb...
        return pipeline(image_shape, image_id)
    return image[None, :], bboxes[None, :], masks[None, :]

if __name__ == '__main__':
    import pdb; pdb.set_trace()
