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
    dirs = [dir for dir in os.listdir(path) if not dir.startswith('.')]
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

    return np.array(image), masks

def mask_to_bounding_box(mask):
    a = np.where(mask != 0)
    return np.array([np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])])

def resize(x, y, image):
    return cv2.resize(image, (x, y), interpolation=cv2.INTER_CUBIC)

def channels_first(image):
    return np.moveaxis(image, 2, 0)

def to_binary_mask(mask):
    mask = (mask / 255).astype(np.uint8)
    return mask[:, :, 1]

def normalize(image):
    image = image.astype(np.float32)
    image /= 255
    image = image[::-1, :, :]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image[:, :, 0] -= mean[0]
    image[:, :, 1] -= mean[1]
    image[:, :, 2] -= mean[2]
    image[:, :, 0] /= std[0]
    image[:, :, 1] /= std[1]
    image[:, :, 2] /= std[2]
    return image

def non_empty(mask):
    if np.max(mask) == 0:
        return False

    x0, y0, x1, y1 = mask_to_bounding_box(mask)
    if (x1 - x0) * (y1 - y0) == 0:
        return False

    return True

def crop(top, left, height, width, image):
    return image[top:top + height, left:left+width].copy()

def generate_random_cropper(height, width, image_height, image_width):
    top = np.random.randint(image_height - height)
    left = np.random.randint(image_width - width)
    return partial(crop, top, left, height, width)

def rotate90(times, image):
    return np.rot90(image, times)

def generate_random_rotator():
    times = np.random.randint(4)
    return partial(rotate90, times)

def pipeline(image_id):
    image, masks = read_image_by_id(image_id)
    cropper = generate_random_cropper(224, 224, image.shape[0], image.shape[1])
    rotator = generate_random_rotator()
    image = cropper(image)
    image = rotator(image)
    image = channels_first(image)
    image = normalize(image)

    masks = map(cropper, masks)
    masks = map(rotator, masks)
    masks = map(to_binary_mask, masks)
    masks = filter(non_empty, masks)
    masks = np.array(list(masks))

    bboxes = np.array(list(map(mask_to_bounding_box, masks)))

    if len(bboxes) == 0:
        # TODO AS: Oh so dumb...
        return pipeline(image_id)
    return image, bboxes, masks

if __name__ == '__main__':
    import pdb; pdb.set_trace()
