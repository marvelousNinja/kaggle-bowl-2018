import numpy as np
from PIL import Image, ImageDraw

from bowl.utils import normalize
from bowl.utils import from_numpy

def random_bbox_in(shape):
    width = np.random.randint(16, int(shape[1] / 5))
    height = np.random.randint(16, int(shape[0] / 5))
    x0 = np.random.randint(shape[1] - width)
    y0 = np.random.randint(shape[0] - height)
    x1, y1 = x0 + width - 1, y0 + height - 1
    return [x0, y0, x1, y1]

def intersects_with_any(bbox, other_bboxes):
    return any(intersects(bbox, other_bbox) for other_bbox in other_bboxes)

def intersects(bbox, other_bbox):
    return range_overlap(bbox[0], bbox[2], other_bbox[0], other_bbox[2]) and range_overlap(bbox[1], bbox[3], other_bbox[1], other_bbox[3])

def range_overlap(a_min, a_max, b_min, b_max):
    return (a_min <= b_max) and (b_min <= a_max)

def generate_segmentation_image(shape):
    image = Image.new('RGB', shape[::-1], 0)
    bboxes = []
    masks = []
    draw = ImageDraw.Draw(image)

    max_tries = 35
    tries = 0

    while tries <= max_tries:
        tries += 1
        mask = Image.new('L', shape[::-1], 0)
        mask_draw = ImageDraw.Draw(mask)
        bbox = random_bbox_in(shape)

        if intersects_with_any(bbox, bboxes):
            tries += 1
            continue

        random_color = tuple(np.random.randint(10, 255, 3))
        if np.random.rand() > 0.5:
            draw.ellipse(bbox, fill=random_color)
            mask_draw.ellipse(bbox, fill=255)
        else:
            draw.rectangle(bbox, fill=random_color)
            mask_draw.rectangle(bbox, fill=255)

        x0, x1 = bbox[::2]
        y0, y1 = bbox[1::2]
        x0, x1 = np.clip([x0, x1], 0, shape[1])
        y0, y1 = np.clip([y0, y1], 0, shape[0])
        bboxes.append([x0, y0, x1, y1])
        masks.append(mask)

    return np.array(image), np.array(bboxes), np.array(list(map(np.array, masks))) / 255

def generate_segmentation_batch(size, shape=(224, 224)):
    images = []
    gt_boxes = []
    masks = []

    for _ in range(size):
        image, bboxes, image_masks = generate_segmentation_image(shape)
        image = np.moveaxis(image, 2, 0)
        images.append(image)
        gt_boxes.append(bboxes)
        masks.append(image_masks)

    images = np.array(images)
    images = normalize(images)
    gt_boxes = np.array(gt_boxes)
    masks = np.array(masks)

    return from_numpy(images), (gt_boxes, masks)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.imshow(generate_segmentation_image((512, 512))[0])
    plt.show()
    images = np.array([generate_segmentation_image((512, 512))[0] for i in range(100)])
    print(f'Mean: {(images / 255).mean(axis=(1, 2)).mean(axis=0)}')
    print(f'Std: {(images / 255).std(axis=(1, 2)).mean(axis=0)}')
