from pycocotools.coco import COCO
import random
from fastai2.vision import *


def unique_list(input_data: list):
    # Now, filter out the repeated images
    output_data = []
    for i in range(len(input_data)):
        if input_data[i] not in output_data:
            output_data.append(input_data[i])

    return output_data

def switch_list(input_list, n):
    output_list = input_list.copy()
    for x in range(n):
        input_list.append(
            input_list.pop(
                input_list.index(input_list[0])
            )
        )
    return input_list

def reorder_list(input_list, order):
    return [input_list[x] for x in order]

def size_bbox_to_points(input_list):
    output_list = input_list[0:2]
    output_list.append(input_list[0] + input_list[2])
    output_list.append(input_list[1] + input_list[3])

    return output_list

def filter_coco_dataset(folder, classes=None, mode='train'):
    # initialize COCO api for instance annotations
    annFile = '{}/annotations/instances_{}.json'.format(folder, mode)
    coco = COCO(annFile)

    images = []
    if classes is not None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            cat_ids = coco.getCatIds(catNms=className)
            img_ids = coco.getImgIds(catIds=cat_ids)
            images += coco.loadImgs(img_ids)

        unique_images = unique_list(images)
    else:
        img_ids = coco.getImgIds()
        unique_images = coco.loadImgs(img_ids)

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return img_ids, dataset_size, coco









