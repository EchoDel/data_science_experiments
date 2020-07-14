from pycocotools.coco import COCO
import random
from fast_ai.coco_support_functions import *
from fastai2.vision.all import *
import numpy as np


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

    return [round(x) for x in output_list]


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


class MaskCoco:
    def load_masks(self, image_id):
        print(image_id)
        annIds = self.coco.getAnnIds(imgIds=image_id, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        cats = self.coco.getCatIds()

        if len(anns) == 0:
            shape = (256, 256)
        else:
            shape = self.coco.annToMask(anns[0]).shape
        base_mask = np.zeros((shape[0], shape[1]))
        masks = []
        for category in cats:
            annotation_short_list = [x for x in anns if x["category_id"] == category]
            if len(annotation_short_list) > 1:
                mask_list = [self.coco.annToMask(ann) for ann in annotation_short_list]
                mask = base_mask
                for x in mask_list:
                    np.maximum(x, mask)
            else:
                mask = base_mask
            mask = self.resize_method(Image.fromarray(mask))
            mask = np.array(mask)
            masks.append(mask)

        return masks

    def load_categories(self, noop):
        return [cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())]

    def get_image_ids(self):
        return self.img_ids

    def __init__(self, folder, classes, mode, resize_method):
        self.img_ids, self.dataset_size, self.coco = \
            filter_coco_dataset(folder, classes, mode)

        self.resize_method = resize_method


class BBoxCoco:
    def load_bbox(self, image_id):
        print(image_id)
        annIds = self.coco.getAnnIds(imgIds=image_id, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        if len(anns) ==0:
            return [0, 0, 0, 0]
        else:
            return [size_bbox_to_points(ann['bbox']) for ann in anns]

    def load_bbox_annotations(self, image_id):
        annIds = self.coco.getAnnIds(imgIds=image_id, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        cat_name = [self.coco.loadCats(ann['category_id'])[0]["name"] for ann in anns]
        cat_id = [ann['category_id'] for ann in anns]
        if len(cat_id)==0:
            return [0]
        else:
            return cat_id

    def get_image_ids(self):
        return self.img_ids

    def __init__(self, folder, classes, mode):
        self.img_ids, self.dataset_size, self.coco = \
            filter_coco_dataset(folder, classes, mode)

