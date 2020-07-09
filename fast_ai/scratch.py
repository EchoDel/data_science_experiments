from fast_ai.coco_support_functions import *
from fastai2.vision.all import *
from fastai2.vision.widgets import *
from pathlib import Path


def load_masks(image_id):
    annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    return [coco.annToMask(ann) for ann in anns]

def load_bbox(image_id):
    annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    return [ann['bbox'] for ann in anns]

def load_annotations(image_id):
    annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    return [ann['category_id'] for ann in anns]



folder = Path('../coco_image/coco')
mode = 'val2017'
classes = None

img_ids, dataset_size, coco = filter_coco_dataset(folder, classes, mode)

img_ids = img_ids[1:100]


def get_train_imgs(noop): return img_ids

getters = [lambda o: folder / "images" / (str(o).zfill(12) + ".jpg"),
           lambda o: load_bbox(o),
           lambda o: load_annotations(o)]

item_tfms = [Resize(224)]
#batch_tfms = [Rotate(), Flip(), Dihedral(), Normalize.from_stats(*imagenet_stats)]


#MaskBlock

images = DataBlock(
    blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
    get_items=get_train_imgs,
    splitter=TrainTestSplitter(valid_pct=0.2, seed=42),
    getters=getters,
    item_tfms=Resize(256, ResizeMethod.Squish),
    n_inp = 1)
#    batch_tfms=batch_tfms)


dls = images.dataloaders("")

dls.c = 1
dls.show_batch()




[len(x) for x in load_masks(508602)]
[x for x in load_annotations(508602)]


import skimage.io as io
import matplotlib.pyplot as plt

folder = Path('../coco_image/coco')
mode = 'val2017'
classes = None
images, dataset_size, coco = filter_coco_dataset(folder, classes, mode)

img = coco.loadImgs(6818)[0]

# load and display instance annotations
I = io.imread(img['coco_url'])
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()


load_bbox(6818)
