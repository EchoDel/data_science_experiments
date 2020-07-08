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

 ,
 6818,
 480985,
 458054,
 331352,
 296649,
 386912,
 502136,
 491497,
 184791,
 348881,
 289393,
 522713,
 181666,
 17627,
 143931,
 303818,
 463730,
 460347,
 322864,
 226111,
 153299,
 308394,
 456496,
 58636,
 41888,
 184321,
 565778,
 297343,
 336587,
 122745,
 219578,
 555705,
 443303,
 500663,
 418281,
 25560,
 403817,
 85329,
 329323,
 239274,
 286994,
 511321,
 314294,
 233771,
 475779,
 301867,
 312421,
 185250,
 356427,
 572517,
 270244,
 516316,
 125211,
 562121,
 360661,
 16228,
 382088,
 266409,
 430961,
 80671,
 577539,
 104612,
 476258,
 448365,
 35197,
 349860,
 180135,
 486438,
 400573,
 109798,
 370677,
 238866,
 369370,
 502737,
 515579,
 515445,
 173383,
 438862,
 180560,
 347693,
 39956,
 321214,
 474028,
 66523,
 355257,
 142092,
 63154,
 199551,
 239347,
 514508,
 473237,
 228144,
 206027,
 78915]