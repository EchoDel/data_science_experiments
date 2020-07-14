from fast_ai.coco_support_functions import *
from fastai2.vision.all import *
from fastai2.vision.widgets import *
from pathlib import Path
import fastai2.metrics as M


folder = Path('../coco_image/coco')
mode = 'val2017'
classes = None

# img_ids, dataset_size, coco = filter_coco_dataset(folder, classes, mode)

resize_method = Resize(256, ResizeMethod.Squish)
item_tfms = [resize_method]
batch_tfms = [Rotate(), Flip(), Dihedral(), Normalize.from_stats(*imagenet_stats)]


def get_train_imgs(noop): return img_ids



# Mask approach
coco_mask = MaskCoco(folder, classes, mode, resize_method)

img_ids = coco_mask.get_image_ids()

getters_mask = [lambda o: folder / "images" / (str(o).zfill(12) + ".jpg"),
           lambda o: coco_mask.load_masks(o),
           lambda o: coco_mask.load_categories(o)]

images_mask = DataBlock(
    blocks=(ImageBlock, MaskBlock, AddMaskCodes),
    get_items=get_train_imgs,
    splitter=TrainTestSplitter(valid_pct=0.2, seed=42),
    getters=getters_mask,
    item_tfms=item_tfms,
    n_inp=1,
    batch_tfms=batch_tfms)


dls = images_mask.dataloaders("", num_workers=0)

dls.c = 20
dls.show_batch()
plt.close()


# Boundary box approach
coco_bbox = BBoxCoco(folder, classes, mode)

img_ids = coco_bbox.get_image_ids()

# bboxes
getters_bbox = [lambda o: folder / "images" / (str(o).zfill(12) + ".jpg"),
               lambda o: coco_bbox.load_bbox(o),
               lambda o: coco_bbox.load_bbox_annotations(o)]

images_mask = DataBlock(
    blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
    get_items=get_train_imgs,
    splitter=TrainTestSplitter(valid_pct=0.2, seed=42),
    getters=getters_bbox,
    item_tfms=item_tfms,
    n_inp=1,
    batch_tfms=batch_tfms)


dls = images_mask.dataloaders("", num_workers=0)

dls.c = 1
dls.show_batch()
plt.close()







learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)


image_id = 124983


[x for x in load_masks(image_id)]
[x for x in load_categories(image_id)]
[x for x in load_bbox_annotations(image_id)]
[x for x in load_bbox(image_id)]




import skimage.io as IO
import matplotlib.pyplot as plt

folder = Path('../coco_image/coco')
mode = 'val2017'
classes = None
images, dataset_size, coco = filter_coco_dataset(folder, classes, mode)

img = coco.loadImgs(image_id)[0]

# load and display instance annotations
I = IO.imread(img['coco_url'])
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()

