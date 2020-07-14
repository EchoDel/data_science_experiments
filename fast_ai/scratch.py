from fast_ai.coco_support_functions import *
from fastai2.vision.all import *
from fastai2.vision.widgets import *
from pathlib import Path


def load_masks(image_id):
    annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.getCatIds()

    if len(anns) == 0:
        shape = (256,256)
    else:
        shape = coco.annToMask(anns[0]).shape
    base_mask = np.zeros((shape[0], shape[1]))
    masks = []
    for category in cats:
        annotation_short_list = [x for x in anns if x["category_id"] == category]
        if len(annotation_short_list) > 1:
            mask_list = [coco.annToMask(ann) for ann in annotation_short_list]
            mask = base_mask
            for x in mask_list:
                np.maximum(x, mask)
        else:
            mask = base_mask
        mask = resize_method(Image.fromarray(mask))
        mask = np.array(mask)
        masks.append(mask)

    return masks

def load_categories(noop):
    return [cat["name"] for cat in coco.loadCats(coco.getCatIds())]


def load_bbox(image_id):
    annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    return [size_bbox_to_points(ann['bbox']) for ann in anns]


def load_bbox_annotations(image_id):
    annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    return [coco.loadCats(ann['category_id'])[0]["name"] for ann in anns]




folder = Path('../coco_image/coco')
mode = 'train2017'
classes = None

img_ids, dataset_size, coco = filter_coco_dataset(folder, classes, mode)


def get_train_imgs(noop): return img_ids


getters = [lambda o: folder / "images" / (str(o).zfill(12) + ".jpg"),
           lambda o: load_masks(o),
           lambda o: load_categories(o)]

resize_method = Resize(256, ResizeMethod.Squish)

item_tfms = [resize_method]
batch_tfms = [Rotate(), Flip(), Dihedral(), Normalize.from_stats(*imagenet_stats)]

invalid_images = []
for x in range(0,len(img_ids)):
    img_ids_short = [img_ids[x]] * 100

    #MaskBlock

    images = DataBlock(
        blocks=(ImageBlock, MaskBlock, AddMaskCodes),
        get_items=get_train_imgs,
        splitter=TrainTestSplitter(valid_pct=0.2, seed=42),
        getters=getters,
        item_tfms=item_tfms,
        n_inp=1,
        batch_tfms=batch_tfms)

    try:
        dls = images.dataloaders("")

        dls.c = 1
        dls.show_batch()
        break
        plt.close()
    except Exception:
        invalid_images.append(img_ids[x])




learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)


image_id = 124983


[x for x in load_masks(image_id)]
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


load_bbox(6818)
