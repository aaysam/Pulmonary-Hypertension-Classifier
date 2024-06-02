from pycocotools.coco import COCO
import os
from matplotlib import image
from pathlib import Path

img_dir = "/home/gena/Desktop/task_maxim_annotations_2024_05_02_13_37_04_coco 1.0/annotations/masks"
annFile = "/home/gena/Desktop/task_maxim_annotations_2024_05_02_13_37_04_coco 1.0/annotations/instances_default.json"

coco=COCO(annFile)

# Get category IDs and annotation IDs
catIds = coco.getCatIds()
annsIds = coco.getAnnIds()

# print(annsIds)

# Create folders named after annotation categories
for cat in catIds:
    Path(os.path.join("./your_output_folder",coco.loadCats(cat)[0]['name'])).mkdir(parents=True, exist_ok=True)

for ann in annsIds:
    # Get individual masks
    mask = coco.annToMask(coco.loadAnns(ann)[0])
    print(mask.sum())
    print("_______")
    # Save masks to BW images
    file_path = os.path.join("./your_output_folder",coco.loadCats(coco.loadAnns(ann)[0]['category_id'])[0]['name'],coco.loadImgs(coco.loadAnns(ann)[0]['image_id'])[0]['file_name'])
    image.imsave(file_path, mask, cmap="gray")
