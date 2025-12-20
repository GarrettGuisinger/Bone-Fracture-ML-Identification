# install fastai library
from fastai.vision.all import *
from pathlib import Path

root = Path("rawdataset")

def get_label(img_path: Path):
    label_file = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    label = label_file.read_text().strip().split()[0]
    if label == "0":
        return "Fracture"
    else:
        return "Healthy"

train_files = get_image_files(root/"train"/"images")
valid_files = get_image_files(root/"valid"/"images")
test_files  = get_image_files(root/"test"/"images")

items = train_files + valid_files
splitter = IndexSplitter(list(range(len(train_files), len(items))))

dblock = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items=lambda p: items,
    get_y=get_label,
    splitter=splitter,
    item_tfms=Resize(224),
)

dls = dblock.dataloaders(root, bs=32)

learn = vision_learner(dls, resnet18, metrics=[accuracy, RocAucBinary()])

learn.fine_tune(5)

learn.export("xray_CNN.pkl")