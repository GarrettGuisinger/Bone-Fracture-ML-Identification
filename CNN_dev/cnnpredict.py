from fastai.vision.all import *
from pathlib import Path

learn = load_learner("xray_fracture_fastai.pkl")

# input image
img_path = Path("rawdataset/test/images/XXX.jpg")

pred_class, pred_idx, probs = learn.predict(img_path)

print("Predicted class:", pred_class)
print("Class probabilities:")
for cls, p in zip(learn.dls.vocab, probs):
    print(f"{cls}: {float(p):.4f}")