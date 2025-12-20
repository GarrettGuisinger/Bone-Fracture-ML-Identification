from ultralytics import YOLO

best_model = YOLO("../weights/best.pt")

for conf in [0.01, 0.03, 0.05]:
    m = best_model.val(
        data="rawdataset_modified.yaml",
        imgsz=640,
        conf=conf,
        iou=0.5,
        plots=False,
        verbose=False
    )
    print(m.results_dict)