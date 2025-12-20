from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    ## train model
    results = model.train(
        data="rawdataset_modified.yaml",
        imgsz=640,
        epochs=80,
        batch=16,
        device=0,
        cos_lr=True,
        patience=30,
        mixup=0.0,
        copy_paste=0.0,
    )

    best_weight_path = results.save_dir / "weights" / "best.pt"

    ## run metrics on newly trained model
    best_model = YOLO(best_weight_path)
    metrics = best_model.val(
        data="rawdataset_modified.yaml",
        imgsz=640,
        conf=0.01,
        iou=0.5,
        plots=True,
    )
    print("Validation metrics:", metrics)

    ## optional: make predictions immediately
    """
    model.predict(
        source="",
        conf=0.01,
        iou=0.5,
        augment=True
    )
    """

if __name__ == "__main__":
    main()