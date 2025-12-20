from ultralytics import YOLO

## commands used to export to js for website
model = YOLO("../weights/best.pt")
model.export(format="tfjs", imgsz=640)