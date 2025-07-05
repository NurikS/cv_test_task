from ultralytics import YOLO

model = YOLO("../models/yolo11n.pt")

results = model.train(data="../data/data.yaml", epochs=100, imgsz=640, device=0, batch=-1)