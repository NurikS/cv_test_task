from ultralytics import YOLO

model = YOLO('runs/detect/train2/weights/best.pt')

search_space = {
    "lr0": (1e-5, 1e-1),
}

results = model.tune(data="../data/data.yaml", epochs=10, iterations=2, space=search_space, batch=-1, resume=True)