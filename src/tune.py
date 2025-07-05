from ultralytics import YOLO

model = YOLO('weights/best.pt')

search_space = {
    "lr0": (1e-5, 1e-1),
}

results = model.tune(data="../data/data.yaml", epochs=10, iterations=2, space=search_space, batch=-1, resume=True)
