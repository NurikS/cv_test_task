import cv2
import os
from ultralytics import YOLO
import yaml

with open("configs/prices.yml", 'r') as f:
    prices = yaml.load(f, Loader=yaml.FullLoader)


def draw_boxes_and_price(frame, detections):
    total_price = 0

    for det in detections:
        xmin, ymin, xmax, ymax, cls_name = det
        price = prices.get(cls_name, 0)
        total_price += price

        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        label = f"{cls_name} - {price}"
        cv2.putText(frame, label, (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    total_label = f"Total: {total_price}"
    cv2.putText(frame, total_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    return frame


if __name__ == "__main__":
    model_path = "weights/best.pt"
    video_path = "clips/2_1.MOV"
    output_dir = "output"
    output_path = os.path.join(output_dir, "1_annotated.mp4")

    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3, verbose=False)
        detections = []

        for box in results[0].boxes.data.cpu().numpy():
            xmin, ymin, xmax, ymax, conf, cls_id = box
            cls_name = model.names[int(cls_id)]
            detections.append((xmin, ymin, xmax, ymax, cls_name))

        annotated_frame = draw_boxes_and_price(frame, detections)

        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Saved annotated video to {output_path}")
