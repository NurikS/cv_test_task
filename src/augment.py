import albumentations as A
import cv2
from pathlib import Path

train_images_dir = Path("../data/train/images")
train_labels_dir = Path("../data/train/labels")


augments = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.Blur(blur_limit=3, p=0.1),
    A.RandomScale(scale_limit=0.1, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

for img_path in train_images_dir.rglob("*.jpg"):
    lbl_path = train_labels_dir / (img_path.stem + '.txt')

    if not lbl_path.exists():
        continue

    img = cv2.imread(str(img_path))
    bboxes = []
    labels = []

    with open(str(lbl_path), "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            bboxes.append([x, y, w, h])
            labels.append(int(cls))

    for i in range(3):
        augmented = augments(image=img, bboxes=bboxes, class_labels=labels)
        aug_img = augmented["image"]
        aug_label = augmented["class_labels"]
        aug_bboxes = augmented["bboxes"]

        out_img = train_images_dir / (f"{img_path.stem}_aug_{i}.jpg")
        cv2.imwrite(str(out_img), aug_img)

        out_label_path = train_labels_dir / f"{img_path.stem}_aug_{i}.txt"
        with open(out_label_path, "w") as f:
            for label, bbox in zip(aug_label, aug_bboxes):
                f.write(f"{label} {' '.join(map(str, bbox))}\n")




