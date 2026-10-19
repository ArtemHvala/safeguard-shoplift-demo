"""Детекция и трекинг товара: YOLOv8n (pretrained COCO) + ByteTrack.

Модель общая, из неё берём только «товарные» COCO-классы.
"""
import cv2
from ultralytics import YOLO

# COCO-классы, которые считаем «товаром» для демо
PRODUCT_CLASSES = {
    "banana", "apple", "orange", "sandwich", "pizza", "cake",
    "bottle", "cup", "bowl",
}

_model = None


def get_model():
    """Ленивая загрузка: веса скачиваются один раз."""
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def _collect(res):
    """Из результата модели оставляет только товары."""
    ids = res.boxes.id
    out = []
    for i, b in enumerate(res.boxes):
        name = res.names[int(b.cls)]
        if name in PRODUCT_CLASSES:
            out.append(
                {
                    "class": name,
                    "conf": float(b.conf),
                    "track_id": int(ids[i]) if ids is not None else None,
                    "box": [float(v) for v in b.xyxy[0].tolist()],
                }
            )
    return out


def detect_products(image, conf=0.35):
    """Разовое обнаружение (без трекинга)."""
    res = get_model().predict(image, conf=conf, verbose=False)[0]
    return _collect(res)


def track_products(image, conf=0.35):
    """Обнаружение + трекинг между кадрами (persist=True)."""
    res = get_model().track(
        image, persist=True, conf=conf, verbose=False, tracker="bytetrack.yaml"
    )[0]
    return _collect(res)


def annotate(image, detections):
    """Рисует боксы товаров и подписи."""
    img = image.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tid = d.get("track_id")
        label = d["class"] + (f" #{tid}" if tid is not None else "")
        label += f" {d['conf']:.2f}"
        if d.get("person"):
            label += f" -> {d['person']}"
        cv2.putText(
            img, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return img
