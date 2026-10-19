"""Связка человек <-> товар.

Лицо даёт только бокс головы, поэтому расширяем его до условной «зоны человека»
и считаем товар принадлежащим тому, в чью зону попал центр бокса товара.
"""
import cv2

# лицо -> зона человека: влево/вправо x1.25, вниз x4 от высоты лица
SCALE_X = 1.25
SCALE_Y = 4.0


def person_region(face_box):
    x1, y1, x2, y2 = face_box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    return (cx - w * SCALE_X, y1, cx + w * SCALE_X, y1 + h * SCALE_Y)


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def link(people, products):
    """people: [{name, box}]; products: [{class, box, ...}].

    Возвращает товары с добавленным полем person (имя) или None.
    """
    regions = [(p.get("name"), person_region(p["box"])) for p in people]
    out = []
    for prod in products:
        c = _center(prod["box"])
        owner = next((name for name, region in regions if _inside(c, region)), None)
        out.append({**prod, "person": owner})
    return out


def draw_people(image, people):
    """Рисует боксы лиц, зоны людей и подписи с именами."""
    img = image.copy()
    for p in people:
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        color = (0, 200, 0) if p.get("known") else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        rx1, ry1, rx2, ry2 = (int(v) for v in person_region(p["box"]))
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 1)
        cv2.putText(
            img, p["name"], (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )
    return img
