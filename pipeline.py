"""Пайплайн обработки кадра без UI: детекция товара + лица + связка.

Используется и Gradio-версией (app.py), и Streamlit-версией (streamlit_app.py).
Тяжёлые зависимости (torch/ultralytics) импортируются ЛЕНИВО — чтобы сервер
успевал открыть порт (важно для Render и подобных хостингов).
"""
import threading

import cv2

import db

FACE_EVERY = 15  # считать лица раз в N кадров

_frame_no = 0
_last_people = []
_pipeline = None
_lock = threading.Lock()


def _get_pipeline():
    """Тяжёлые модули (torch/ultralytics) подгружаются при первом использовании."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                import linkage
                import recognize
                from detector import annotate, track_products
                from faces import embed_faces
                from linkage import draw_people

                _pipeline = {
                    "link": linkage.link,
                    "annotate": annotate,
                    "track_products": track_products,
                    "embed_faces": embed_faces,
                    "draw_people": draw_people,
                    "recognize": recognize,
                }
    return _pipeline


def _detect_people(frame_rgb):
    global _last_people
    p = _get_pipeline()
    recognize = p["recognize"]
    people = []
    for f in p["embed_faces"](frame_rgb):
        res = recognize.identify_or_register(f["embedding"])
        people.append(
            {
                "name": recognize.display_name(res),
                "known": not res["name"].startswith(recognize.GUEST_PREFIX),
                "box": f["box"],
            }
        )
    _last_people = people
    return people


def process_frame(frame_rgb):
    """RGB numpy -> размеченный RGB numpy."""
    global _frame_no
    if frame_rgb is None:
        return None
    p = _get_pipeline()
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    products = p["track_products"](bgr)
    _frame_no += 1
    people = _detect_people(frame_rgb) if (_frame_no % FACE_EVERY == 1) else _last_people
    img = p["annotate"](bgr, p["link"](people, products))
    img = p["draw_people"](img, people)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def last_people():
    """Люди из последнего обработанного кадра."""
    return list(_last_people)


def init():
    """Подготовка БД (вызывать один раз при старте)."""
    db.init()
