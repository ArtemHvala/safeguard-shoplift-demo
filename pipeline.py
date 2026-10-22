"""Пайплайн обработки кадра без UI: детекция товара + лица + связка.

Используется и Gradio-версией (app.py), и Streamlit-версией (streamlit_app.py).
Тяжёлые зависимости (torch/ultralytics) импортируются ЛЕНИВО — чтобы сервер
успевал открыть порт (важно для Render и подобных хостингов).

Две «камеры»: физическая камера одна, режим переключается вручную
(set_mode(1) / set_mode(2)). Для каждого человека запоминается ПЕРВОЕ
появление в каждом режиме; разница — время перехода.
"""
import threading
import time

import cv2

import db

FACE_EVERY = 15  # считать лица раз в N кадров

_frame_no = 0
_last_people = []
_pipeline = None
_lock = threading.Lock()
_mode = 1  # текущий режим камеры: 1 или 2


def set_mode(mode):
    """Переключить «камеру» (1 или 2)."""
    global _mode
    _mode = 1 if int(mode) == 1 else 2


def get_mode():
    return _mode


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
                "id": res["id"],
                "name": recognize.display_name(res),
                "known": not res["name"].startswith(recognize.GUEST_PREFIX),
                "box": f["box"],
            }
        )
    _last_people = people

    # фиксируем первое появление каждого человека в текущем режиме камеры
    if people:
        now = time.time()
        for person in people:
            db.note_sighting(person["id"], _mode, now)
    return people


def process_frame(frame_rgb, force_faces=False):
    """RGB numpy -> размеченный RGB numpy.

    force_faces=True — считать лица на каждом кадре (режим «снимок»,
    где кэш между кадрами бессмыслен).
    """
    global _frame_no
    if frame_rgb is None:
        return None
    p = _get_pipeline()
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    products = p["track_products"](bgr)
    _frame_no += 1
    need_faces = force_faces or (_frame_no % FACE_EVERY == 1)
    people = _detect_people(frame_rgb) if need_faces else _last_people
    img = p["annotate"](bgr, p["link"](people, products))
    img = p["draw_people"](img, people)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def last_people():
    """Люди из последнего обработанного кадра."""
    return list(_last_people)


# ---------- учёт времени между «камерами» ----------

_PRETTY_PREFIX = "Гость #"


def _pretty(name):
    """'Гость #3' -> '#3'."""
    if name.startswith(_PRETTY_PREFIX):
        return "#" + name[len(_PRETTY_PREFIX):]
    return name


def _fmt(ts):
    return "" if ts is None else time.strftime("%H:%M:%S", time.localtime(ts))


def timeline_data():
    """Строки для таблицы: [человек, камера 1, камера 2, прошло, с]."""
    rows = []
    for _pid, name, t1, t2 in db.cam_timeline():
        delta = ""
        if t1 is not None and t2 is not None and t2 >= t1:
            delta = f"{t2 - t1:.1f}"
        rows.append([_pretty(name), _fmt(t1), _fmt(t2), delta])
    return rows


def timeline_headers():
    return ["Человек", "Камера 1", "Камера 2", "Прошло, с"]


def reset_timeline():
    """Сбросить записанные времена (люди остаются в базе)."""
    db.reset_sightings()


def init():
    """Подготовка БД (вызывать один раз при старте)."""
    db.init()
