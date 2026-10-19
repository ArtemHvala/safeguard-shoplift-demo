"""Сколько памяти реально ест приложение с загруженными моделями."""
import os

import psutil

proc = psutil.Process(os.getpid())


def mb():
    return round(proc.memory_info().rss / 1024 / 1024)


print("старт:", mb(), "МБ", flush=True)

import app  # noqa: E402

print("после импорта app:", mb(), "МБ", flush=True)

app._get_pipeline()
print("после загрузки моделей:", mb(), "МБ", flush=True)

import cv2  # noqa: E402

frame = cv2.imread("data/sample_banana.jpg")
frame = cv2.resize(frame, (640, 480))
app.on_frame(frame)
print("после обработки кадра:", mb(), "МБ", flush=True)
