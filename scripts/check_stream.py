"""Проверка Этапа 1b: трекинг между кадрами + on_frame не падает."""
import pathlib

import cv2

from app import on_frame
from detector import track_products

src = pathlib.Path("data") / "sample_banana.jpg"
bgr = cv2.imread(str(src))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # как отдаёт Gradio

for i in range(2):
    dets = track_products(bgr)
    print(f"frame {i}: {[(d['class'], d['track_id']) for d in dets]}")

out = on_frame(rgb)
print("on_frame ->", None if out is None else out.shape)
