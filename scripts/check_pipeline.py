"""Проверка полного пайплайна: лицо и товар по отдельности."""
import pathlib

import cv2

from app import on_frame


def run(name):
    p = pathlib.Path("data") / name
    rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    out = on_frame(rgb)
    print(f"{name} -> {None if out is None else out.shape}")


run("sample_face.jpg")
run("sample_banana.jpg")
