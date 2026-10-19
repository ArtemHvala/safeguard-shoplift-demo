"""Сохраняет результат пайплайна для фото (доказательство работы)."""
import cv2

from app import on_frame

for name, out_name in [
    ("sample_face.jpg", "out_face.jpg"),
    ("sample_banana.jpg", "out_banana.jpg"),
]:
    rgb = cv2.cvtColor(cv2.imread(f"data/{name}"), cv2.COLOR_BGR2RGB)
    res = on_frame(rgb)  # на выходе RGB
    cv2.imwrite(f"data/{out_name}", cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    print("saved data/" + out_name)
