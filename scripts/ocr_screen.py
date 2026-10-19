"""Распознаёт текст со скриншота экрана (Windows OCR через rapidocr)."""
import os
import pathlib

from rapidocr_onnxruntime import RapidOCR

path = pathlib.Path(os.environ["TEMP"]) / "oc_screen.png"
print("file:", path, "exists:", path.exists())

engine = RapidOCR()
result, _ = engine(str(path))
if not result:
    print("(текст не найден)")
else:
    for item in result:
        print(item[1])
