"""Разовая проверка Этапа 1: детекция товара на реальном фото."""
import pathlib
import urllib.request

import cv2

from detector import detect_products, annotate

URL = "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg"
src = pathlib.Path("data") / "sample_banana.jpg"
src.parent.mkdir(exist_ok=True)
if not src.exists():
    req = urllib.request.Request(
        URL, headers={"User-Agent": "safeguard-shoplift-demo/0.1"}
    )
    with urllib.request.urlopen(req) as r, open(src, "wb") as f:
        f.write(r.read())

img = cv2.imread(str(src))
dets = detect_products(img)
print("detections:", dets)

out = src.with_name("sample_banana_out.jpg")
cv2.imwrite(str(out), annotate(img, dets))
print("saved:", out)
