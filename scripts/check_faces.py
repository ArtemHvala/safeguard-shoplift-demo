"""Проверка Этапа 2: регистрация + идентификация лица."""
import pathlib
import urllib.request

import cv2
import numpy as np

import db
import recognize
from faces import embed_faces

URL = "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
src = pathlib.Path("data") / "sample_face.jpg"
src.parent.mkdir(exist_ok=True)
if not src.exists():
    req = urllib.request.Request(URL, headers={"User-Agent": "safeguard-shoplift-demo/0.1"})
    with urllib.request.urlopen(req) as r, open(src, "wb") as f:
        f.write(r.read())

img = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
faces = embed_faces(img)
print("faces found:", len(faces))

pathlib.Path(db.DB_PATH).unlink(missing_ok=True)  # чистый тест
db.init()
if faces:
    pid = db.add_person("Тест", faces[0]["embedding"])
    print("registered id:", pid)
    print("identify same:", recognize.identify(faces[0]["embedding"]))

rnd = np.random.rand(512).astype(np.float32)
print("identify random:", recognize.identify(rnd))

banana = cv2.cvtColor(cv2.imread("data/sample_banana.jpg"), cv2.COLOR_BGR2RGB)
print("faces in banana photo:", len(embed_faces(banana)))
