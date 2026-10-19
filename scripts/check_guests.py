"""Проверка нумерации новых лиц: первый раз «неизвестный #N», потом «#N».

Использует временную базу, реальную people.db не трогает.
Векторы — randn (центрированные), как настоящие эмбеддинги лиц.
"""
import pathlib

import numpy as np

import db
import recognize

db.DB_PATH = "data/test_guests.db"
pathlib.Path(db.DB_PATH).unlink(missing_ok=True)
db.init()

v1 = np.random.randn(512).astype(np.float32)
r1 = recognize.identify_or_register(v1)
print("первый раз: ", recognize.display_name(r1), "| new =", r1["new"])

r2 = recognize.identify_or_register(v1)
print("второй раз: ", recognize.display_name(r2), "| new =", r2["new"])

v3 = np.random.randn(512).astype(np.float32)
r3 = recognize.identify_or_register(v3)
print("другое лицо:", recognize.display_name(r3), "| new =", r3["new"])
