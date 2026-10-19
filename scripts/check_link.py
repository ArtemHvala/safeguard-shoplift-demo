"""Проверка Этапа 3 на синтетических боксах (без моделей)."""
from linkage import link, person_region

face = {"name": "Тест", "box": [100, 100, 200, 200]}
print("region:", person_region(face["box"]))

products = [
    {"class": "banana", "box": [120, 300, 180, 400]},   # внутри зоны
    {"class": "cup", "box": [500, 300, 560, 400]},      # снаружи
]
for p in link([face], products):
    print(p["class"], "->", p["person"])
