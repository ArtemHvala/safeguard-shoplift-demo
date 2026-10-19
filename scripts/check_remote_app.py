"""Проверка deployed-приложения на Render: прогрев + прогон кадра через очередь."""
import inspect
import pathlib
import time
import urllib.request

import cv2
from gradio_client import Client, handle_file

URL = "https://safeguard-shoplift.onrender.com"

# 1) прогрев (free-инстанс Render засыпает, первый запрос может ждать ~50-60 с)
s = time.time()
urllib.request.urlopen(URL + "/", timeout=300).read(50)
print("warm-up:", round(time.time() - s, 1), "s", flush=True)

# 2) маленькая тестовая картинка, чтобы прогон не тянулся
small = pathlib.Path("data/small_banana.jpg")
img = cv2.imread("data/sample_banana.jpg")
h = int(img.shape[0] * 640 / img.shape[1])
cv2.imwrite(str(small), cv2.resize(img, (640, h)))
print("test image:", small, img.shape, flush=True)

# 3) клиент с длинным таймаутом (по умолчанию он слишком короткий для free-инстанса)
kwargs = {}
if "httpx_kwargs" in inspect.signature(Client.__init__).parameters:
    kwargs["httpx_kwargs"] = {"timeout": 600}
client = Client(URL, verbose=False, **kwargs)
print("connected", flush=True)

result = client.predict(handle_file(str(small)), api_name="/on_frame")
print("predict ->", result, flush=True)
