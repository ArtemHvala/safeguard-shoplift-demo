"""Локальная проверка отдачи страницы через ASGI (как делает Modal)."""
from fastapi.testclient import TestClient

import app as shoplift

client = TestClient(shoplift.build().app)
resp = client.get("/")
print("status:", resp.status_code)
print(resp.text[:400])
