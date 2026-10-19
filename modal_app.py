"""Деплой демо на Modal.

Отдаём Gradio как ASGI-приложение (@modal.asgi_app) — Modal поддерживает для него
WebSocket, который нужен очереди Gradio. Тема и конфиг готовятся в app.build().

Запуск:
    modal deploy modal_app.py
"""
import modal

APP_NAME = "safeguard-shoplift"
REMOTE_DIR = "/root/app"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .workdir(REMOTE_DIR)
    # заранее скачиваем веса в образ — холодный старт без загрузки моделей
    .run_commands(
        "python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"",
        "python -c \"from facenet_pytorch import InceptionResnetV1; "
        "InceptionResnetV1(pretrained='vggface2')\"",
    )
    # add_local_* должен идти ПОСЛЕДНИМ (требование Modal)
    .add_local_dir(
        ".",
        remote_path=REMOTE_DIR,
        ignore=[".venv", "data", ".git", "__pycache__", "*.db", ".gradio", "scripts"],
    )
)

app = modal.App(APP_NAME)


@app.function(image=image, max_containers=1, timeout=3600)
@modal.asgi_app()
def ui():
    import sys

    sys.path.insert(0, REMOTE_DIR)

    import app as shoplift  # наш app.py

    return shoplift.build().app
