"""Создаёт приватный HuggingFace Space (ZeroGPU) и загружает файлы проекта.

ZeroGPU — бесплатный вариант для Gradio Space (cpu-basic теперь требует PRO).
Требуется аккаунт старше 30 дней с подтверждённой почтой.

Токен берётся из HF_TOKEN (или из сохранённого логина huggingface_hub).
"""
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO_NAME = "safeguard-shoplift-demo"
HARDWARE = "zero-a10g"  # ZeroGPU

# что НЕ грузим в Space: локальные артефакты и лишнее
IGNORE = [
    "data/*", ".venv/*", "__pycache__/*", "*.pyc", ".git/*",
    "*.db", "yolov8n.pt", "scripts/*",
]


def main():
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{REPO_NAME}"
    print("repo:", repo_id)

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=HARDWARE,
            private=True,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        print("НЕ УДАЛОСЬ создать Space:", exc)
        print(
            "Подсказка: бесплатный Gradio-Space требует ZeroGPU и аккаунт "
            "старше 30 дней с подтверждённой почтой."
        )
        raise

    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE,
    )
    print("OK. Space:", f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
