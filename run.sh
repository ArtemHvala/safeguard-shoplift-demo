#!/usr/bin/env bash
# Запуск демо на macOS / Linux: bash run.sh
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "=== Первый запуск: создаю окружение и ставлю зависимости ==="
  echo "Это займёт несколько минут."
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements-local.txt
fi

echo "=== Запускаю приложение ==="
echo "Открой в браузере: http://127.0.0.1:7860/"
exec ./.venv/bin/python app.py
