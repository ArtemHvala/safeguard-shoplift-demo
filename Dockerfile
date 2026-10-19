FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# заранее скачиваем веса в образ — холодный старт без загрузки моделей
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
 && python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')"

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0

# Cloud Run передаёт порт в переменной PORT
CMD ["python", "app.py"]
