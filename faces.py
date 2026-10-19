"""Распознавание лиц: MTCNN (детекция) + InceptionResnetV1 (эмбеддинг 512-d)."""
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = "cpu"
_mtcnn = None
_resnet = None


def get_models():
    """Ленивая загрузка; веса vggface2 скачиваются один раз."""
    global _mtcnn, _resnet
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=True, device=DEVICE)
        _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    return _mtcnn, _resnet


def embed_faces(image_rgb):
    """Список лиц: [{box:[x1,y1,x2,y2], prob, embedding: ndarray(512)}]."""
    mtcnn, resnet = get_models()
    boxes, probs = mtcnn.detect(image_rgb)
    if boxes is None:
        return []
    faces = mtcnn.extract(image_rgb, boxes, save_path=None)
    with torch.no_grad():
        embs = resnet(faces).cpu().numpy()
    return [
        {"box": [float(v) for v in box], "prob": float(prob), "embedding": emb}
        for box, prob, emb in zip(boxes, probs, embs)
    ]


def cosine(a, b):
    """Косинусная близость двух эмбеддингов."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))
