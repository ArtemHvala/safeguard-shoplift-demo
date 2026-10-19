"""Streamlit-версия демо: снимок с камеры -> разметка.

Рассчитана на Streamlit Community Cloud (1 ГБ памяти, бесплатно, без карты).
Запуск: streamlit run streamlit_app.py
"""
import numpy as np
import streamlit as st
from PIL import Image

import pipeline

st.set_page_config(page_title="Safeguard Shoplift Demo", layout="wide")


@st.cache_resource
def _boot():
    pipeline.init()
    pipeline._get_pipeline()  # прогрев моделей один раз на процесс
    return True


with st.spinner("Загружаю модели…"):
    _boot()

st.title("🛡️ Safeguard Shoplift Demo")
st.caption(
    "Распознавание человека и товара в руках. "
    "Снимите кадр — покажем разметку."
)

photo = st.camera_input("Снимите кадр")

if photo is not None:
    frame = np.array(Image.open(photo).convert("RGB"))
    with st.spinner("Обрабатываю кадр…"):
        result = pipeline.process_frame(frame)
    st.image(result, caption="Результат")
    people = pipeline.last_people()
    if people:
        st.write("**Люди в кадре:**")
        for person in people:
            st.write(f"- {person['name']}")
