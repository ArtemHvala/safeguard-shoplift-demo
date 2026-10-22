"""Streamlit-версия демо с ЖИВЫМ потоком с камеры (WebRTC).

Две «камеры» — один поток в двух режимах (переключаются радио-кнопкой).
Считаем время от первого появления человека в режиме 1 до появления в режиме 2.

Рассчитано на Streamlit Community Cloud.
"""
import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

import pipeline

st.set_page_config(page_title="Safeguard Shoplift Demo", layout="wide")


@st.cache_resource
def _boot():
    """БД + модели грузим ДО старта потока, иначе первые кадры «висят»."""
    pipeline.init()
    pipeline._get_pipeline()
    return True


with st.spinner("Загружаю модели…"):
    _boot()


PROCESS_EVERY = 3  # обрабатывать каждый N-й кадр (экономия CPU); промежуточные — последний результат


class ShopliftProcessor(VideoProcessorBase):
    """Обрабатывает кадры из WebRTC-потока (с пропуском для экономии CPU)."""

    def __init__(self):
        self._n = 0
        self._last = None

    def recv(self, frame):
        bgr = frame.to_ndarray(format="bgr24")
        self._n += 1
        if self._last is None or self._n % PROCESS_EVERY == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out = pipeline.process_frame(rgb)
            self._last = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(self._last, format="bgr24")


st.title("🛡️ Safeguard Shoplift Demo")
st.caption(
    "Живой поток с камеры: детекция товара в руках и распознавание человека."
)

# ---------- две «камеры»: один поток в двух режимах ----------
mode_label = st.radio(
    "Режим камеры",
    ["Камера 1 (вход)", "Камера 2 (зал)"],
    index=pipeline.get_mode() - 1,
    horizontal=True,
)
pipeline.set_mode(1 if mode_label.startswith("Камера 1") else 2)

# STUN «знакомит» напрямую, TURN ретранслирует трафик — нужен, если сеть/VPN режет UDP.
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turns:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
}

webrtc_streamer(
    key="shoplift",
    video_processor_factory=ShopliftProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"width": 640, "height": 480},
        "audio": False,
    },
)

# ---------- таблица времён ----------
st.subheader("Время появления")
rows = pipeline.timeline_data()
headers = pipeline.timeline_headers()
if rows:
    st.markdown(
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "|" + "|".join(["---"] * len(headers)) + "|",
                *(
                    "| " + " | ".join(str(cell) for cell in row) + " |"
                    for row in rows
                ),
            ]
        )
    )
else:
    st.caption("Пока никого не видели.")

if st.button("Обновить таблицу / Сбросить время", type="secondary"):
    pipeline.reset_timeline()
    st.rerun()
