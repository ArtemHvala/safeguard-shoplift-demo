"""Streamlit-версия демо с ЖИВЫМ потоком с камеры (WebRTC).

Камера стримит в браузере, каждый кадр уходит в общий pipeline,
разметка возвращается в поток. Рассчитано на Streamlit Community Cloud.
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


class ShopliftProcessor(VideoProcessorBase):
    """Обрабатывает каждый кадр из WebRTC-потока."""

    def recv(self, frame):
        bgr = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = pipeline.process_frame(rgb)
        return av.VideoFrame.from_ndarray(
            cv2.cvtColor(out, cv2.COLOR_RGB2BGR), format="bgr24"
        )


st.title("🛡️ Safeguard Shoplift Demo")
st.caption(
    "Живой поток с камеры: детекция товара в руках и распознавание человека."
)

# STUN «знакомит» напрямую, TURN ретранслирует трафик — нужен, если сеть/VPN режет UDP.
# TCP/TLS на 443 проходит почти везде.
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
