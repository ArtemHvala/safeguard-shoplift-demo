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

webrtc_streamer(
    key="shoplift",
    video_processor_factory=ShopliftProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
