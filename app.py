"""Gradio-версия демо (удобно запускать локально, с живым потоком).

Запуск:
    python app.py
"""
import os
import threading

import gradio as gr

import pipeline


def build():
    with gr.Blocks(title="Safeguard Shoplift Demo") as demo:
        gr.Markdown(
            "# Safeguard Shoplift Demo\n"
            "Распознавание человека + идентификация товара в руках."
        )
        with gr.Row():
            inp = gr.Image(
                sources=["webcam"], streaming=True, type="numpy", label="Камера"
            )
            out = gr.Image(label="Результат", streaming=True)
        inp.stream(pipeline.process_frame, inputs=inp, outputs=out, time_limit=30)
    # Gradio 6: при запуске без launch() надо самим подготовить тему и конфиг
    demo.theme = gr.themes.Default()
    demo.config = demo.get_config_file()
    demo.theme_css = demo.theme._get_theme_css()
    return demo


def _warmup():
    """Прогрев моделей в фоне — уже после того, как сервер откроет порт."""
    try:
        pipeline._get_pipeline()
    except Exception as exc:  # прогрев не должен ронять приложение
        print("warmup failed:", exc)


if __name__ == "__main__":
    pipeline.init()
    port = int(os.environ.get("PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
    demo = build()
    threading.Thread(target=_warmup, daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)
