"""Gradio-версия демо (локально, живой поток с камеры).

Две «камеры» — это один поток в двух режимах: переключаются кнопкой.
Считаем время от первого появления человека в режиме 1 до появления в режиме 2.

Запуск:
    python app.py
"""
import os
import threading

import gradio as gr

import pipeline


def on_frame(frame):
    """Кадр -> размеченная картинка + таблица времён."""
    img = pipeline.process_frame(frame)
    return img, pipeline.timeline_data()


def on_mode(mode):
    pipeline.set_mode(mode)
    return pipeline.timeline_data()


def on_reset():
    pipeline.reset_timeline()
    return pipeline.timeline_data()


def build():
    with gr.Blocks(title="Safeguard Shoplift Demo") as demo:
        gr.Markdown(
            "# Safeguard Shoplift Demo\n"
            "Распознавание человека + идентификация товара в руках.\n\n"
            "**Две камеры:** переключайте режим, когда человек проходит дальше. "
            "Время считается от первого появления в «Камере 1» до появления "
            "того же человека в «Камере 2»."
        )
        with gr.Row():
            mode = gr.Radio(
                choices=[("Камера 1 (вход)", 1), ("Камера 2 (зал)", 2)],
                value=1,
                label="Режим камеры",
            )
            reset = gr.Button("Сбросить время")
        with gr.Row():
            inp = gr.Image(
                sources=["webcam"], streaming=True, type="numpy", label="Камера"
            )
            out = gr.Image(label="Результат", streaming=True)
        table = gr.Dataframe(
            headers=pipeline.timeline_headers(),
            value=pipeline.timeline_data(),
            interactive=False,
            label="Время появления",
        )
        inp.stream(on_frame, inputs=inp, outputs=[out, table], time_limit=30)
        mode.change(on_mode, inputs=mode, outputs=table)
        reset.click(on_reset, outputs=table)
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
