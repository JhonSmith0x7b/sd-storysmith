import gradio as gr
from PIL import Image
import numpy as np
import os


def init() -> None:
    global BATCH_SIZE, OUTPUT_DIR, EPOCH
    BATCH_SIZE = 4
    OUTPUT_DIR = "./output"
    EPOCH = 0
    

def save_images(checkbox_group: list[str], *imgs: np.ndarray) -> None:
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    for i, img in enumerate(imgs):
        if f"Image{i}" in checkbox_group:
            image = Image.fromarray(img)
            image.save(open(f"{OUTPUT_DIR}/{EPOCH}_{i}.png", 'wb'), "PNG")


def main() -> None:
    with gr.Blocks() as app:
        gr.Label(f"Epoch {EPOCH}")
        images = []
        positive = gr.Textbox(label="Positive Prompt")
        negative = gr.Textbox(label="Negative Prompt")
        with gr.Row():
            checkbox_group = gr.CheckboxGroup([f"Image{i}" for i in range(BATCH_SIZE)], show_label=False),
            btn_recreate = gr.Button("Recreate")
            btn_save = gr.Button("Save")
            btn_next = gr.Button("Next")
        with gr.Row():
            for i in range(BATCH_SIZE):
                images.append(gr.Image(f"temp_{i}.png", width="25vw", label=f"Image{i}"))
        btn_save.click(fn=save_images, inputs=[checkbox_group[0], *images])
    app.launch()


if __name__ == '__main__':
    init()
    main()