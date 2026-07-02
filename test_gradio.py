import gradio as gr
import tempfile
import os

f1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
f2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
with open(f1, 'w') as f: f.write("a")
with open(f2, 'w') as f: f.write("b")

with gr.Blocks() as demo:
    f = gr.File(value=[f1, f2], label="Upload", file_count="multiple", elem_classes=["my-file"])

demo.launch(prevent_thread_lock=True)
import time
time.sleep(2)
