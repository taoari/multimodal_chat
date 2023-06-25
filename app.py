# app.py : Multimodal Chatbot
import gradio as gr
import random
import time
from pprint import pprint


TITLE = "Chat Demo"

DESCRIPTION = """Description in Markdown
"""

DEFAULT_INSTRUCTIONS = """
"""

PARAMETERS = {
    'max_output_tokens': dict(type='Slider', minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens"),
    'temperature': dict(type='Slider', minimum=0, maximum=1, value=1, step=0.1, interactive=True, label="Temperature"),
    'top_k': dict(type='Slider', minimum=1, maximum=5, value=3, step=1, interactive=True, label="Top K"),
    'top_p': dict(type='Slider', minimum=0, maximum=1, value=0.9, step=0.1, interactive=True, label="Top p"),
}

CONFIG = {
    'upload_button': False,
}

def user(user_message, history):
    return gr.update(value="", interactive=False), history + [[user_message, None]]

def user_upload_file(history, file):
    history = history + [((file.name,), None)]
    return history

def bot(history, instructions, *parameters):
    bot_message = random.choice(["How are you?", 
        'I love cat <img src="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg" alt="Italian Trulli">', 
        'I hate cat ![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/2560px-Felis_catus-cat_on_snow.jpg)',
        "I'm **very hungry**", 
        ("https://upload.wikimedia.org/wikipedia/commons/5/53/Sheba1.JPG",),
        ("https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav",),
        ("https://www.africau.edu/images/default/sample.pdf",),
        ("https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj")])
    # history[-1][1] = ""
    # for character in bot_message:
    #     history[-1][1] += character
    #     time.sleep(0.05)
    #     yield history
    history[-1][1] = bot_message
    print({name: value for name, value in zip(PARAMETERS.keys(), parameters)})
    pprint(history)
    return history

def get_demo():
    with gr.Blocks() as demo:
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        with gr.Accordion("Description", open=False):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image()
                audio = gr.Audio(source="microphone")
                # video = gr.Video()
                file = gr.File()
                treeD = gr.Model3D()

                parameters = {}
                with gr.Accordion("Parameters", open=False) as parameters_accordin:
                    for name, kwargs in PARAMETERS.items():
                        type_ = kwargs['type']; del kwargs['type']
                        parameters[name] = getattr(gr, type_)(**kwargs)

                with gr.Accordion("Instructions", open=False) as instructions_accordin:
                    instructions = gr.Textbox(
                        placeholder="LLM instructions",
                        value=DEFAULT_INSTRUCTIONS,
                        show_label=False,
                        interactive=True,
                    ).style(container=False)

            with gr.Column(scale=9):
                chatbot = gr.Chatbot()
                with gr.Row():
                    with gr.Column(scale=8):
                        msg = gr.Textbox(show_label=False,
                            placeholder="Enter text and press ENTER").style(container=False)
                    with gr.Column(scale=1, min_width=60):
                        if CONFIG['upload_button']:
                            upload = gr.UploadButton("📁") #, file_types=["image", "video", "audio", "file"])
                        else:
                            submit = gr.Button(value="Submit")
                    with gr.Column(scale=1, min_width=60):
                        clear = gr.ClearButton([msg, chatbot])

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, instructions] + list(parameters.values()), chatbot
        ).then(lambda: gr.update(interactive=True), None, [msg], queue=False)
        if CONFIG['upload_button']:
            upload.upload(user_upload_file, [chatbot, upload], [chatbot], queue=False)
            # upload.upload(user_upload_file, [chatbot, btn], [chatbot], queue=False).then(
            #     bot, [chatbot, instructions] + list(parameters.values()), chatbot
            # )
        else:
            submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [chatbot, instructions] + list(parameters.values()), chatbot
            ).then(lambda: gr.update(interactive=True), None, [msg], queue=False)

    return demo

if __name__ == '__main__':
    demo = get_demo()
    demo.queue().launch()