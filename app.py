# app.py : Multimodal Chatbot
import gradio as gr
import random
import time
import os
from pprint import pprint
import mimetypes
import langchain
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

DEBUG = True

TITLE = "Multimodal Chat Demo"

DESCRIPTION = """

# This demo shows how to scan a barcode and search in Walgreens catalog. 

* HOW TO USE: Put a product barcode in front of the webcam or upload an image. Then Click the "Submit" button to test.

* NOTE: You may unfold the "Captured Image" (if `--stream` is enabled) to check on the barcode detected frame. But I would recommand to fold it since it is annoying for blinking.

"""

DEFAULT_INSTRUCTIONS = """
"""

PARAMETERS = {}

ATTACHMENTS = {}

CONFIG = {
    'upload_button': True,
}

def user(history, msg, *attachments):
    _attachments = {name: filepath for name, filepath in zip(ATTACHMENTS.keys(), attachments)}
    print(_attachments)
    for name, filepath in _attachments.items():
        if filepath is not None:
            if name in ['image', 'webcam']:
                msg += f'\n<img src="\\file={filepath}" alt="{os.path.basename(filepath)}"/>'
    return history + [[msg, None]], gr.update(value="", interactive=False), \
        *([gr.update(value=None, interactive=False)] * len(attachments))

def user_post():
    if len(ATTACHMENTS) == 0:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=True), *([gr.update(interactive=True)] * len(ATTACHMENTS))

def user_upload_file(msg, filepath):
    # history = history + [((file.name,), None)]
    mtype = mimetypes.guess_type(filepath.name)[0]
    if mtype.startswith('image'):
        msg += f'\n<img src="\\file={filepath.name}" alt="{os.path.basename(filepath.name)}"/>'
    return msg

def bot(history, instructions, chat_mode, *parameters):
    user_message = history[-1][0]
    if chat_mode == 'Stateful':
        bot_message = conversation_chain.predict(input=user_message)
    elif chat_mode == 'Stateless':
        bot_message = llm_chain.run(input=user_message)
    elif chat_mode == "Barcode":
        import barcode
        bot_message = barcode.bot(user_message, DEBUG=DEBUG)
    else:
        bot_message = f"Unknow chat mode: {chat_mode}"

    if isinstance(bot_message, str):
        history[-1][1] = bot_message
    else:
        history[-1][1] = bot_message[0]
        history.extend([(None, msg) for msg in bot_message[1:]])
    print(chat_mode)
    print({name: value for name, value in zip(PARAMETERS.keys(), parameters)})
    pprint(history)
    return history

def clear_chat():
    conversation_chain.memory.clear()
    return [], "", *([None] * len(ATTACHMENTS))

def barcode_capture(webcam, output):
    global CONFIG
    from barcode import get_barcodes_pil
    res, img = get_barcodes_pil(webcam, draw=False)

    if "barcode" not in CONFIG:
        CONFIG["barcode"] = ""
        CONFIG["barcode_image"] = webcam

    if res is not None and len(res) > 0:
        _bar = res[-1]
        if CONFIG["barcode"] != _bar:
            CONFIG["barcode"] = _bar
            CONFIG["barcode_image"] = webcam
        
    return CONFIG["barcode_image"]

def get_demo():
    global ATTACHMENTS
    with gr.Blocks() as demo:
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        with gr.Accordion("Description", open=True):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            attachments = {}
            parameters = {}
            chat_mode = gr.State(value="Barcode")
            instructions = gr.State()
            with gr.Column(scale=1):
                if args.stream:
                    webcam_stream = gr.Image(type="pil", source="webcam", streaming=True)
                    with gr.Accordion("Captured Image", open=False) as _accordin:
                        attachments = {'image': gr.Image(type="filepath")}
                else:
                    attachments = {'webcam': gr.Image(type="filepath", source="webcam")}
            ATTACHMENTS = attachments

            with gr.Column(scale=9):
                chatbot = gr.Chatbot()
                with gr.Row():
                    if CONFIG['upload_button']:
                        with gr.Column(scale=0.5, min_width=30):
                            upload = gr.UploadButton("📁", file_types=["image", "video", "audio"])
                    with gr.Column(scale=8):
                        msg = gr.Textbox(show_label=False,
                            placeholder="Enter text and press ENTER").style(container=False)
                    with gr.Column(scale=1, min_width=60):
                        submit = gr.Button(value="Submit")
                    with gr.Column(scale=1, min_width=60):
                        # clear = gr.ClearButton([msg, chatbot])
                        clear = gr.Button("Clear") # also clear chatbot memory

        if CONFIG['upload_button']:
            upload.upload(user_upload_file, [msg, upload], [msg], queue=False)
        msg.submit(user, [chatbot, msg] + list(attachments.values()), [chatbot, msg] + list(attachments.values()), queue=False).then(
            bot, [chatbot, instructions, chat_mode] + list(parameters.values()), chatbot
        ).then(
            user_post, None, [msg] + list(attachments.values()), queue=False)
        submit.click(user, [chatbot, msg] + list(attachments.values()), [chatbot, msg] + list(attachments.values()), queue=False).then(
            bot, [chatbot, instructions, chat_mode] + list(parameters.values()), chatbot
        ).then(
            user_post, None, [msg] + list(attachments.values()), queue=False)
        clear.click(clear_chat, [], [chatbot, msg] + list(attachments.values()))

        if args.stream:
            webcam_stream.change(barcode_capture, inputs=webcam_stream, outputs=attachments['image'])

    return demo


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--port", default=7860, type=int,
                        help="port number")
    parser.add_argument("--debug", action='store_true',
                        help="run in debug mode")
    parser.add_argument("--stream", action='store_true',
                        help="enable webcam streaming")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    from langchain.llms import HuggingFaceTextGenInference
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import ConversationChain, LLMChain

    args = parse_args()
    # DEBUG = args.debug # Always in debug mode now

    # llm = HuggingFaceTextGenInference(
    #     inference_server_url="https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
    # )

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        stop=['\nHuman:', '<|endoftext|>'],
    )

    print(repr(llm)) # NOTE: prints private tokens

    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{input}"), verbose=True)
    conversation_chain = ConversationChain(llm=llm, verbose=True)

    demo = get_demo()
    demo.queue().launch(share=True, server_port=args.port)