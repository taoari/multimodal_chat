# app.py : Multimodal Chatbot
import gradio as gr
import random
import time
import os
from pprint import pprint
import mimetypes
import langchain
from dotenv import load_dotenv

from utils import parse_message, format_to_message

load_dotenv()  # take environment variables from .env.

TITLE = "Multimodal Chat Demo"

DESCRIPTION = """
* Chat mode:
  * Random: a chatbot template to randomly generate multimodal responses
  * OpenAI: stateless raw ChatGPT
  * ChatOpenAI: stateful (w/ memory) prompt engineered ChatGPT
"""

DEFAULT_INSTRUCTIONS = """
"""

PARAMETERS = {
    'max_output_tokens': dict(cls='Slider', minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens"),
    'temperature': dict(cls='Slider', minimum=0, maximum=1, value=1, step=0.1, interactive=True, label="Temperature"),
    'top_k': dict(cls='Slider', minimum=1, maximum=5, value=3, step=1, interactive=True, label="Top K"),
    'top_p': dict(cls='Slider', minimum=0, maximum=1, value=0.9, step=0.1, interactive=True, label="Top p"),
}

ATTACHMENTS = {
    'image': dict(cls='Image', type="filepath"),
    'webcam': dict(cls='Image', type="filepath", source="webcam"),
    'audio': dict(cls='Audio', type="filepath"),
    'microphone': dict(cls="Audio", type="filepath", source="microphone"),
    'video': dict(cls="Video"),
    "file": dict(cls="File", type="file"),
    # 'model3d': dict(cls="Model3D"),
}

CONFIG = {
    'upload_button': True,
}

if CONFIG['upload_button']:
    ATTACHMENTS = {}

def user(history, msg, *attachments):
    _attachments = {name: filepath for name, filepath in zip(ATTACHMENTS.keys(), attachments)}
    print(_attachments)
    msg_dict = dict(text=msg, images=[], audios=[], videos=[], files=[])
    for name, filepath in _attachments.items():
        if filepath is not None:
            if name in ['image', 'webcam']:
                msg_dict['images'].append(filepath)
            elif name in ['audio', 'microphone']:
                msg_dict['audios'].append(filepath)
            elif name in ['video']:
                msg_dict['videos'].append(filepath)
            else:
                msg_dict['files'].append(filepath)
    return history + [[format_to_message(msg_dict), None]], gr.update(value="", interactive=False), \
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
        msg += f'<img src="\\file={filepath.name}" alt="{os.path.basename(filepath.name)}"/>'
    elif mtype.startswith('audio'):
        msg += f'<audio controls><source src="\\file={filepath.name}">{os.path.basename(filepath.name)}</audio>'
    elif mtype.startswith('video'):
        msg += f'<video controls><source src="\\file={filepath.name}">{os.path.basename(filepath.name)}</video>'
    else:
        msg += f'<a href="\\file={filepath.name}">📁 {os.path.basename(filepath.name)}</a>'
    return msg

def bot(history, instructions, chat_mode, *parameters):
    user_message = history[-1][0]
    if chat_mode.startswith('OpenAI'):
        bot_message = conversation_chain.predict(input=user_message)
    elif chat_mode.startswith('ChatOpenAI'):
        bot_message = llm_chain.run(input=user_message)
    elif chat_mode.startswith('StabilityAI'):
        import stability_ai
        msg_dict = parse_message(user_message)
        img = stability_ai.generate(msg_dict["text"], 
            msg_dict["images"][-1] if "images" in msg_dict and len(msg_dict["images"]) > 1 else None)
        if img is not None:
            import tempfile
            fname = tempfile.NamedTemporaryFile(prefix='gradio/stability_ai-', suffix='.png').name
            img.save(fname)
            bot_message = format_to_message(dict(images=[fname]))
        else:
            bot_message = "Sorry, stability.ai failed to generate image."
    else:
        # Example multimodal messages
        bot_message = random.choice([
            format_to_message(dict(text="I love cat", images=["https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"])),
            format_to_message(dict(text="I hate cat", images=["https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/2560px-Felis_catus-cat_on_snow.jpg"])),
            format_to_message(dict(audios=["https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav"])),
            format_to_message(dict(videos=["https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"])),
            format_to_message(dict(files=["https://www.africau.edu/images/default/sample.pdf"])),
         ])
    # # streaming
    # history[-1][1] = ""
    # for character in bot_message:
    #     history[-1][1] += character
    #     time.sleep(0.05)
    #     yield history

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

def get_demo():
    with gr.Blocks() as demo:
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        with gr.Accordion("Description", open=True):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            with gr.Column(scale=1):
                attachments = {}
                for name, kwargs in ATTACHMENTS.items():
                    cls_ = kwargs['cls']; del kwargs['cls']
                    attachments[name] = getattr(gr, cls_)(**kwargs)

                with gr.Accordion("Chat mode", open=True) as chat_mode_accordin:
                    chat_mode = gr.Radio(["Random", "OpenAI", "ChatOpenAI", "StabilityAI"], value='ChatOpenAI', show_label=False, 
                        info="")
                
                parameters = {}
                with gr.Accordion("Parameters", open=False) as parameters_accordin:
                    for name, kwargs in PARAMETERS.items():
                        cls_ = kwargs['cls']; del kwargs['cls']
                        parameters[name] = getattr(gr, cls_)(**kwargs)

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
                    if CONFIG['upload_button']:
                        with gr.Column(scale=0.5, min_width=30):
                            upload = gr.UploadButton("📁") #, file_types=["image", "video", "audio", "file"])
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

    return demo


if __name__ == '__main__':
    from langchain.llms import HuggingFaceTextGenInference
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import ConversationChain, LLMChain

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
    demo.queue().launch(share=True)