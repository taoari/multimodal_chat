# app.py : Multimodal Chatbot
import gradio as gr
import random
import time
import os
from pprint import pprint
import langchain
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


TITLE = "Multimodal Chat Demo"

DESCRIPTION = """Description in Markdown
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
    'audio': dict(cls='Audio', type="filepath"),
    'microphone': dict(cls="Audio", type="filepath", source="microphone"),
    'video': dict(cls="Video"),
    "file": dict(cls="File", type="file"),
    # 'model3d': dict(cls="Model3D"),
}

CONFIG = {
    'upload_button': False,
}

def user(history, msg, *attachments):
    _attachments = {name: filepath for name, filepath in zip(ATTACHMENTS.keys(), attachments)}
    print(_attachments)
    for name, filepath in _attachments.items():
        if filepath is not None:
            if name in ['image']:
                msg += f'\n<img src="\\file={filepath}" alt="{os.path.basename(filepath)}"/>'
            elif name in ['audio', 'microphone']:
                msg += f'\n<audio controls><source src="\\file={filepath}">{os.path.basename(filepath)}</audio>'
            elif name in ['video']:
                msg += f'\n<video controls><source src="\\file={filepath}">{os.path.basename(filepath)}</video>'
            else:
                msg += f'\n<a href="\\file={filepath.name}">📁 {os.path.basename(filepath.name)}</a>'
    return history + [[msg, None]], gr.update(value="", interactive=False), \
        *([gr.update(value=None, interactive=False)] * len(attachments))

def user_post():
    return gr.update(interactive=True), *([gr.update(interactive=True)] * len(ATTACHMENTS))

def user_upload_file(history, file):
    history = history + [((file.name,), None)]
    return history

def bot(history, instructions, chat_mode, *parameters):
    user_message = history[-1][0]
    if chat_mode == 'Stateful':
        bot_message = conversation_chain.predict(input=user_message)
    elif chat_mode == 'Stateless':
        bot_message = llm_chain.run(input=user_message)
    else:
        # Example multimodal messages
        bot_message = random.choice([
            'I love cat <img src="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg" alt="Italian Trulli">', 
            'I hate cat ![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/2560px-Felis_catus-cat_on_snow.jpg)',
            # "I'm **very hungry**", 
            # ("https://upload.wikimedia.org/wikipedia/commons/5/53/Sheba1.JPG",),
            ("https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav",),
            ("https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",),
            # ("https://www.africau.edu/images/default/sample.pdf",), # files are not shown, use HTML
            '<a href="https://www.africau.edu/images/default/sample.pdf">📁 sample.pdf</a>',
            # ("https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj"),
            '<a href="https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj">📁 stanford-bunny.obj</a>',
        ])
    # # streaming
    # history[-1][1] = ""
    # for character in bot_message:
    #     history[-1][1] += character
    #     time.sleep(0.05)
    #     yield history

    history[-1][1] = bot_message
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
        with gr.Accordion("Description", open=False):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            with gr.Column(scale=1):
                attachments = {}
                for name, kwargs in ATTACHMENTS.items():
                    cls_ = kwargs['cls']; del kwargs['cls']
                    attachments[name] = getattr(gr, cls_)(**kwargs)

                with gr.Accordion("Chat mode", open=True) as chat_mode_accordin:
                    chat_mode = gr.Radio(["Random", "Stateless", "Stateful"], value='Stateful', show_label=False, 
                        info="use memory (stateful) or not (stateless)")
                
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
                    with gr.Column(scale=8):
                        msg = gr.Textbox(show_label=False,
                            placeholder="Enter text and press ENTER").style(container=False)
                    with gr.Column(scale=1, min_width=60):
                        if CONFIG['upload_button']:
                            upload = gr.UploadButton("📁") #, file_types=["image", "video", "audio", "file"])
                        else:
                            submit = gr.Button(value="Submit")
                    with gr.Column(scale=1, min_width=60):
                        # clear = gr.ClearButton([msg, chatbot])
                        clear = gr.Button("Clear") # also clear chatbot memory

        msg.submit(user, [chatbot, msg] + list(attachments.values()), [chatbot, msg] + list(attachments.values()), queue=False).then(
            bot, [chatbot, instructions, chat_mode] + list(parameters.values()), chatbot
        ).then(
            user_post, None, [msg] + list(attachments.values()), queue=False)
        if CONFIG['upload_button']:
            upload.upload(user_upload_file, [chatbot, upload], [chatbot], queue=False)
            # upload.upload(user_upload_file, [chatbot, btn], [chatbot], queue=False).then(
            #     bot, [chatbot, instructions] + list(parameters.values()), chatbot
            # )
        else:
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
    demo.queue().launch()