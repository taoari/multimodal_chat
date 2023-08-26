# app.py : Multimodal Chatbot
import gradio as gr
import random
import os
from pprint import pprint
import mimetypes
from dotenv import load_dotenv

from utils import parse_message, format_to_message

load_dotenv()  # take environment variables from .env.

# Used for Radio, CheckboxGroup, Dropdown for convert between text and display text
TEXT2DISPLAY = { 
        'auto': 'Auto', 'random': 'Random', 'openai': 'OpenAI', # for chat engine
    }

DISPLAY2TEXT = {v:k for k,v in TEXT2DISPLAY.items()}

#################################################################################

TITLE = "Multimodal Chatbot Template"

DESCRIPTION = """
Markdown description here.
"""

SETTINGS = {
    'chat_engine': dict(cls='Radio', choices=list(map(TEXT2DISPLAY.get, ['auto', 'random', 'openai'])), value=TEXT2DISPLAY['auto'], 
            interactive=True, label="Chat engine"),
}

PARAMETERS = {
}

ATTACHMENTS = {
    # 'image': dict(cls='Image', type="filepath"),
    # 'webcam': dict(cls='Image', type="filepath", source="webcam"),
    # 'audio': dict(cls='Audio', type="filepath"),
    # 'microphone': dict(cls="Audio", type="filepath", source="microphone"),
    # 'video': dict(cls="Video"),
    # "file": dict(cls="File", type="file"),
    # 'model3d': dict(cls="Model3D"),
}

#################################################################################

CONFIG = {
    'upload_button': True,
}

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

def bot(history, *args):

    _settings = {name: value for name, value in zip(SETTINGS.keys(), args[:len(SETTINGS)])}
    _parameters = {name: value for name, value in zip(PARAMETERS.keys(), args[len(SETTINGS):])}

    # Update settings for chat mode
    # 1. convert gr.Radio and gr.CheckboxGroup from display back to text
    # 2. auto select chat engine according chat mode if it is "auto"
    _settings['chat_engine'] = DISPLAY2TEXT[_settings['chat_engine']]
    if _settings['chat_engine'] == 'auto':
        _settings['chat_engine'] = 'random'
    
    user_message = history[-1][0]
    chat_engine = _settings['chat_engine']

    try:
        # recommend to write as external functions:
        #   bot_message = <mod>.bot(user_message, **kwargs)
        bot_message = None
        if chat_engine == 'random':
            # Example multimodal messages
            bot_message = random.choice([
                format_to_message(dict(text="I love cat", images=["https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"])),
                format_to_message(dict(text="I hate cat", images=["https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/2560px-Felis_catus-cat_on_snow.jpg"])),
                format_to_message(dict(audios=["https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav"])),
                format_to_message(dict(videos=["https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"])),
                format_to_message(dict(files=["https://www.africau.edu/images/default/sample.pdf"])),
                format_to_message(dict(text="Hello, how can I assist you today?", buttons=['Primary', 'Secondary'])),
            ])
        elif chat_engine == 'openai':
            raise NotImplementedError(f"Placeholder: {chat_engine} not implemented")
        
    except Exception as e:
        bot_message = 'ERROR: ' + str(e)

    history[-1][1] = bot_message

    print(_settings); print(_parameters)
    pprint(history)
    return history

def bot_undo(history, user_message):
    if len(history) >= 1:
        user_message = history[-1][0]
        return history[:-1], user_message
    return history, user_message

def clear_chat():
    # conversation_chain.memory.clear()
    return [], "", *([None] * len(ATTACHMENTS))

def get_demo():

    def _create_from_dict(PARAMS):
        params = {}
        for name, kwargs in PARAMS.items():
            cls_ = kwargs['cls']; del kwargs['cls']
            params[name] = getattr(gr, cls_)(**kwargs)
        return params
    
    # use css and elem_id to format
    css="""#chatbot {
min-height: 600px;
}"""

    with gr.Blocks(css=css) as demo:
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        with gr.Accordion("Expand to see Introduction and Usage", open=False):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            with gr.Column(scale=1):
                attachments = _create_from_dict(ATTACHMENTS)
                with gr.Accordion("Settings", open=False) as settings_accordin:
                    settings = _create_from_dict(SETTINGS)
                with gr.Accordion("Parameters", open=False) as parameters_accordin:
                    parameters = _create_from_dict(PARAMETERS)

            with gr.Column(scale=9):
                chatbot = gr.Chatbot(elem_id='chatbot')
                with gr.Row():
                    if CONFIG['upload_button']:
                        with gr.Column(scale=0.5, min_width=30):
                            upload = gr.UploadButton("📁", file_types=["image", "video", "audio", "file"])
                    with gr.Column(scale=8):
                        # NOTE: elem_id for chatbot message buttons to work
                        msg = gr.Textbox(show_label=False,
                            placeholder="Enter text and press ENTER", container=False, elem_id="inputTextBox")
                    with gr.Column(scale=1, min_width=60):
                        submit = gr.Button(value="Submit")
                    with gr.Column(scale=1, min_width=60):
                        undo = gr.Button(value="Undo")
                    with gr.Column(scale=1, min_width=60):
                        # clear = gr.ClearButton([msg, chatbot])
                        clear = gr.Button("Clear") # also clear chatbot memory

        if CONFIG['upload_button']:
            upload.upload(user_upload_file, [msg, upload], [msg], queue=False)
        msg.submit(user, [chatbot, msg] + list(attachments.values()), [chatbot, msg] + list(attachments.values()), queue=False).then(
            bot, [chatbot] + list(settings.values()) + list(parameters.values()), chatbot,
        ).then(
            user_post, None, [msg] + list(attachments.values()), queue=False)
        submit.click(user, [chatbot, msg] + list(attachments.values()), [chatbot, msg] + list(attachments.values()), queue=False).then(
            bot, [chatbot] + list(settings.values()) + list(parameters.values()), chatbot,
        ).then(
            user_post, None, [msg] + list(attachments.values()), queue=False)
        undo.click(bot_undo, [chatbot, msg], [chatbot, msg])
        clear.click(clear_chat, [], [chatbot, msg] + list(attachments.values()))

        with gr.Accordion("Examples", open=False) as examples_accordin:
            chat_examples = gr.Examples(
                ["What's the Everett interpretation of quantum mechanics?",
                 'Give me a list of the top 10 dive sites you would recommend around the world.',
                 'Write a Python code to calculate Fibonacci numbers.'
                ],
                inputs=msg, label="AI Chat Examples",
            )
    return demo

def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Multimodal Chatbot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-p', '--port', default=7860, type=int,
        help='port number.')
    parser.add_argument(
        '--debug', action='store_true', 
        help='debug mode.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    # WARNING: gobal variables are shared accross users, and should be avoided.

    demo = get_demo()
    from utils import reload_javascript
    reload_javascript()
    demo.queue().launch(share=True, server_port=args.port)