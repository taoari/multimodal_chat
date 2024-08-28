# app.py : Multimodal Chatbot
import os
import time
import logging
import jinja2
import gradio as gr

################################################################
# Load .env and logging
################################################################

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN, format='%(asctime)-15s] %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p %Z")

def print(*args, **kwargs):
    sep = kwargs['sep'] if 'sep' in kwargs else ' '
    logger.warning(sep.join([str(val) for val in args])) # use level WARN for print, as gradio level INFO print unwanted messages

from utils import llms
llms.print = print

################################################################
# Globals
################################################################

_default_session_state = dict(
        context_switch_at=0, # history before context_switch_at should be ignored (e.g. upload an image or a file)
        message=None,
        previous_message=None,
    )

TITLE = """ Gradio Multimodal Chatbot Template """

DESCRIPTION = """Welcome
"""

SETTINGS = {
    'Info': {
        '__metadata': {'open': False, 'tabbed': True},
        'image': dict(cls='Image', type='filepath', label="Input"), #, source='webcam'),
        'system_prompt': dict(cls='Textbox', interactive=True, lines=5, label="System prompt"),
        'status': dict(cls='JSON', label='Status'),
    },
    'Settings': {
        '__metadata': {'open': False, 'tabbed': False},
        'session_state': dict(cls='State', value=_default_session_state),
        'chat_engine': dict(cls='Radio', choices=['auto', 'random', 'gpt-3.5-turbo'], value='auto', 
                interactive=True, label="Chat engine"),
        'speech_synthesis': dict(cls='Checkbox', value=False, 
                interactive=True, label="Speech Synthesis"),
    },
    'Parameters': {
        '__metadata': {'open': True, 'tabbed': False},
        'temperature': dict(cls='Slider', minimum=0, maximum=1, value=0.7, step=0.1, interactive=True, label="Temperature")
    }
}

# SETTINGS = {}

################################################################
# Utils
################################################################

def _create_from_dict(PARAMS, tabbed=False):
    params = {}
    for name, kwargs in PARAMS.items():
        if name.startswith('__'):
            continue
        cls_ = kwargs['cls']; del kwargs['cls']
        if not tabbed:
            params[name] = getattr(gr, cls_)(**kwargs)
        else:
            tab_name = kwargs['label'] if 'label' in kwargs else name
            with gr.Tab(tab_name):
                params[name] = getattr(gr, cls_)(**kwargs)
    return params

def _clear(session_state):
    session_state.clear()
    session_state.update(_default_session_state)
    return session_state

def transcribe(audio=None):
    try:
        from utils.azure_speech import speech_recognition
        return speech_recognition(audio)
    except Exception as e:
        return f"Microphone is not supported: {e}"

################################################################
# Bot fn
################################################################

from utils.llms import _llm_call, _llm_call_stream, _random_bot_fn
bot_fn = _llm_call_stream

################################################################
# Demo
################################################################

def get_demo():
    css="""#chatbot {
    min-height: 600px;
    }
    .full-container label {
    display: block;
    padding-left: 8px;
    padding-right: 8px;
    }"""
    with gr.Blocks(css=css) as demo:
        # title
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        # description
        with gr.Accordion("Expand to see Introduction and Usage", open=False):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            with gr.Column(scale=1):
                # settings
                for section_name, _settings in SETTINGS.items():
                    metadata = _settings['__metadata']
                    with gr.Accordion(section_name, open=metadata.get('open', False)):
                        settings = _create_from_dict(_settings, tabbed=metadata.get('tabbed', False))
            with gr.Column(scale=9):
                # chatbot
                from utils.gradio import ChatInterface
                chatbot = ChatInterface(bot_fn, type='messages', 
                        multimodal=False,
                        avatar_images=('assets/user.png', 'assets/bot.png'))
                chatbot.audio_btn.click(transcribe, [], [chatbot.textbox], queue=False, api_name=False)
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

    demo = get_demo()
    from utils.gradio import reload_javascript
    reload_javascript()
    demo.queue().launch(server_name='0.0.0.0', server_port=args.port)
