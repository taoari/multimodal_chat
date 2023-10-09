# app.py : Multimodal Chatbot
import os
import time
import logging
from dotenv import load_dotenv
import gradio as gr

################################################################
# Load .env and logging
################################################################

load_dotenv()  # take environment variables from .env.

logging.basicConfig(level=logging.WARN, format='%(asctime)-15s] %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p %Z")

def print(*args, **kwargs):
    sep = kwargs['sep'] if 'sep' in kwargs else ' '
    logging.warning(sep.join([str(val) for val in args])) # use level WARN for print, as gradio level INFO print unwanted messages

################################################################
# Extra loading
################################################################

from utils import _reformat_message, _reformat_history, format_to_message
from llms import _random_bot_fn, _openai_stream_bot_fn

DEBUG = True

# One can assume that keys of _default_session_state always exist
_default_session_state = dict(
        context_switch_at=0, # history before context_switch_at should be ignored (e.g. upload an image or a file)
        message=None,
        previous_message=None,
    )

################################################################
# Global variables
################################################################

TITLE = "Walgreens Janus AI Chatbot"

DESCRIPTION = """
Markdown description here.
"""

ATTACHMENTS = {
    'image': dict(cls='Image', type='filepath', label="Input"), #, source='webcam'),
    'system_prompt': dict(cls='Textbox', interactive=True, lines=5, label="System prompt"),
    'status': dict(cls='JSON', label='Status info'),
}

SETTINGS = {
    'session_state': dict(cls='State', value=_default_session_state),
    'chat_engine': dict(cls='Radio', choices=['auto', 'rasa', 'gpt-3.5-turbo'], value='auto', 
            interactive=True, label="Chat engine"),
    '_format': dict(cls='Radio', choices=['auto', 'html', 'plain', 'json'], value='auto', 
            interactive=True, label="Bot response format"),
}

PARAMETERS = {
    'sender_id': dict(cls='Textbox', value="default", label="Sender ID"),
    'generate_sender_id': dict(cls='Button', value="Generate", container=False, show_label=False),
    'server_url': dict(cls='Textbox', value='http://172.23.72.43:5005/webhooks/rest/webhook', label='Server URL', lines=2),
}
    
KWARGS = {} # use for chatbot additional_inputs, do NOT change
    
################################################################
# utils
################################################################

def _create_from_dict(PARAMS, tabbed=False):
    params = {}
    for name, kwargs in PARAMS.items():
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

from collections import defaultdict

def list_of_dicts_to_dict_of_list(list_of_dicts):
    dict_of_list = defaultdict(list)
    
    for d in list_of_dicts:
        for key, value in d.items():
            dict_of_list[key].append(value)
    
    return dict(dict_of_list)

def format_rasa_to_message(rasa_json):
    res = list_of_dicts_to_dict_of_list(rasa_json)
    buttons = []
    if 'buttons' in res:
        for btns in res['buttons']:
            for btn in btns:
                buttons.append({'text': btn['title'], 'value': btn['payload']})
    res = {'text': '<br />'.join(res['text']), 
           'images': res['image'] if 'image' in res else [],
           'buttons': buttons,
           }
    return format_to_message(res)

def generate_sender_id():
    import uuid
    return str(uuid.uuid4())

################################################################
# Bot fn
################################################################

def _echo_bot_fn(message, history, **kwargs):
    return message

def _rasa_bot_fn(message, history, **kwargs):
    import requests
    import json

    url = kwargs['server_url'] # 'http://172.23.72.43:5005/webhooks/rest/webhook'

    # Define the JSON data
    data = {"sender": kwargs['sender_id'], "message": message}

    # Set the headers
    headers = {'Content-Type': 'application/json'}

    # Send the POST request
    response = requests.post(url, json=data, headers=headers)
    return format_rasa_to_message(response.json())

def bot_fn(message, history, *args):
    __TIC = time.time()
    kwargs = {name: value for name, value in zip(KWARGS.keys(), args)}
    kwargs['verbose'] = True # auto print llm calls
    session_state = kwargs['session_state']
    if len(history) == 0 or message == '/clear':
        _clear(session_state)
    # unformated LLM history for rich response applications, keep only after latest context switch
    history = _reformat_history(history[session_state['context_switch_at']:])
    plain_message = _reformat_message(message)

    """ BEGIN: Update only this part if necessary """

    AUTOS = {'chat_engine': 'rasa'}
    # set param to default value if param is "auto"
    for param, default_value in AUTOS.items():
        kwargs[param] = default_value if kwargs[param] == 'auto' else kwargs[param]

    bot_message = {'random': _random_bot_fn,
        'echo': _echo_bot_fn,
        'gpt-3.5-turbo': _openai_stream_bot_fn,
        'rasa': _rasa_bot_fn,
        }.get(kwargs['chat_engine'])(message, history, **kwargs)
    
    session_state['message'] = message
    status = {**kwargs} # session_state, settings, and elapsed_time

    """ END: Update only this part if necessary """
    
    # NOTE: _reformat_message could double check parse_message and format_to_message integrity
    _format = kwargs['_format'] if '_format' in kwargs else 'auto'
    if isinstance(bot_message, str):
        __TOC = time.time(); status['elapsed_time'] = __TOC - __TIC
        yield _reformat_message(bot_message, _format=_format), session_state, status
    else:
        for m in bot_message:
            __TOC = time.time(); status['elapsed_time'] = __TOC - __TIC
            yield _reformat_message(m, _format=_format), session_state, status
        bot_message = m # for print

    __TOC = time.time()
    print(f'Elapsed time: {__TOC-__TIC}')
    session_state['previous_message'] = message

################################################################
# Gradio app
################################################################

def get_demo():

    # use css and elem_id to format
    css="""#chatbot {
min-height: 600px;
}"""
    with gr.Blocks(css=css) as demo:
        # title
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        # description
        with gr.Accordion("Expand to see Introduction and Usage", open=False):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            # attachments, settings, and parameters
            with gr.Column(scale=1):
                attachments = _create_from_dict(ATTACHMENTS, tabbed=True)
                with gr.Accordion("Settings", open=False) as settings_accordin:
                    settings = _create_from_dict(SETTINGS)
                with gr.Accordion("Parameters", open=True) as parameters_accordin:
                    parameters = _create_from_dict(PARAMETERS)

            with gr.Column(scale=9):
                # chatbot
                global KWARGS
                KWARGS = {**attachments, **settings, **parameters}
                KWARGS = {k: v for k, v in KWARGS.items() if not isinstance(v, (gr.Markdown, gr.HTML, gr.JSON))}
                import chat_interface
                chatbot = chat_interface.ChatInterface(bot_fn, # chatbot=_chatbot, textbox=_textbox,
                        additional_inputs=list(KWARGS.values()),
                        additional_outputs=[KWARGS['session_state'], attachments['status']] if 'session_state' in KWARGS else None,
                        upload_btn="📁",
                        retry_btn="Retry", undo_btn="Undo", clear_btn="Clear",
                    )
                chatbot.chatbot.elem_id = 'chatbot' # for css
                chatbot.textbox.elem_id = 'inputTextBox' # for buttons
                chatbot.chatbot.avatar_images = ("assets/user.png", "assets/bot.png")

                # examples
                with gr.Accordion("Examples", open=False) as examples_accordin:
                    chitchat_examples = gr.Examples(
                        ["Hello"],
                        inputs=chatbot.textbox, label="Chitchat Examples",
                    )
                    store_info_examples = gr.Examples(
                        ["Can you help me to find the nearest store?",
                         "980055", "98005", "/affirm",
                        ],
                        inputs=chatbot.textbox, label="Store Info Examples",
                    )
                    product_search_examples = gr.Examples(
                        ["Can I buy some toothpaste?",
                        #  "What kind of non-drowsy allergy meds does Walgreens carry?",
                        ],
                        inputs=chatbot.textbox, label="Product Search Examples",
                    )
                    otc_faq_examples = gr.Examples(
                        ["What is the active ingredient in the anti-diarrheal caplets?",
                         "The barcode is 31191705594", '/affirm', "What are the warnings of this product?", '/deny',
                        ],
                        inputs=chatbot.textbox, label="OTC FAQ Examples",
                    )
            # additional handlers
            for name, attach in attachments.items():
                if hasattr(chatbot, '_upload_fn') and isinstance(attach, (gr.Image, gr.Audio, gr.Video, gr.File)):
                    attach.change(chatbot._upload_fn,
                        [chatbot.textbox, attach], 
                        [chatbot.textbox], queue=False, api_name=False)
            parameters['generate_sender_id'].click(generate_sender_id, [], [parameters['sender_id']], api_name=False)
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
    from utils import reload_javascript
    reload_javascript()
    demo.queue().launch(server_name='0.0.0.0', server_port=args.port)
