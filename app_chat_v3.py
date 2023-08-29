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

from utils import _reformat_message
from llms import HF_ENDPOINTS
from llms import _llm_call, _llm_call_stream
from prompts import PROMPTS, split_prompt

################################################################
# Global variables
################################################################

TITLE = "AI Chat (v3)"

DESCRIPTION = """
# AI Chat

Simply enter text and press ENTER in the textbox to interact with the chatbot.
"""

ATTACHMENTS = {
    # 'image': dict(cls='Image', type='filepath'), #, source='webcam'),
    'chat_role': dict(cls='Dropdown', choices=list(PROMPTS.keys()), value="<none>", 
            interactive=True, label='Awesome ChatGPT Prompt', 
            info='Set system prompt to act as the specified role.'),
    'system_prompt': dict(cls='Textbox', interactive=True, lines=5, label="System prompt"),
}

SETTINGS = {
    'chat_engine': dict(cls='Radio', choices=['auto', 'gpt-3.5-turbo-16k', 'gpt-4'] + list(HF_ENDPOINTS.keys()), #, 'falcon-7b-instruct']
            value='auto', 
            interactive=True, label="Chat engine"),
    '_format': dict(cls='Radio', choices=['auto', 'html', 'plain', 'json'], value='auto', 
            interactive=True, label="Bot response format"),
}

PARAMETERS = {
    'temperature': dict(cls='Slider', minimum=0, maximum=1, value=0.7, step=0.1, interactive=True, label="Temperature",
            info="Lower temperator for determined output; hight temperate for randomness"),
    'max_tokens': dict(cls='Slider', minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max new tokens"),
    # 'top_k': dict(cls='Slider', minimum=1, maximum=5, value=3, step=1, interactive=True, label="Top K"),
    # 'top_p': dict(cls='Slider', minimum=0, maximum=1, value=0.9, step=0.1, interactive=True, label="Top p"),

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

################################################################
# Bot fn
################################################################


def bot_fn(message, history, *args):
    __TIC = time.time()
    kwargs = {name: value for name, value in zip(KWARGS.keys(), args)}
    kwargs['verbose'] = True # auto print llm calls

    AUTOS = {'chat_engine': 'gpt-3.5-turbo-16k'}
    # set param to default value if param is "auto"
    for param, default_value in AUTOS.items():
        kwargs[param] = default_value if kwargs[param] == 'auto' else kwargs[param]

    bot_message = _llm_call_stream(message, history, **kwargs)
    
    _format = kwargs['_format'] if '_format' in kwargs else 'auto'
    if isinstance(bot_message, str):
        yield _reformat_message(bot_message, _format=_format)
    else:
        for m in bot_message:
            yield _reformat_message(m, _format=_format)
        # bot_message = m # for print

    print(kwargs)
    __TOC = time.time()
    print(f'Elapsed time: {__TOC-__TIC}')

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
                attachments = _create_from_dict(ATTACHMENTS)
                with gr.Accordion("Settings", open=False) as settings_accordin:
                    settings = _create_from_dict(SETTINGS)
                with gr.Accordion("Parameters", open=False) as parameters_accordin:
                    parameters = _create_from_dict(PARAMETERS)

            with gr.Column(scale=9):
                # chatbot
                global KWARGS
                KWARGS = {**attachments, **settings, **parameters}
                KWARGS = {k: v for k, v in KWARGS.items() if not isinstance(v, (gr.Markdown, gr.HTML, gr.JSON))}
                import chat_interface
                chatbot = chat_interface.ChatInterface(bot_fn, # chatbot=_chatbot, textbox=_textbox,
                        additional_inputs=list(KWARGS.values()),
                        # additional_outputs=[KWARGS['session_state'], attachments['status']] if 'session_state' in KWARGS else None,
                        # upload_btn="📁",
                        retry_btn="Retry", undo_btn="Undo", clear_btn="Clear",
                    )
                chatbot.chatbot.elem_id = 'chatbot' # for css
                # chatbot.textbox.elem_id = 'inputTextBox' # for buttons
                chatbot.chatbot.avatar_images = ("assets/user.png", "assets/openai.png")

                # examples
                with gr.Accordion("Examples", open=False) as examples_accordin:
                    chat_examples = gr.Examples(
                        ["What's the Everett interpretation of quantum mechanics?",
                        'Give me a list of the top 10 dive sites you would recommend around the world.',
                        'Write a Python code to calculate Fibonacci numbers.'
                        ],
                        inputs=chatbot.textbox, label="AI Chat Examples",
                    )
            # additional handlers
            for name, attach in attachments.items():
                if hasattr(chatbot, '_upload_fn') and isinstance(attach, (gr.Image, gr.Audio, gr.Video, gr.File)):
                    attach.change(chatbot._upload_fn,
                        [chatbot.textbox, attach], 
                        [chatbot.textbox], queue=False, api_name=False)
            if 'chat_role' in attachments and 'system_prompt' in attachments:
                chat_role = attachments['chat_role']
                system_prompt = attachments['system_prompt']
                chat_role.change(lambda x: PROMPTS[x]['prompt'].strip(), chat_role, system_prompt)
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
