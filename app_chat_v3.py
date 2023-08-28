# app.py : Multimodal Chatbot
import os
import time
import logging
from pprint import pprint
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

from llms import HF_ENDPOINTS
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

def _create_from_dict(PARAMS):
    params = {}
    for name, kwargs in PARAMS.items():
        cls_ = kwargs['cls']; del kwargs['cls']
        params[name] = getattr(gr, cls_)(**kwargs)
    return params

from llms import _bot_slash_fn, _llm_call, _llm_call_stream, _llm_call_langchain

def bot_fn(message, history, *args):
    __TIC = time.time()
    kwargs = {name: value for name, value in zip(KWARGS.keys(), args)}
    kwargs['chat_engine'] = 'gpt-3.5-turbo-16k' if kwargs['chat_engine'] == 'auto' else kwargs['chat_engine']

    if message.startswith('/'):
        bot_message = _bot_slash_fn(message, history, **kwargs)
    else:
        # bot_message = _llm_call(message, history, **kwargs)
        bot_message = _llm_call_stream(message, history, **kwargs)
        # bot_message = _llm_call_langchain(message, history, **kwargs)
    
    if isinstance(bot_message, str):
        yield bot_message
    else:
        for m in bot_message:
            yield m
        # bot_message = m # for printing
    __TOC = time.time()
    print(f'Elapsed time: {__TOC-__TIC}')
    print(kwargs)

################################################################
# Gradio app
################################################################

def get_demo():

    # use css and elem_id to format
    css="""#chatbot {
min-height: 600px;
}"""

    # NOTE: can not be inside another gr.Blocks
    # _chatbot = gr.Chatbot(elem_id="chatbot", avatar_images = ("assets/user.png", "assets/bot.png"))
    # _textbox = gr.Textbox(container=False, show_label=False, placeholder="Type a message...", scale=10, elem_id='inputTextBox', min_width=300)

    with gr.Blocks(css=css) as demo:
        # title
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        # description
        with gr.Accordion("Expand to see Introduction and Usage", open=False):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            # attachements, settings, and parameters
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
                import chat_interface
                chatbot = chat_interface.ChatInterface(bot_fn, # chatbot=_chatbot, textbox=_textbox,
                        additional_inputs=list(KWARGS.values()),
                        # additional_outputs=[KWARGS['session_state']] if 'session_state' in KWARGS else None,
                        # upload_btn=None,
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
                if hasattr(chatbot, '_upload_fn') and isinstance(attach, gr.Image):
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
    demo.queue().launch(share=True, server_port=args.port)
