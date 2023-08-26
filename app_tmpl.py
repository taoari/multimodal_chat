# app.py : Multimodal Chatbot
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
# Global variables
################################################################

TITLE = "Multimodal Chatbot Template"

DESCRIPTION = """
Markdown description here. Features:

* Upload button (auto displayed)
* Session state (additional outputs)
* Chatbot messages:
  * Avatar images (auto displayed)
  * Buttons (type in "button")
  * Cards (type in "card")
"""

ATTACHMENTS = {
    'image': dict(cls='Image', type='filepath'), #, source='webcam'),
    'system_prompt': dict(cls='Textbox', interactive=True, lines=5, label="System prompt"),
}

SETTINGS = {
    'session_state': dict(cls='State', value={}),
    'chat_engine': dict(cls='Radio', choices=['auto', 'random', 'openai', 'openai_stream'], value='auto', 
            interactive=True, label="Chat engine"),
}

PARAMETERS = {
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


from app_tmpl_bot_fn import _random_bot_fn, _openai_bot_fn, _openai_stream_bot_fn

def _bot_fn(message, history, *args):
    kwargs = {name: value for name, value in zip(KWARGS.keys(), args)}

    kwargs['chat_engine'] = 'random' if kwargs['chat_engine'] == 'auto' else kwargs['chat_engine']

    bot_message = {'random': _random_bot_fn,
        'openai': _openai_bot_fn,
        'openai_stream': _openai_stream_bot_fn,
        }.get(kwargs['chat_engine'])(message, history, **kwargs)
    
    if isinstance(bot_message, str):
        yield bot_message
    else:
        for m in bot_message:
            yield m

    print(kwargs)
    pprint(history + [[message, bot_message]])

def _bot_fn_session_state(message, history, *args):
    kwargs = {name: value for name, value in zip(KWARGS.keys(), args)}

    session_state = kwargs['session_state']
    kwargs['chat_engine'] = 'random' if kwargs['chat_engine'] == 'auto' else kwargs['chat_engine']

    bot_message = {'random': _random_bot_fn,
        'openai': _openai_bot_fn,
        'openai_stream': _openai_stream_bot_fn,
        }.get(kwargs['chat_engine'])(message, history, **kwargs)
    
    if isinstance(bot_message, str):
        yield bot_message, session_state
    else:
        for m in bot_message:
            yield m, session_state

    print(kwargs)
    pprint(history + [[message, bot_message]])
    session_state['previous_message'] = message

bot_fn = _bot_fn_session_state if 'session_state' in SETTINGS else _bot_fn

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
                        additional_outputs=[KWARGS['session_state']] if 'session_state' in KWARGS else None,
                        retry_btn="Retry", undo_btn="Undo", clear_btn="Clear",
                    )

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
                if hasattr(chatbot, '_upload_fn') and hasattr(attach, 'change'):
                    attach.change(chatbot._upload_fn,
                        [chatbot.textbox, attach], 
                        [chatbot.textbox], queue=False, api_name=False)
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
