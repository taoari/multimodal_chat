# app.py : Multimodal Chatbot
import gradio as gr
import random
from pprint import pprint
from dotenv import load_dotenv

from utils import parse_message, format_to_message

load_dotenv()  # take environment variables from .env.

import logging

logging.basicConfig(level=logging.WARN, format='%(asctime)-15s] %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p %Z")

def print(*args, **kwargs):
    sep = kwargs['sep'] if 'sep' in kwargs else ' '
    logging.warning(sep.join([str(val) for val in args])) # use level WARN for print, as gradio level INFO print unwanted messages


#################################################################################

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
}

SETTINGS = {
    'session_state': dict(cls='State', value={}),
    'chat_engine': dict(cls='Radio', choices=['auto', 'random', 'openai'], value='auto', 
            interactive=True, label="Chat engine"),
    'system_prompt': dict(cls='Textbox', interactive=True, lines=5, label="System prompt"),
}

PARAMETERS = {
}

KWARGS = {}

#################################################################################

def _create_from_dict(PARAMS):
    params = {}
    for name, kwargs in PARAMS.items():
        cls_ = kwargs['cls']; del kwargs['cls']
        params[name] = getattr(gr, cls_)(**kwargs)
    return params

def _random_bot_fn(message, history, **kwargs):
    # Example multimodal messages
    samples = [
        format_to_message(dict(text="I love cat", images=["https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"])),
        format_to_message(dict(text="I hate cat", images=["https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/2560px-Felis_catus-cat_on_snow.jpg"])),
        format_to_message(dict(audios=["https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav"])),
        format_to_message(dict(videos=["https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"])),
        format_to_message(dict(files=["https://www.africau.edu/images/default/sample.pdf"])),
        format_to_message(dict(text="Hello, how can I assist you today?", buttons=['Primary', dict(text='Secondary', value="the second choice")])),
        format_to_message(dict(text="We found the following items:", cards=[
            dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", title="Siam Lilac Point", 
                 text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads, with silver-gray fur surrounding those points."),
            dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", 
                 title="Siam Lilac Point", text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads, with silver-gray fur surrounding those points.",
                 extra="""<a href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg" class="btn btn-primary btn-sm text-white">More</a>"""),
        ])),
    ]
    if 'pdf' in message:
        bot_message = samples[4]
    elif 'button' in message:
        bot_message = samples[5]
    elif 'card' in message:
        bot_message = samples[6]
    else:
        bot_message = random.choice(samples)
    return bot_message

def _openai_bot_fn(message, history, **kwargs):
    import openai, os
    openai.api_key = os.environ["OPENAI_API_KEY"]

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": message},
            ],
    )
    return resp.choices[0].message.content

def _openai_bot_stream_fn(message, history, **kwargs):
    import openai, os
    openai.api_key = os.environ["OPENAI_API_KEY"]

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": message},
            ],
        stream=True,
    )

    bot_message = ""
    for _resp in resp:
        if 'content' in _resp.choices[0].delta: # last resp delta is empty
            bot_message += _resp.choices[0].delta.content # need to accumulate previous message
        yield bot_message.strip() # accumulated message can easily be postprocessed

def bot_fn(message, history, *args):
    kwargs = {name: value for name, value in zip(KWARGS.keys(), args)}

    session_state = kwargs['session_state']
    kwargs['chat_engine'] = 'random' if kwargs['chat_engine'] == 'auto' else kwargs['chat_engine']

    bot_message = {'random': _random_bot_fn,
        'openai': _openai_bot_stream_fn,
        }.get(kwargs['chat_engine'])(message, history, **kwargs)
    
    if isinstance(bot_message, str):
        for i in range(len(bot_message)):
            yield bot_message[:i+1], session_state
    else:
        for m in bot_message:
            yield m, session_state

    print(kwargs); print(session_state)
    pprint(history + [[message, bot_message]])
    session_state['previous_message'] = message


def get_demo():

    # use css and elem_id to format
    css="""#chatbot {
min-height: 600px;
}"""

    # NOTE: can not be inside another gr.Blocks
    # _chatbot = gr.Chatbot(elem_id="chatbot", avatar_images = ("user.png", "bot.png"))
    # _textbox = gr.Textbox(container=False, show_label=False, placeholder="Type a message...", scale=10, elem_id='inputTextBox', min_width=300)

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
                global KWARGS
                KWARGS = {**attachments, **settings, **parameters}
                import chat_interface
                chatbot = chat_interface.ChatInterface(bot_fn, # chatbot=_chatbot, textbox=_textbox,
                        additional_inputs=list(KWARGS.values()),
                        additional_outputs=[KWARGS['session_state']],
                        retry_btn="Retry", undo_btn="Undo", clear_btn="Clear",
                    )

                with gr.Accordion("Examples", open=False) as examples_accordin:
                    chat_examples = gr.Examples(
                        ["What's the Everett interpretation of quantum mechanics?",
                        'Give me a list of the top 10 dive sites you would recommend around the world.',
                        'Write a Python code to calculate Fibonacci numbers.'
                        ],
                        inputs=chatbot.textbox, label="AI Chat Examples",
                    )

            if hasattr(chatbot, '_upload_fn'):
                for name, attach in attachments.items():
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
