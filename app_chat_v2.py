# app.py : Multimodal Chatbot
import gradio as gr
import random
from pprint import pprint
from dotenv import load_dotenv
import os

from utils import parse_message, format_to_message

load_dotenv()  # take environment variables from .env.

# Used for Radio, CheckboxGroup, Dropdown for convert between text and display text
TEXT2DISPLAY = { 
        'auto': 'Auto', 'random': 'Random', 'openai': 'OpenAI', # for chat engine
        'gpt-3.5-turbo': 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k': 'gpt-3.5-turbo-16k',
        'falcon-7b-instruct': 'Falcon (7b Instruct)',
    }

DISPLAY2TEXT = {v:k for k,v in TEXT2DISPLAY.items()}

#################################################################################

TITLE = "AI Chat (v2)"

DESCRIPTION = """
# AI Chat

Simply enter text and press ENTER in the textbox to interact with the chatbot.
"""

SETTINGS = {
    'chat_engine': dict(cls='Radio', choices=list(map(TEXT2DISPLAY.get, 
            ['auto', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'falcon-7b-instruct'])),
            value=TEXT2DISPLAY['auto'], 
            interactive=True, label="Chat engine"),
}

PARAMETERS = {
    'temperature': dict(cls='Slider', minimum=0, maximum=1, value=0.7, step=0.1, interactive=True, label="Temperature",
            info="Lower temperator for determined output; hight temperate for randomness"),
    'max_tokens': dict(cls='Slider', minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max new tokens"),
    # 'top_k': dict(cls='Slider', minimum=1, maximum=5, value=3, step=1, interactive=True, label="Top K"),
    # 'top_p': dict(cls='Slider', minimum=0, maximum=1, value=0.9, step=0.1, interactive=True, label="Top p"),

}

#################################################################################

def _create_from_dict(PARAMS):
    params = {}
    for name, kwargs in PARAMS.items():
        cls_ = kwargs['cls']; del kwargs['cls']
        params[name] = getattr(gr, cls_)(**kwargs)
    return params

def _random_bot_fn(message, history, _settings, _parameters):
    # Example multimodal messages
    bot_message = random.choice([
        format_to_message(dict(text="I love cat", images=["https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"])),
        format_to_message(dict(text="I hate cat", images=["https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/2560px-Felis_catus-cat_on_snow.jpg"])),
        format_to_message(dict(audios=["https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav"])),
        format_to_message(dict(videos=["https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"])),
        format_to_message(dict(files=["https://www.africau.edu/images/default/sample.pdf"])),
        format_to_message(dict(text="Hello, how can I assist you today?", buttons=['Primary', 'Secondary'])),
    ])
    return bot_message

def _format_messages(history, message=None, system=None, format='openai'):
    _history = []
    if system:
        _history.append({'role': 'system', 'content': system})
    for human, ai in history:
        if human:
            _history.append({'role': 'user', 'content': human})
        if ai:
            _history.append({'role': 'assistant', 'content': ai})
    if message:
        _history.append({'role': 'user', 'content': message})
    return _history

def _openai_bot_fn(message, history, _settings, _parameters):
    import openai, os
    openai.api_key = os.environ["OPENAI_API_KEY"]

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=_format_messages(history, message),
    )
    return resp.choices[0].message.content

def _openai_stream_bot_fn(message, history, _settings, _parameters):
    kwargs = dict(temperature=_parameters['temperature'], max_tokens=_parameters['max_tokens'])

    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]

    resp = openai.ChatCompletion.create(
        model=_settings['chat_engine'],
        messages=_format_messages(history, message),
        stream=True,
        **kwargs,
    )

    bot_message = ""
    for _resp in resp:
        if 'content' in _resp.choices[0].delta: # last resp delta is empty
            bot_message += _resp.choices[0].delta.content # need to accumulate previous message
        yield bot_message.strip() # accumulated message can easily be postprocessed

def _hf_stream_bot_fn(message, history, _settings, _parameters):
    # NOTE: temperature > 0 for HF models, max_new_tokens instead of max_tokens
    kwargs = dict(temperature=max(0.001, _parameters['temperature']), max_new_tokens=_parameters['max_tokens'])

    from text_generation import Client
    API_URL = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'
    API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    client = Client(API_URL, headers=headers)

    bot_message = ""
    for response in client.generate_stream(message, **kwargs):
        if not response.token.special:
            bot_message += response.token.text
            yield bot_message.strip()

def bot_fn(message, history, *args):
    yield "" # Not crash if empty user input

    _settings = {name: value for name, value in zip(SETTINGS.keys(), args[:len(SETTINGS)])}
    _parameters = {name: value for name, value in zip(PARAMETERS.keys(), args[len(SETTINGS):])}

    _settings['chat_engine'] = DISPLAY2TEXT[_settings['chat_engine']]
    _settings['chat_engine'] = 'gpt-3.5-turbo' if _settings['chat_engine'] == 'auto' else _settings['chat_engine']

    bot_message = {'random': _random_bot_fn,
        'gpt-3.5-turbo': _openai_stream_bot_fn,
        'gpt-3.5-turbo-16k': _openai_stream_bot_fn,
        'falcon-7b-instruct': _hf_stream_bot_fn,
        }.get(_settings['chat_engine'])(message, history, _settings, _parameters)
    
    if isinstance(bot_message, str):
        for i in range(len(bot_message)):
            yield bot_message[:i+1]
    else:
        for m in bot_message:
            yield m

    print(_settings); print(_parameters)
    pprint(history + [[message, bot_message]])


def get_demo():

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
                # attachments = _create_from_dict(ATTACHMENTS)
                with gr.Accordion("Settings", open=False) as settings_accordin:
                    settings = _create_from_dict(SETTINGS)
                with gr.Accordion("Parameters", open=False) as parameters_accordin:
                    parameters = _create_from_dict(PARAMETERS)

            with gr.Column(scale=9):
                import chat_interface
                chatbot = chat_interface.ChatInterface(bot_fn,
                        additional_inputs=list(settings.values()) + list(parameters.values()),
                        )

                with gr.Accordion("Examples", open=False) as examples_accordin:
                    chat_examples = gr.Examples(
                        ["What's the Everett interpretation of quantum mechanics?",
                        'Give me a list of the top 10 dive sites you would recommend around the world.',
                        'Write a Python code to calculate Fibonacci numbers.'
                        ],
                        inputs=chatbot.textbox, label="AI Chat Examples",
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

    demo = get_demo()
    from utils import reload_javascript
    reload_javascript()
    demo.queue().launch(share=True, server_port=args.port)