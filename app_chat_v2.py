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
        'gpt2': 'GPT-2', 'falcon-7b-instruct': 'Falcon (7b Instruct)',
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
            ['auto', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt2', 'falcon-7b-instruct'])),
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


DEFAULT_INSTRUCTIONS = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."""
DEFAULT_INSTRUCTIONS_FALCON = """The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, and a human user, called User. In the following interactions, User and Falcon will converse in natural language, and Falcon will answer User's questions. Falcon was built to be respectful, polite and inclusive. Falcon was built by the Technology Innovation Institute in Abu Dhabi. Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins."""

def _format_messages(history, message=None, system=None, format='plain', 
        user_name='user', bot_name='assistant'):
    _history = []
    if format == 'openai_chat':
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
    
    elif format == 'chatml':
        if system:
            _history.append('<|im_start|>system\n{system}<|im_end|>')
        for human, ai in history:
            if human:
                _history.append(f'<|im_start|>{user_name}\n{human}<|im_end|>')
            if ai:
                _history.append(f'<|im_start|>{bot_name}\n{ai}')
        if message:
            _history.append(f'<|im_start|>{user_name}\n{message}<|im_end|>')
            _history.append(f'<|im_start|>{bot_name}\n')
        return '\n'.join(_history)
    
    elif format == 'plain':
        if system:
            _history.append(system)
        for human, ai in history:
            if human:
                _history.append(f'{user_name}: {human}')
            if ai:
                _history.append(f'{bot_name}: {ai}')
        if message:
            _history.append(f'{user_name}: {message}')
            _history.append(f'{bot_name}: ')
        return '\n'.join(_history)
    
    else:
        raise ValueError(f"Invalid messages to format: {format}")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def _print_messages(history, message, bot_message, system=None,
    user_name='user', bot_name='assistant'):
    """history is list of tuple [(user_msg, bot_msg), ...]"""
    prompt = _format_messages(history, message, system=system, user_name=user_name, bot_name=bot_name)
    print(f'{bcolors.OKCYAN}{prompt}{bcolors.OKGREEN}{bot_message}{bcolors.ENDC}')


def _openai_bot_fn(message, history, _settings, _parameters):
    import openai, os
    openai.api_key = os.environ["OPENAI_API_KEY"]

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=_format_messages(history, message, format='openai_chat'),
    )
    return resp.choices[0].message.content

def _openai_stream_bot_fn(message, history, _settings, _parameters):
    # NOTE: do not limit max_tokens, as ChatGPT is capable of writing long essay.
    kwargs = dict(temperature=_parameters['temperature']) # , max_tokens=_parameters['max_tokens'])

    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]

    resp = openai.ChatCompletion.create(
        model=_settings['chat_engine'],
        messages=_format_messages(history, message, format='openai_chat'),
        stream=True,
        **kwargs,
    )

    bot_message = ""
    for _resp in resp:
        if 'content' in _resp.choices[0].delta: # last resp delta is empty
            bot_message += _resp.choices[0].delta.content # need to accumulate previous message
        yield bot_message.strip() # accumulated message can easily be postprocessed

    _print_messages(history, message, bot_message)

def _hf_gpt2_bot_fn(message, history, _settings, _parameters):
    # NOTE: text_generation.Client got error for gpt2
    kwargs = dict(temperature=max(0.001, _parameters['temperature']), max_new_tokens=min(250, _parameters['max_tokens'])) # max_new_tokens <= 250 for gpt2
    import json
    import requests
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    def query(payload):
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))
    
    # system, user_name, bot_name = DEFAULT_INSTRUCTIONS, 'Human', 'AI'
    # prompt = _format_messages(history, message, system=system, user_name=user_name, bot_name=bot_name)
    bot_message = query({'inputs': message, 'parameters': kwargs})[0]['generated_text']
    return bot_message

def _hf_stream_bot_fn(message, history, _settings, _parameters):
    # NOTE: temperature > 0 for HF models, max_new_tokens instead of max_tokens
    kwargs = dict(temperature=max(0.001, _parameters['temperature']), max_new_tokens=_parameters['max_tokens'])

    from text_generation import Client
    API_URL = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'
    API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    client = Client(API_URL, headers=headers)

    system, user_name, bot_name = DEFAULT_INSTRUCTIONS_FALCON, 'User', 'Falcon'
    prompt = _format_messages(history, message, system=system, user_name=user_name, bot_name=bot_name)

    # bot_message = client.generate(prompt, **kwargs).generated_text.strip().split(f'\n{user_name}')[0]
    bot_message = ""
    for response in client.generate_stream(prompt, **kwargs):
        if not response.token.special:
            bot_message += response.token.text
            yield bot_message.strip().split(f'\n{user_name}')[0] # stop word

    bot_message = bot_message.strip().split(f'\n{user_name}')[0]
    _print_messages(history, message, bot_message, system=system, user_name=user_name, bot_name=bot_name)
    return bot_message

def bot_fn(message, history, *args):
    yield "" # Not crash if empty user input

    _settings = {name: value for name, value in zip(SETTINGS.keys(), args[:len(SETTINGS)])}
    _parameters = {name: value for name, value in zip(PARAMETERS.keys(), args[len(SETTINGS):])}

    _settings['chat_engine'] = DISPLAY2TEXT[_settings['chat_engine']]
    _settings['chat_engine'] = 'gpt-3.5-turbo' if _settings['chat_engine'] == 'auto' else _settings['chat_engine']

    bot_message = {'random': _random_bot_fn,
        'gpt-3.5-turbo': _openai_stream_bot_fn,
        'gpt-3.5-turbo-16k': _openai_stream_bot_fn,
        'gpt2': _hf_gpt2_bot_fn,
        'falcon-7b-instruct': _hf_stream_bot_fn,
        }.get(_settings['chat_engine'])(message, history, _settings, _parameters)
    
    if isinstance(bot_message, str):
        for i in range(len(bot_message)):
            yield bot_message[:i+1]
    else:
        for m in bot_message:
            yield m
        bot_message = m # for printing

    print(_settings); print(_parameters)
    # pprint(history + [[message, bot_message]]


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