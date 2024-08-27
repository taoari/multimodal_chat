# app.py : Multimodal Chatbot
import os
import time
import logging
import jinja2
from dotenv import load_dotenv
import gradio as gr

################################################################
# Load .env and logging
################################################################

load_dotenv()  # take environment variables from .env.

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN, format='%(asctime)-15s] %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p %Z")

def print(*args, **kwargs):
    sep = kwargs['sep'] if 'sep' in kwargs else ' '
    logger.warning(sep.join([str(val) for val in args])) # use level WARN for print, as gradio level INFO print unwanted messages

def _print_messages(messages, title='Chat history:'):
    icons = {'system': '🖥️', 'user': '👤', 'assistant': '🤖'}
    res = [] if title is None else [title]
    for message in messages:
        res.append(f'{icons[message["role"]]}: {message["content"]}')
    print('\n'.join(res))

def _bot_fn(message, history, **kwargs):
    messages = history + [{'role': 'user', 'content': message}]
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
    )
    bot_message = resp.choices[0].message.content
    _print_messages(messages + [{'role': 'user', 'content': bot_message }],)
    return bot_message

def _bot_stream_fn(message, history, **kwargs):
    messages = history + [{'role': 'user', 'content': message}]
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )
    bot_message = ""
    for _resp in resp:
        if hasattr(_resp.choices[0].delta, 'content') and _resp.choices[0].delta.content:
            bot_message += _resp.choices[0].delta.content
        yield bot_message.strip()
    _print_messages(messages + [{'role': 'assistant', 'content': bot_message }])

bot_fn = _bot_stream_fn

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
        from utils.chatbot import ChatInterface
        chatbot = ChatInterface(bot_fn, type='messages', 
                multimodal=False,
                avatar_images=('assets/user.png', 'assets/bot.png'))
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
