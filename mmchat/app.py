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
# Utils
################################################################

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
