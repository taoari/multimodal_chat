# app.py : Multimodal Chatbot
import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import app_template
import app_chat
import app_chat_v3
import app_create

TABS = {
    'AI Chat': app_chat_v3.get_demo,
    # 'AI Chat': app_chat.get_demo,
    'AI Create': app_create.get_demo,
    # 'Template': app_template.get_demo,
}

css="""#chatbot {
min-height: 600px;
}"""

def get_demo():
    with gr.Blocks(css=css) as demo:
        for name, get_tab_func in TABS.items():
            with gr.Tab(name):
                get_tab_func()

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
    # WARNING: gobal variables are shared accross users, and should be avoided.

    demo = get_demo()
    from utils import reload_javascript
    reload_javascript()
    demo.queue().launch(server_name='0.0.0.0', server_port=args.port)