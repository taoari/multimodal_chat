# app.py : Multimodal Chatbot
import logging
import time
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

from llms import HF_ENDPOINTS, _get_llm, _llm_call_langchain
AVAILABLE_TOOLS = ['Search', 'OCR']

#########

TITLE = "AI Orchestration"

DESCRIPTION = """
# AI Orchestration

Simply enter text and press ENTER in the textbox to interact with the chatbot.
"""

ATTACHMENTS = {
    'session_state': dict(cls='State', value={}),
    # 'image': dict(cls='Image', type='filepath'), #, source='webcam'),
    # 'system_prompt': dict(cls='Textbox', interactive=True, lines=5, label="System prompt"),
    'status': dict(cls='JSON', label='Status'),
}

SETTINGS = {
    'tools': dict(cls='CheckboxGroup', choices=AVAILABLE_TOOLS, 
            value=AVAILABLE_TOOLS,
            interactive=True, label='Tools'),
    'chat_engine': dict(cls='Radio', choices=['auto', 'gpt-3.5-turbo-0613', 'gpt-4'] + list(HF_ENDPOINTS.keys()),
            value='auto', 
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

################################################################
# Bot fn
################################################################

mrkl = None

def _langchain_agent_bot_fn(message, history, **kwargs):
    session_state = kwargs['session_state']

    # TODO: mrkl is shared accross users, need to be in sesson state
    global mrkl
    if mrkl is None:
        from langchain.agents import initialize_agent, Tool
        from langchain.agents import AgentType
        from langchain.chat_models import ChatOpenAI

        # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        llm = _get_llm(chat_engine=kwargs.get('chat_engine', "gpt-3.5-turbo-0613"), temperature=0)
        
        from tools.utils import get_tool
        tools = [get_tool(name, llm=llm) for name in kwargs['tools']]

        from langchain.prompts import MessagesPlaceholder
        from langchain.memory import ConversationBufferMemory

        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }
        memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

        mrkl = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True,
                    agent_kwargs=agent_kwargs, memory=memory)
        
    from utils import parse_message
    msg_dict = parse_message(message)
    if 'images' in msg_dict and len(msg_dict['images']) > 0:
        _msg = f"{msg_dict['text']}: {msg_dict['images'][-1]}"
        session_state['current_file'] = msg_dict['images'][-1]
    else:
        _msg = msg_dict['text']
    return mrkl.run(_msg)

def bot_fn(message, history, *args):
    __TIC = time.time()
    kwargs = {name: value for name, value in zip(KWARGS.keys(), args)}

    session_state = kwargs['session_state']
    kwargs['chat_engine'] = 'gpt-3.5-turbo-16k' if kwargs['chat_engine'] == 'auto' else kwargs['chat_engine']

    # TODO: 1. agent history
    # 2. fallback LLM
    # need to modify kwargs['session_state']['current_file']
    try:
        bot_message = _langchain_agent_bot_fn(message, history, **kwargs)
    except:
        bot_message = _llm_call_langchain(message, history, **kwargs)
    
    session_state['message'] = message
    status = {**session_state}
    
    if isinstance(bot_message, str):
        __TOC = time.time(); status['elapsed_time'] = __TOC - __TIC
        yield bot_message, session_state, status
    else:
        for m in bot_message:
            __TOC = time.time(); status['elapsed_time'] = __TOC - __TIC
            yield m, session_state, status

    __TOC = time.time()
    print(f'Elapsed time: {__TOC-__TIC}')
    print(kwargs)
    pprint(history + [[message, bot_message]])
    session_state['previous_message'] = message

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
                with gr.Accordion("Settings", open=True) as settings_accordin:
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
                        additional_outputs=[KWARGS['session_state'], KWARGS['status']] if 'session_state' in KWARGS else None,
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
                if hasattr(chatbot, '_upload_fn') and isinstance(attach, (gr.Image, gr.Audio, gr.Video, gr.File)):
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

    import langchain
    langchain.debug = args.debug

    demo = get_demo()
    from utils import reload_javascript
    reload_javascript()
    demo.queue().launch(server_name='0.0.0.0', server_port=args.port)
