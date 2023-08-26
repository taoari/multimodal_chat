# app.py : Multimodal Chatbot
import os
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

from llms import HF_ENDPOINTS, _get_llm, _llm_call_langchain, _llm_call_stream
from utils import parse_message, format_to_message, get_spinner
AVAILABLE_TOOLS = ['Search', 'OCR']

#########

TITLE = "AI Orchestration"

DESCRIPTION = """
# AI Orchestration

Simply enter text and press ENTER in the textbox to interact with the chatbot.
"""

_default_session_state = dict(current_file=None, context=None)

ATTACHMENTS = {
    'session_state': dict(cls='State', value=_default_session_state),
    # 'image': dict(cls='Image', type='filepath'), #, source='webcam'),
    # 'system_prompt': dict(cls='Textbox', interactive=True, lines=5, label="System prompt"),
    'status': dict(cls='JSON', label='Status'),
}

SETTINGS = {
    'tools': dict(cls='CheckboxGroup', choices=AVAILABLE_TOOLS, 
            value=[t for t in AVAILABLE_TOOLS if t not in {"Search"}],
            interactive=True, label='Tools'),
    'chat_engine': dict(cls='Radio', choices=['auto', 'gpt-3.5-turbo-0613', 'gpt-4'] + list(HF_ENDPOINTS.keys()),
            value='auto', 
            interactive=True, label="Chat engine"),
}

PARAMETERS = {
    'max_pages': dict(cls='Slider', minimum=0, maximum=16, value=8, step=1, 
            interactive=True, label="Max pages", info="Max pages to be processed for vector store (0 for all)."),
    'query_k': dict(cls='Slider', minimum=1, maximum=10, value=3, step=1, 
            interactive=True, label="Query k"),
}
    
KWARGS = {} # use for chatbot additional_inputs, do NOT change

SESSION_STATE = {"current_vs": None} # for complex object

DEBUG = True
    
################################################################
# utils
################################################################

def _create_from_dict(PARAMS):
    params = {}
    for name, kwargs in PARAMS.items():
        cls_ = kwargs['cls']; del kwargs['cls']
        params[name] = getattr(gr, cls_)(**kwargs)
    return params

def _build_vs(fname, chunk_size=0, persist_directory=None, 
            max_pages=0, verbose=False):
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(fname)
    pages = loader.load()

    print(f'len(pages) = {len(pages)}')

    if chunk_size > 0:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        docs = r_splitter.split_documents(pages)
    else:
        docs = pages

    print(f'len(docs) = {len(docs)}')

    if max_pages > 0:
        docs = docs[:max_pages]
        print(f'len(docs) for vs = {len(docs)}')

    from langchain.embeddings import HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings()

    from langchain.vectorstores import Chroma
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persist_directory
    )
    if persist_directory is not None:
        vectordb.persist()
    print(f'vector db for {fname} done!')

    return vectordb

def _load_vs(persist_directory):
    from langchain.embeddings import HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings()

    from langchain.vectorstores import Chroma
    vectordb = Chroma(
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb

################################################################
# Bot fn
################################################################

memory = None

def _langchain_agent_bot_fn(message, history, **kwargs):
    session_state = kwargs['session_state']

    # TODO: mrkl is shared accross users, need to be in sesson state
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
    global memory
    if memory is None:
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


def _is_document_qa(session_state):
    cond1 = 'current_file' in session_state and session_state['current_file'] is not None and session_state['current_file'].endswith('.pdf')
    cond2 = 'current_vs' in SESSION_STATE and SESSION_STATE['current_vs'] is not None
    return cond1 and cond2

def _beautify_status(status):
    max_len = 100
    if 'context' in status and status['context'] is not None and len(status['context']) >= max_len:
        status['context'] = status['context'][:max_len] + ' ...(truncated)'

    return status

def bot_fn(message, history, *args):
    global SESSION_STATE
    __TIC = time.time()
    kwargs = {name: value for name, value in zip(KWARGS.keys(), args)}

    session_state = kwargs['session_state']
    kwargs['chat_engine'] = 'gpt-3.5-turbo-16k' if kwargs['chat_engine'] == 'auto' else kwargs['chat_engine']

    msg_dict = parse_message(message)
    if len(msg_dict['images']) > 0:
        session_state['current_file'] = msg_dict['images'][-1]
        session_state['context'] = None
        SESSION_STATE['current_vs'] = None
    elif len(msg_dict['files']) > 0:
        session_state['current_file'] = msg_dict['files'][-1]
        yield get_spinner() + f"Building vector store for **{os.path.basename(session_state['current_file'])}**, please be patient.", session_state, session_state
        if session_state['current_file'].endswith('.pdf'):
            vs = _build_vs(session_state['current_file'], max_pages=kwargs.get('max_pages', 0))
            SESSION_STATE['current_vs'] = vs

    if _is_document_qa(session_state):
        # Document QA if a PDF file is uploaded
        if  msg_dict['text']:
            vectordb = SESSION_STATE['current_vs']
            res = vectordb.similarity_search(msg_dict['text'], k=kwargs.get('query_k', 3))
            context = '\n\n'.join([doc.page_content for doc in res])
            _kwargs = {'system_prompt': context, **kwargs}
            bot_message = _llm_call_stream(message, history, **_kwargs)
            session_state['context'] = context
        else:
            bot_message = format_to_message(dict(
                    text=f"You have uploaded {os.path.basename(session_state['current_file'])}. How can I help you today?",
                    buttons=[dict(text='Summarize', value="Summarize the text.")],
                ))
    else:
        # TODO: 1. agent history
        # 2. fallback LLM
        # need to modify kwargs['session_state']['current_file']
        try:
            bot_message = _langchain_agent_bot_fn(message, history, **kwargs)
        except:
            bot_message = _llm_call_stream(message, history, **kwargs)
    
    session_state['message'] = message
    status = _beautify_status({**session_state, 'SESSION_STATE_KEYS': list(SESSION_STATE.keys())})
    
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
                with gr.Accordion("Parameters", open=True) as parameters_accordin:
                    parameters = _create_from_dict(PARAMETERS)

            with gr.Column(scale=9):
                # chatbot
                global KWARGS
                KWARGS = {**attachments, **settings, **parameters}
                import chat_interface
                chatbot = chat_interface.ChatInterface(bot_fn, # chatbot=_chatbot, textbox=_textbox,
                        additional_inputs=list(KWARGS.values()),
                        additional_outputs=[KWARGS['session_state'], KWARGS['status']] if 'session_state' in KWARGS else None,
                        upload_btn="📁",
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
    langchain.debug = DEBUG

    demo = get_demo()
    from utils import reload_javascript
    reload_javascript()
    demo.queue().launch(server_name='0.0.0.0', server_port=args.port)
