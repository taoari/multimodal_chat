# app.py : Multimodal Chatbot
import os
import time
import logging
from dotenv import load_dotenv
import gradio as gr
from jinja2 import Template

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

from utils import _reformat_message, _reformat_history
from utils import parse_message, format_to_message, get_spinner
from llms import HF_ENDPOINTS, _get_llm, _llm_call, _llm_call_stream, _print_messages
from vectorstore import _get_hash, _build_vs_dedup, _build_vs_collection

DEBUG = True
GLOBAL_CACHE = {"vs": {}} # for complex object
AVAILABLE_TOOLS = ['Search', 'OCR', 'Barcode']

PROMPT_TEMPLATE_QA = """"Use the following pieces of context and chat history to answer the question.

{context}"""

PROMPT_TEMPLATE_QA_JINJA2 = """Use the following pieces of context and chat history to answer the question.

{% for doc in docs %}


    {{ doc.page_content }}
{% endfor %}
"""

# One can assume that keys of _default_session_state always exist
_default_session_state = dict(current_file=None, 
        context=None, 
        current_vs=None, # current_vs can be caluclated from _get_hash(current_file, is_file=True)
        context_switch_at=0, # history before context_switch_at should be ignored
        message=None,
        previous_message=None,
   )

################################################################
# Global variables
################################################################

TITLE = "AI Orchestration"

DESCRIPTION = """
# AI Orchestration

Simply enter text and press ENTER in the textbox to interact with the chatbot.
"""

ATTACHMENTS = {
    'session_state': dict(cls='State', value=_default_session_state),
    # 'image': dict(cls='Image', type='filepath'), #, source='webcam'),
    'system_prompt': dict(cls='Textbox', interactive=True, lines=10, label="System prompt", 
            value=PROMPT_TEMPLATE_QA_JINJA2,
            info="Jinja2 template syntax. Refer to https://atufashireen.medium.com/creating-templates-with-jinja-in-python-3ff3b87d6740 for a quick intro for advanced usage."),
    'status': dict(cls='JSON', label='Status info'),
}

SETTINGS = {
    'tools': dict(cls='CheckboxGroup', choices=AVAILABLE_TOOLS, 
            value=[], # [t for t in AVAILABLE_TOOLS if t not in {"Search"}],
            interactive=True, label='Tools'),
    'chat_engine': dict(cls='Dropdown', choices=['auto', 'gpt-3.5-turbo-0613', 'gpt-4', 'gpt-4-1106-preview'] + list(HF_ENDPOINTS.keys()),
            value='auto',
            interactive=True, label="Chat engine"),
    'with_memory': dict(cls='Checkbox', value=False, interactive=True, label='With memory'),
    'with_score': dict(cls='Checkbox', value=True, interactive=True, label='With score'),
    '_format': dict(cls='Radio', choices=['auto', 'html', 'plain', 'json'], value='auto', 
            interactive=True, label="Bot response format"),
}

PARAMETERS = {
    'qa_separator': dict(cls='Markdown', value="**Document QA parameters**"),
    # 'max_pages': dict(cls='Slider', minimum=0, maximum=16, value=8, step=1, 
    #         interactive=True, label="Max pages", info="Max pages to be processed (0 to process all)."),
    'collection': dict(cls='Dropdown', choices=['none', 'default_collection'],
            value='none',
            interactive=True, label="Collection"),
    'query_k': dict(cls='Slider', minimum=1, maximum=10, value=2, step=1, 
            interactive=True, label="Query k", info="Increase for better QA results. If error occurs (retrieved doc exceeds LLM context length), reduce its value."),
    # 'show_citations': dict(cls='Checkbox', value=False, interactive=True, label='Show citations (Experimental)'),

}
    
KWARGS = {} # use for chatbot additional_inputs, do NOT change
    
################################################################
# utils
################################################################

def _create_from_dict(PARAMS, tabbed=False):
    params = {}
    for name, kwargs in PARAMS.items():
        cls_ = kwargs['cls']; del kwargs['cls']
        if not tabbed:
            params[name] = getattr(gr, cls_)(**kwargs)
        else:
            tab_name = kwargs['label'] if 'label' in kwargs else name
            with gr.Tab(tab_name):
                params[name] = getattr(gr, cls_)(**kwargs)
    return params

def _clear(session_state):
    session_state.clear()
    session_state.update(_default_session_state)
    return session_state

def _generator_prefix(generator, prefix="", surfix=""):
    return (prefix + _str + surfix for _str in generator)


################################################################
# Bot fn
################################################################

def _langchain_agent_bot_fn(message, history, **kwargs):
    session_state = kwargs['session_state']
    chat_engine = kwargs.get('chat_engine', "gpt-3.5-turbo-0613")

    # TODO: mrkl is shared accross users, need to be in sesson state
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    from langchain.chat_models import ChatOpenAI

    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm = _get_llm(chat_engine=chat_engine, temperature=0)
    
    from tools.utils import get_tool
    tools = [get_tool(name, llm=llm) for name in kwargs['tools']]

    from langchain.prompts import MessagesPlaceholder
    from langchain.memory import ConversationBufferMemory

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    for human, ai in history:
        memory.save_context({"input": human}, {"output": ai})
    print(memory.load_memory_variables({}))

    mrkl = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True,
                agent_kwargs=agent_kwargs, memory=memory)
        
    from utils import parse_message
    msg_dict = parse_message(message)
    if 'images' in msg_dict and len(msg_dict['images']) > 0:
        _msg = f"{msg_dict['text']}: {msg_dict['images'][-1]}"
        session_state['current_file'] = msg_dict['images'][-1]
    else:
        _msg = msg_dict['text']
    bot_message = mrkl.run(_msg)
    if 'verbose' in kwargs and kwargs['verbose']:
        _print_messages(history, message, bot_message, variant='secondary', tag=f'langchain_agent_openai_functions ({chat_engine})')
    return bot_message

def _slash_bot_fn(message, history, **kwargs):
    cmds = message.split(' ', maxsplit=1)
    cmd, rest = cmds[0], cmds[1] if len(cmds) == 2 else ''
    return message

def _is_document_qa(session_state):
    # TODO: might cause bug with Undo
    cond1 = 'current_file' in session_state and session_state['current_file'] is not None and session_state['current_file'].endswith('.pdf')
    cond2 = 'vs' in GLOBAL_CACHE and 'current_vs' in session_state and session_state['current_vs'] in GLOBAL_CACHE['vs']
    return cond1 and cond2

def _beautify_status(status):
    max_len = 500
    if 'context' in status and status['context'] is not None and len(status['context']) >= max_len:
        status['context'] = status['context'][:max_len] + ' ...(truncated)'

    return status

def _format_sources(docs, scores=None):
    if scores is None:
        files = ['{}#page={}'.format(doc.metadata['source'], doc.metadata['page'] + 1) for doc in docs]
        return "\n\n{}".format(format_to_message(dict(text="**Sources**", files=files)))
    else:
        # has to use HTML to format
        res = '<br /> <b>Sources</b> <br />'
        for doc, score in zip(docs, scores):
            # <a href="\file=data/default_collection/days_of_supply_vf.pdf#page=3" target="_blank">📁 days_of_supply_vf.pdf#page=3</a>
            _f = '{}#page={}'.format(doc.metadata['source'], doc.metadata['page'] + 1)
            res += f'<a href="\\file={_f}" target="_blank">📁 {os.path.split(_f)[-1]}</a> <span class="badge badge-info text-black">score: {score:.2f}</span> <br />'
        return res
        

def prebuild_vs():
    GLOBAL_CACHE['vs']['default_collection'] = _build_vs_collection('data/default_collection', 'default_collection')

def _custom_bot_fn(message, history, **kwargs):
    """
    Args
    ======

    message: user input (Str)
        plain text or html
    history: chat history (List[Tuple[Str, Str]])
        [(user, bot), (user, bot), ...]
    kwargs: kwargs defined in ATTACHMENTS, SETTINGS, or PARAMETERS (Dict)
        e.g. chat_engine, session_state

    Returns
    =======
    
    bot_message: bot response (Str or Generator)
        generator response enables streaming support
    """
    bot_message = _llm_call(message.removeprefix('/gpt '), history, chat_engine='gpt-3.5-turbo', temperature=0)
    # bot_message = _llm_call_stream(message, history, **kwargs)
    return bot_message

def bot_fn(message, history, *args):
    __TIC = time.time()
    kwargs = {name: value for name, value in zip(KWARGS.keys(), args)}
    kwargs['verbose'] = True # auto print llm calls
    session_state = kwargs['session_state']
    if len(history) == 0 or message == '/clear':
        _clear(session_state)
    # unformated LLM history for rich response applications, keep only after latest context switch
    history = _reformat_history(history[session_state['context_switch_at']:]) if kwargs['with_memory'] else []
    plain_message = _reformat_message(message)

    """ BEGIN: Update only this part if necessary """

    AUTOS = {'chat_engine': 'gpt-3.5-turbo-0613'}

    # set param to default value if param is "auto"
    for param, default_value in AUTOS.items():
        kwargs[param] = default_value if kwargs[param] == 'auto' else kwargs[param]

    # slash cmd
    if message.startswith('/gpt '):
        bot_message = _custom_bot_fn(message, history, **kwargs)
    elif message.startswith('/'):
        bot_message = _slash_bot_fn(message, history, **kwargs)
    else:
        # image or file uploaded
        msg_dict = parse_message(message)
        if len(msg_dict['images']) > 0:
            _clear(session_state)
            session_state['current_file'] = msg_dict['images'][-1]
            session_state['context_switch_at'] = len(history)
            from utils import fix_exif_orientation
            fix_exif_orientation(session_state['current_file'])
        elif len(msg_dict['files']) > 0:
            fname = msg_dict['files'][-1]
            if fname.endswith('.pdf'):
                _clear(session_state)
                session_state['current_file'] = fname
                session_state['current_vs'] = _get_hash(fname, is_file=True)
                session_state['context_switch_at'] = len(history)
                yield get_spinner() + f"Building vector store for **{os.path.basename(fname)}**, please be patient.", session_state, session_state
                vs = _build_vs_dedup(fname, max_pages=kwargs.get('max_pages', 0))
                GLOBAL_CACHE['vs'][session_state['current_vs']] = vs

        # document QA
        if kwargs['collection'] == 'default_collection':
            # Document QA for a folder
            if  msg_dict['text']:
                vectordb = GLOBAL_CACHE['vs'][kwargs['collection']]

                # get docs and scores from vectordb similarity search
                if not kwargs['with_score']:
                    docs = vectordb.similarity_search(msg_dict['text'], k=kwargs.get('query_k', 3))
                    scores = None
                else:
                    res = vectordb.similarity_search_with_score(msg_dict['text'], k=kwargs.get('query_k', 3))
                    docs = [_r[0] for _r in res] # compatible with similarity_search format
                    scores = [1.0 - _r[1] for _r in res] # extract scores

                # custom document qa system prompt
                if kwargs['system_prompt']:
                    system_prompt = Template(kwargs['system_prompt']).render(docs=docs)
                    session_state['context'] = system_prompt
                else:
                    context = '\n\n'.join([doc.page_content for doc in docs])
                    session_state['context'] = context
                    system_prompt = PROMPT_TEMPLATE_QA.format(context=context)

                # llm call
                _kwargs = {**kwargs, 'system_prompt': system_prompt, 'max_tokens': 1024} # overwrite system_prompt
                bot_message = _llm_call_stream(plain_message, history, **_kwargs)
                bot_message = _generator_prefix(bot_message, surfix=_format_sources(docs, scores))
        elif _is_document_qa(session_state):
            # Document QA if a PDF file is uploaded
            if  msg_dict['text']:
                # if not kwargs['show_citations']:
                    vectordb = GLOBAL_CACHE['vs'][session_state['current_vs']]
                    res = vectordb.similarity_search(msg_dict['text'], k=kwargs.get('query_k', 3))
                    context = '\n\n'.join([doc.page_content for doc in res])
                    session_state['context'] = context
                    system_prompt = PROMPT_TEMPLATE_QA.format(context=context)
                    _kwargs = {**kwargs, 'system_prompt': system_prompt} # overwrite system_prompt
                    # bot_message = _llm_call_stream(plain_message, history, **_kwargs)
                    bot_message = _llm_call(plain_message, history, **_kwargs) + _format_sources(res)
                # else:
                #     llm = _get_llm(chat_engine=kwargs['chat_engine'], temperature=0)
                #     vectordb = SESSION_STATE['vs'][session_state['current_vs']]

                #     from langchain.chains import RetrievalQAWithSourcesChain
                #     from langchain.chains.qa_with_sources import load_qa_with_sources_chain

                #     qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
                #     retriever = vectordb.as_retriever()
                #     qa = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=retriever)
                #     response = qa({"question": plain_message})
                #     bot_message = str(response)

            else:
                bot_message = format_to_message(dict(
                        text=f"You have uploaded {os.path.basename(session_state['current_file'])}. How can I help you today?",
                        buttons=[dict(text='Summarize', value="Summarize the text.")],
                    ))
        elif len(kwargs['tools']) == 0:
            bot_message = _llm_call_stream(plain_message, history, **kwargs)
        else:
            # agent and fallback
            try:
                bot_message = _langchain_agent_bot_fn(plain_message, history, **kwargs)
            except Exception as e:
                print(e)
                exception_msg = format_to_message({"collapses": [dict(title="Agent Exception", text=str(e), before=True)]})
                bot_message = _generator_prefix(_llm_call_stream(plain_message, history, **kwargs), prefix=exception_msg)
    
    session_state['message'] = message
    _parameters = {k: v for k,v in kwargs.items() if k not in {'session_state', 'status'}}
    status = _beautify_status({**session_state, 'SESSION_STATE_KEYS': list(GLOBAL_CACHE.keys()),
            'VS_KEYS': list(GLOBAL_CACHE['vs'].keys()),
            **_parameters})

    """ END: Update only this part if necessary """
    
    # NOTE: _reformat_message could double check parse_message and format_to_message integrity
    _format = kwargs['_format'] if '_format' in kwargs else 'auto'
    if isinstance(bot_message, str):
        __TOC = time.time(); status['elapsed_time'] = __TOC - __TIC
        yield _reformat_message(bot_message, _format=_format), session_state, status
    else:
        for m in bot_message:
            __TOC = time.time(); status['elapsed_time'] = __TOC - __TIC
            yield _reformat_message(m, _format=_format), session_state, status
        bot_message = m # for print

    __TOC = time.time()
    print(f'Elapsed time: {__TOC-__TIC}')
    session_state['previous_message'] = message

################################################################
# Gradio app
################################################################

def get_demo():

    # use css and elem_id to format
    css="""#chatbot {
min-height: 600px;
}"""
    with gr.Blocks(css=css) as demo:
        # title
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        # description
        with gr.Accordion("Expand to see Introduction and Usage", open=False):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            # attachments, settings, and parameters
            with gr.Column(scale=1):
                with gr.Accordion("Info", open=False) as info_accordin:
                    attachments = _create_from_dict(ATTACHMENTS)
                with gr.Accordion("Settings", open=False) as settings_accordin:
                    settings = _create_from_dict(SETTINGS)
                with gr.Accordion("Parameters", open=True) as parameters_accordin:
                    parameters = _create_from_dict(PARAMETERS)

            with gr.Column(scale=9):
                # chatbot
                global KWARGS
                KWARGS = {**attachments, **settings, **parameters}
                KWARGS = {k: v for k, v in KWARGS.items() if not isinstance(v, (gr.Markdown, gr.HTML, gr.JSON))}
                import chat_interface
                chatbot = chat_interface.ChatInterface(bot_fn, # chatbot=_chatbot, textbox=_textbox,
                        additional_inputs=list(KWARGS.values()),
                        additional_outputs=[KWARGS['session_state'], attachments['status']] if 'session_state' in KWARGS else None,
                        upload_btn="📁",
                        retry_btn="Retry", undo_btn="Undo", clear_btn="Clear",
                    )
                chatbot.chatbot.elem_id = 'chatbot' # for css
                chatbot.textbox.elem_id = 'inputTextBox' # for buttons
                chatbot.chatbot.avatar_images = ("assets/user.png", "assets/bot.png")

                # examples
                with gr.Accordion("Examples", open=False) as examples_accordin:
                    qa_examples = gr.Examples(
                        ['<a href="\\file=test_files/flash_attention_v2.pdf">📁 flash_attention_v2.pdf</a>',
                         'Summarize the text.',
                         'Tell me the main idea of the proposed algorithm.',
                         'Why flash attention v2 is better than v1?'],
                        inputs=chatbot.textbox, label="Documment QA Examples",
                    )
                    image_examples = gr.Examples(
                        ['<img src="\\file=test_files/JohnSmith-Example.jpg" alt="JohnSmith-Example.jpg"/>',
                         'What is the patient name?',
                         'What is the prescription?'],
                        inputs=chatbot.textbox, label="Image OCR Examples",
                    )
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

    # _build_vs_v2('test_files/sample.pdf')

    import langchain
    langchain.debug = DEBUG

    prebuild_vs()

    demo = get_demo()
    from utils import reload_javascript
    reload_javascript()
    demo.queue().launch(server_name='0.0.0.0', server_port=args.port)
