# app.py : Multimodal Chatbot
import os
import random
from pprint import pprint
from dotenv import load_dotenv

from utils import parse_message, format_to_message

################################################################
# HF Endpoints
################################################################

HF_ENDPOINTS = {}

def parse_endpoints_from_environ():
    global HF_ENDPOINTS
    for name, value in os.environ.items():
        if name.startswith('HF_INFERENCE_ENDPOINT_'):
            HF_ENDPOINTS[name[len('HF_INFERENCE_ENDPOINT_'):].lower()] = value

parse_endpoints_from_environ()

################################################################
# Format LLM messages
################################################################

DEFAULT_INSTRUCTIONS = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."""
DEFAULT_INSTRUCTIONS_FALCON = """The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, and a human user, called User. In the following interactions, User and Falcon will converse in natural language, and Falcon will answer User's questions. Falcon was built to be respectful, polite and inclusive. Falcon was built by the Technology Innovation Institute in Abu Dhabi. Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins."""
DEFAULT_INSTRUCTIONS_MPT = """A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."""
DEFAULT_INSTRUCTIONS_LLAMA = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
DEFAULT_INSTRUCTIONS_LLAMA = """"""

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
    
    elif format == 'langchain_chat':
        from langchain.schema import AIMessage, HumanMessage, SystemMessage
        if system:
            _history.append(SystemMessage(content=system))
        for human, ai in history:
            if human:
                _history.append(HumanMessage(content=human))
            if ai:
                _history.append(AIMessage(content=ai))
        if message:
            _history.append(HumanMessage(content=message))
        return _history
    
    elif format == 'chatml':
        if system:
            _history.append(f'<|im_start|>system\n{system}<|im_end|>')
        for human, ai in history:
            if human:
                _history.append(f'<|im_start|>{user_name}\n{human}<|im_end|>')
            if ai:
                _history.append(f'<|im_start|>{bot_name}\n{ai}')
        if message:
            _history.append(f'<|im_start|>{user_name}\n{message}<|im_end|>')
            _history.append(f'<|im_start|>{bot_name}\n')
        return '\n'.join(_history)

    elif format == 'llama':
        system = "" if system is None else system
        _history.append(f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n ")
        for human, ai in history:
            human = "" if human is None else human
            ai = "" if ai is None else ai
            _history.append(f"{human} [/INST] {ai} </s><s> [INST] ")
        if message:
            _history.append(f"{message} [/INST] ")
        return ''.join(_history)
    
    elif format == 'llama3':
        _history.append('<|begin_of_text|>')
        if system:
            _history.append(f'<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>')
        for human, ai in history:
            if human:
                _history.append(f'<|start_header_id|>{user_name}<|end_header_id|>\n\n{human}<|eot_id|>')
            if ai:
                _history.append(f'<|start_header_id|>{bot_name}<|end_header_id|>\n\n{ai}<|eot_id|>')
        if message:
            _history.append(f'<|start_header_id|>{user_name}<|end_header_id|>\n\n{message}<|eot_id|>')
            _history.append(f'<|start_header_id|>{bot_name}<|end_header_id|>\n\n')
        return ''.join(_history)
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
    user_name='user', bot_name='assistant', format='plain', variant='primary', tag=None):
    """history is list of tuple [(user_msg, bot_msg), ...]"""
    prompt = _format_messages(history, message, system=system, user_name=user_name, bot_name=bot_name, format=format)
    prompt = prompt.replace('</s><s>', '</s>\n<s>') # better print for llama models
    bot_msg_color = {'primary': bcolors.OKGREEN, 'secondary': bcolors.HEADER, 
            'warning': bcolors.WARNING, 'error': bcolors.FAIL}.get(variant, bcolors.BOLD)
    tag = f'\n:: {tag}' if tag is not None else ''
    print(f'{bcolors.OKCYAN}{prompt}{bot_msg_color}{bot_message}{bcolors.WARNING}{tag}{bcolors.ENDC}')


################################################################
# LLM bot fn
################################################################

def _random_bot_fn(message, history, **kwargs):
    from utils import get_spinner

    # Example multimodal messages
    samples = {}
    target = dict(text="I love cat", images=["https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"])
    samples['image'] = format_to_message(target)
    target = dict(audios=["https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav"])
    samples['audio'] = format_to_message(target)
    target = dict(videos=["https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"])
    samples['video'] = format_to_message(target)
    target = dict(files=["https://www.africau.edu/images/default/sample.pdf"])
    samples['pdf'] = format_to_message(target)
    target = dict(text="Hello, how can I assist you today?", 
            buttons=['Primary', dict(text='Secondary', value="the second choice"), 
                    dict(text="More", href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")])
    samples['button'] = format_to_message(target)
    target = dict(text="We found the following items:", cards=[
        dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", title="Siam Lilac Point", 
                text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads.", buttons=[]),
        dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", 
                title="Siam Lilac Point", text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads.",
                buttons=[dict(text="Search", value="/search"),
                         dict(text="More", href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")])])
    samples['card'] = format_to_message(target)
    _message = get_spinner() + " Please be patient"
    samples['spinner'] = _message
    target = dict(text="Final results goes here", collapses=[dict(
            title="Show progress", text="Scratch pad goes here", before=True)])
    samples['collapse_before'] = format_to_message(target)
    target = dict(text="Final results goes here", collapses=[dict(
            title="Show progress", text="Scratch pad goes here", before=False)])
    samples['collapse'] = format_to_message(target)
    target = dict(text="This is a reference", references=[dict(title="Sources", sources=[
        dict(text="📁 hello.pdf", link="https://hello.com", score=0.5),
        dict(text="📁 World.pdf", link="https://world.com", score=0.3),
    ])])
    samples['reference'] = format_to_message(target)

    if message in samples:
        bot_message = samples[message]
    elif message == 'all':
        bot_message = '\n'.join(samples.values())
    else:
        bot_message = random.choice(list(samples.values()))
    if 'verbose' in kwargs and kwargs['verbose']:
        _print_messages(history, message, bot_message)
    return bot_message

def _openai_bot_fn(message, history, **kwargs):
    _kwargs = dict(temperature=kwargs.get('temperature', 0))
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else None
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]

    resp = openai.ChatCompletion.create(
        model=chat_engine,
        messages=_format_messages(history, message, system=system, format='openai_chat'),
        **_kwargs,
    )
    bot_message = resp.choices[0].message.content
    if 'verbose' in kwargs and kwargs['verbose']:
        _print_messages(history, message, bot_message, system=system, tag=f'openai ({chat_engine})')
    return bot_message

def _openai_langchain_bot_fn(message, history, **kwargs):
    _kwargs = dict(temperature=kwargs.get('temperature', 0))
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else None
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model=chat_engine, **_kwargs)
    bot_message = llm(_format_messages(history, message, system=system, format='langchain_chat')).content
    if 'verbose' in kwargs and kwargs['verbose']:
        _print_messages(history, message, bot_message, system=system, tag=f'openai_langchain ({chat_engine})')
    return bot_message

def _openai_stream_bot_fn(message, history, **kwargs):
    _kwargs = dict(temperature=kwargs.get('temperature', 0))
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else None
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]

    resp = openai.ChatCompletion.create(
        model=chat_engine,
        messages=_format_messages(history, message, system=system, format='openai_chat'),
        stream=True,
        **_kwargs,
    )

    bot_message = ""
    for _resp in resp:
        if 'content' in _resp.choices[0].delta: # last resp delta is empty
            bot_message += _resp.choices[0].delta.content # need to accumulate previous message
        yield bot_message.strip() # accumulated message can easily be postprocessed
    if 'verbose' in kwargs and kwargs['verbose']:
        _print_messages(history, message, bot_message, system=system, tag=f'openai_stream ({chat_engine})')
    return bot_message

def __hf_helper_fn(chat_engine):
    if chat_engine.startswith('falcon'):
        system, user_name, bot_name, _format = DEFAULT_INSTRUCTIONS_FALCON, 'User', 'Falcon', 'plain'
    elif chat_engine.startswith('mpt'):
        system, user_name, bot_name, _format = DEFAULT_INSTRUCTIONS_MPT, 'user', 'assistant', 'chatml'
    elif chat_engine.lower().startswith('meta-llama'):
        system, user_name, bot_name, _format = DEFAULT_INSTRUCTIONS_LLAMA, 'user', 'assistant', 'llama3'
    elif chat_engine.lower().startswith('llama'):
        system, user_name, bot_name, _format = DEFAULT_INSTRUCTIONS_LLAMA, 'user', 'assistant', 'llama'
    else:
        system, user_name, bot_name, _format = DEFAULT_INSTRUCTIONS, 'Human', 'AI', 'plain'
    return system, user_name, bot_name, _format

def _hf_bot_fn(message, history, **kwargs):
    # NOTE: temperature > 0 for HF models, max_new_tokens instead of max_tokens
    _kwargs = dict(temperature=max(0.001, kwargs.get('temperature', 0.001)), 
                   max_new_tokens=kwargs.get('max_tokens', 512))

    from text_generation import Client
    chat_engine = kwargs['chat_engine']
    # API_URL = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'
    API_URL = HF_ENDPOINTS[chat_engine]
    API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    client = Client(API_URL, headers=headers)
    
    system, user_name, bot_name, _format = __hf_helper_fn(chat_engine)
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else system
    prompt = _format_messages(history, message, system=system, user_name=user_name, bot_name=bot_name, format=_format)

    bot_message = client.generate(prompt, **_kwargs).generated_text.strip().split(f'\n{user_name}')[0]
    if 'verbose' in kwargs and kwargs['verbose']:
        _print_messages(history, message, bot_message, system=system, 
                user_name=user_name, bot_name=bot_name, format=_format, tag=f'hf ({chat_engine})')
    return bot_message

def _hf_stream_bot_fn(message, history, **kwargs):
    # NOTE: temperature > 0 for HF models, max_new_tokens instead of max_tokens
    _kwargs = dict(temperature=max(0.001, kwargs.get('temperature', 0.001)), 
                   max_new_tokens=kwargs.get('max_tokens', 512))

    from text_generation import Client
    chat_engine = kwargs['chat_engine']
    # API_URL = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'
    API_URL = HF_ENDPOINTS[chat_engine]
    API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    client = Client(API_URL, headers=headers)

    system, user_name, bot_name, _format = __hf_helper_fn(chat_engine)
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else system
    prompt = _format_messages(history, message, system=system, user_name=user_name, bot_name=bot_name, format=_format)

    stop_word = f'\n{user_name}'
    bot_message = ""
    for response in client.generate_stream(prompt, **_kwargs):
        if not response.token.special:
            bot_message += response.token.text
            yield bot_message.strip().split(stop_word)[0] # stop word
            if stop_word in bot_message:
                break
    bot_message = bot_message.strip().split(stop_word)[0]
    if 'verbose' in kwargs and kwargs['verbose']:
        _print_messages(history, message, bot_message, system=system, 
                user_name=user_name, bot_name=bot_name, format=_format, tag=f'hf_stream ({chat_engine})')
    return bot_message

def _llm_call(message, history, **kwargs):
    """Call LLM with native engine (OpenAI ChatGPT or HuggingFace TGI).

    kwargs:
        * chat_engine : for llm model selection
        * system_prompt : system prompt
        * verbose : print chat history and bot response
        * other possible kwargs: temperature, max_tokens, etc.
    """
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')
    if chat_engine.startswith('gpt-'):
        bot_message = _openai_bot_fn(message, history, **kwargs)
    elif chat_engine in HF_ENDPOINTS:
        bot_message = _hf_bot_fn(message, history, **kwargs)
    else:
        bot_message = f'ERROR: Invalid chat_engine: {chat_engine}'
    return bot_message

def _llm_call_stream(message, history, **kwargs):
    """Call LLM with native engine in streaming mode (OpenAI ChatGPT or HuggingFace TGI).

    kwargs:
        * chat_engine : for llm model selection
        * system_prompt : system prompt
        * verbose : print chat history and bot response
        * other possible kwargs: temperature, max_tokens, etc.
    """
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')
    if chat_engine.startswith('gpt'):
        bot_message = _openai_stream_bot_fn(message, history, **kwargs)
    elif chat_engine in HF_ENDPOINTS:
        bot_message = _hf_stream_bot_fn(message, history, **kwargs)
    else:
        bot_message = f'ERROR: Invalid chat_engine: {chat_engine}'
    return bot_message

def _get_llm(chat_engine='gpt-3.5-turbo', **kwargs):
    # chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')

    if chat_engine.startswith('gpt'):
        _kwargs = dict(temperature=kwargs.get('temperature', 0)) # ignore max_tokens
        from langchain.chat_models import ChatOpenAI
        llm = ChatOpenAI(model=kwargs.get('chat_engine', 'gpt-3.5-turbo'), **_kwargs)
        return llm
    elif chat_engine in HF_ENDPOINTS:
        _kwargs = dict(temperature=max(0.001, kwargs.get('temperature', 0.001)), 
                    max_new_tokens=kwargs.get('max_tokens', 512))
        from langchain.llms import HuggingFaceTextGenInference
        llm = HuggingFaceTextGenInference(
            inference_server_url=HF_ENDPOINTS[chat_engine],
            stop_sequences=[f'\nUser', f'\nHuman'], # for falcon and langchain
            **_kwargs,
        )
        return llm
    else:
        raise ValueError(f"Invalid chat engine: {chat_engine}")

def _llm_call_langchain(message, history, **kwargs):
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else None
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')
    llm = _get_llm(**kwargs)

    if chat_engine.startswith('gpt'):
        # _kwargs = dict(temperature=kwargs.get('temperature', 0)) # ignore max_tokens
        # from langchain.chat_models import ChatOpenAI
        # llm = ChatOpenAI(model=kwargs.get('chat_engine', 'gpt-3.5-turbo'), **_kwargs)
        bot_message = llm(_format_messages(history, message, system=system, format='langchain_chat')).content
    elif chat_engine in HF_ENDPOINTS:
        # _kwargs = dict(temperature=max(0.001, kwargs.get('temperature', 0.001)), 
        #             max_new_tokens=kwargs.get('max_tokens', 512))
        system, user_name, bot_name, _format = __hf_helper_fn(chat_engine)
        system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else system
        prompt = _format_messages(history, message, system=system, user_name=user_name, bot_name=bot_name, format=_format)
        # from langchain.llms import HuggingFaceTextGenInference
        # llm = HuggingFaceTextGenInference(
        #     inference_server_url=HF_ENDPOINTS[chat_engine],
        #     stop_sequences=[f'\n{user_name}'],
        #     **_kwargs,
        # )
        bot_message = llm(prompt)
    else:
        bot_message = f'ERROR: Invalid chat_engine: {chat_engine}'
    if 'verbose' in kwargs and kwargs['verbose']:
        _print_messages(history, message, bot_message, system=system, tag=f'langchain ({chat_engine})')
    return bot_message

# LangChain Gradio stream
# https://gist.github.com/mortymike/70711b028311681e5f3c6511031d5d43

from threading import Thread
from queue import Queue, Empty
from collections.abc import Generator
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any

# Defined a QueueCallback, which takes as a Queue object during initialization. Each new token is pushed to the queue.
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()


# Create a function that will return our generator
def _stream(input_text, **kwargs) -> Generator:

    # Create a Queue
    q = Queue()
    job_done = object()

    llm = _get_llm(**kwargs)
    llm.streaming = True
    llm.callbacks = [QueueCallback(q)]

    # Create a funciton to call - this will run in a thread
    def task():
        if isinstance(input_text, str):
            resp = llm.predict(input_text)
        else:
            resp = llm(input_text).content
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token, content
        except Empty:
            continue

def _llm_call_stream_langchain(message, history, **kwargs):
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else None
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')
    llm = _get_llm(**kwargs)

    if chat_engine.startswith('gpt'):
        messages = _format_messages(history, message, system=system, format='langchain_chat')
        for next_token, partial_message in _stream(messages, **kwargs):
            yield partial_message
        bot_message = partial_message
    elif chat_engine in HF_ENDPOINTS:
        system, user_name, bot_name, _format = __hf_helper_fn(chat_engine)
        system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else system
        prompt = _format_messages(history, message, system=system, user_name=user_name, bot_name=bot_name, format=_format)
        for next_token, partial_message in _stream(prompt, **kwargs):
            yield partial_message
        bot_message = partial_message
    else:
        bot_message = f'ERROR: Invalid chat_engine: {chat_engine}'
    if 'verbose' in kwargs and kwargs['verbose']:
        _print_messages(history, message, bot_message, system=system, tag=f'langchain_stream ({chat_engine})')
    return bot_message
