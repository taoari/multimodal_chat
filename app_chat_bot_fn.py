# app.py : Multimodal Chatbot
import gradio as gr
import random
from pprint import pprint
from dotenv import load_dotenv
import os
import json, requests

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
    user_name='user', bot_name='assistant', format='plain'):
    """history is list of tuple [(user_msg, bot_msg), ...]"""
    prompt = _format_messages(history, message, system=system, user_name=user_name, bot_name=bot_name, format=format)
    print(f'{bcolors.OKCYAN}{prompt}{bcolors.OKGREEN}{bot_message}{bcolors.ENDC}')


################################################################
# LLM bot fn
################################################################

def _openai_bot_fn(message, history, **kwargs):
    _kwargs = dict(temperature=kwargs.get('temperature', 0))
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else None
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]

    resp = openai.ChatCompletion.create(
        model=kwargs.get('chat_engine', 'gpt-3.5-turbo'),
        messages=_format_messages(history, message, system=system, format='openai_chat'),
        **_kwargs,
    )
    return resp.choices[0].message.content

def _openai_langchain_bot_fn(message, history, **kwargs):
    _kwargs = dict(temperature=kwargs.get('temperature', 0))
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else None
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model=kwargs.get('chat_engine', 'gpt-3.5-turbo'), **_kwargs)
    # return llm.predict(_format_messages(history, message, system=system, format='langchain_chat'))
    return llm(_format_messages(history, message, system=system, format='langchain_chat')).content

def _openai_stream_bot_fn(message, history, **kwargs):
    _kwargs = dict(temperature=kwargs.get('temperature', 0))
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else None
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]

    resp = openai.ChatCompletion.create(
        model=kwargs.get('chat_engine', 'gpt-3.5-turbo'),
        messages=_format_messages(history, message, system=system, format='openai_chat'),
        stream=True,
        **_kwargs,
    )

    bot_message = ""
    for _resp in resp:
        if 'content' in _resp.choices[0].delta: # last resp delta is empty
            bot_message += _resp.choices[0].delta.content # need to accumulate previous message
        yield bot_message.strip() # accumulated message can easily be postprocessed
    _print_messages(history, message, bot_message, system=system)

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

    if chat_engine.startswith('falcon'):
        system, user_name, bot_name, _format = DEFAULT_INSTRUCTIONS_FALCON, 'User', 'Falcon', 'plain'
    elif chat_engine.startswith('mpt'):
        system, user_name, bot_name, _format = DEFAULT_INSTRUCTIONS_MPT, 'user', 'assistant', 'chatml'
    elif chat_engine.lower().startswith('llama'):
        system, user_name, bot_name, _format = DEFAULT_INSTRUCTIONS_LLAMA, 'user', 'assistant', 'llama'
    else:
        system, user_name, bot_name, _format = DEFAULT_INSTRUCTIONS, 'Human', 'AI', 'plain'
    
    system = kwargs['system_prompt'] if 'system_prompt' in kwargs and kwargs['system_prompt'] else system
    prompt = _format_messages(history, message, system=system, user_name=user_name, bot_name=bot_name, format=_format)

    bot_message = ""
    for response in client.generate_stream(prompt, **_kwargs):
        if not response.token.special:
            bot_message += response.token.text
            yield bot_message.strip().split(f'\n{user_name}')[0] # stop word

    bot_message = bot_message.strip().split(f'\n{user_name}')[0]
    _print_messages(history, message, bot_message, system=system, 
            user_name=user_name, bot_name=bot_name, format=_format)
    # bot_message = client.generate(prompt, **_kwargs).generated_text.strip().split(f'\n{user_name}')[0]
    # _print_messages(history, message, bot_message, system=system, 
    #         user_name=user_name, bot_name=bot_name, format=_format)
    # return bot_message

def _bot_slash_fn(message, history, **kwargs):
    bot_message = message
    return bot_message

def _llm_call(message, history, **kwargs):
    chat_engine = kwargs.get('chat_engine', 'gpt-3.5-turbo')
    if chat_engine.startswith('gpt'):
        bot_message = _openai_stream_bot_fn(message, history, **kwargs)
        # bot_message = _openai_langchain_bot_fn(message, history, **kwargs)
    elif chat_engine in HF_ENDPOINTS:
        bot_message = _hf_stream_bot_fn(message, history, **kwargs)
    else:
        bot_message = f'ERROR: Invalid chat_engine: {chat_engine}'
    return bot_message
