import os

from langchain.llms import HuggingFaceTextGenInference
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain

HF_ENDPOINTS = {}

DEFAULT_INSTRUCTIONS = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""

DEFAULT_INSTRUCTIONS_FALCON = """The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, and a human user, called User. In the following interactions, User and Falcon will converse in natural language, and Falcon will answer User's questions. Falcon was built to be respectful, polite and inclusive. Falcon was built by the Technology Innovation Institute in Abu Dhabi. Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins.
{history}
User: {input}
Falcon:"""

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

def parse_endpoints_from_environ():
    global HF_ENDPOINTS
    for name, value in os.environ.items():
        if name.startswith('HF_INFERENCE_ENDPOINT_'):
            HF_ENDPOINTS[name[len('HF_INFERENCE_ENDPOINT_'):].lower()] = value

def register_endpoints_to_text2display(TEXT2DISPLAY):
    for _name, _value in HF_ENDPOINTS.items():
        if 'huggingface.co' in _value:
            TEXT2DISPLAY[_name] = f'HF ({_name})'
        else:
            TEXT2DISPLAY[_name] = f'Self-Host ({_name})'

def _format_history(history=[], bot_name='AI', user_name='Human'):
    _history = []
    for user, bot in history:
        if user:
            _history.append(f'{user_name}: {user}')
        if bot:
            _history.append(f'{bot_name}: {bot}')
    return '\n'.join(_history)


def bot(history, chat_engine, chat_state, _parameters):
    user_message = history[-1][0]
    
    instructions, user_name, bot_name = DEFAULT_INSTRUCTIONS, 'Human', 'AI'
    if 'falcon' in chat_engine:
        instructions, user_name, bot_name = DEFAULT_INSTRUCTIONS_FALCON, 'User', 'Falcon'

    if chat_engine == 'openai':
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=_parameters['temperature'],
            stop=[f'\n{user_name}', '<|endoftext|>'],
            verbose=True,
        )
    elif chat_engine in HF_ENDPOINTS:
        llm = HuggingFaceTextGenInference(
            inference_server_url=HF_ENDPOINTS[chat_engine],
            temperature=_parameters['temperature'],
            max_new_tokens=_parameters['max_new_tokens'],
            stop_sequences=[f'\n{user_name}', '<|endoftext|>'],
            verbose=True,
        )
    else:
        raise ValueError(f"Invalid chat engine: {chat_engine}")

    prompt = None
    bot_message = None
    if chat_state == 'stateless':
        prompt = user_message
    elif chat_state == 'stateless_prompted':
        prompt = instructions.format(history="", input=user_message)
    elif chat_state == 'history':
        prompt = instructions.format(
            history=_format_history(history[:-1], user_name=user_name, bot_name=bot_name), 
            input=user_message)
        
    if prompt is not None:
        bot_message = llm.predict(prompt).strip() # follow OpenAI convention
        print(f'{bcolors.OKCYAN}{prompt} {bcolors.OKGREEN}{bot_message}{bcolors.ENDC}')

    return bot_message
            
    # llm = HuggingFaceTextGenInference(
    #     inference_server_url="https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
    # )

    # print(repr(llm))
    
    # HuggingFaceTextGenInference(cache=None, verbose=False, callbacks=None, callback_manager=None, tags=None, 
    # max_new_tokens=512, top_k=None, top_p=0.95, typical_p=0.95, temperature=0.8, 
    # repetition_penalty=None, stop_sequences=[], seed=None, 
    # inference_server_url='https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct', 
    # timeout=120, server_kwargs={}, stream=False, 
    # client=<text_generation.client.Client object at 0x000001FABCD86F80>, 
    # async_client=<text_generation.client.AsyncClient object at 0x000001FAC9547880>)

    # llm = ChatOpenAI(
    #     model_name="gpt-3.5-turbo",
    #     stop=['\nHuman:', '<|endoftext|>'],
    # )

    # print(repr(llm)) # NOTE: prints private tokens
    # # ChatOpenAI(cache=None, verbose=False, callbacks=None, callback_manager=None, tags=None, 
    # # client=<class 'openai.api_resources.chat_completion.ChatCompletion'>, 
    # # model_name='gpt-3.5-turbo', temperature=0.7, model_kwargs={'stop': ['\nHuman:', '<|endoftext|>']}, 
    # # openai_api_key='...', openai_api_base='', openai_organization='', openai_proxy='', 
    # # request_timeout=None, max_retries=6, streaming=False, n=1, max_tokens=None)

    # llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{input}"), verbose=True)
    # conversation_chain = ConversationChain(llm=llm, verbose=True)
    # print(conversation_chain.prompt.template)