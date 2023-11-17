from gradio_client import Client

def _upload(client, fname, message=""):
    result = client.predict(
            message,	# str in 'Message' Textbox component
            fname,	# str (filepath or URL to file) in '📁' Uploadbutton component
            api_name="/upload"
    )
    return result

def _chat(client, message, chat_engine="auto"):
    result = client.predict(
            message,	# str  in 'Message' Textbox component
            None,	# filepath  in 'Input' Image component
            "",	# str  in 'System prompt' Textbox component
            chat_engine,	# Literal[auto, random, echo, gpt-3.5-turbo]  in 'Chat engine' Radio component
            "auto",	# Literal[auto, html, plain, json]  in 'Bot response format' Radio component
            api_name="/chat"
    )
    return result


class Chatbot(object):

    def __init__(self, server_url="http://localhost:7860/", **kwargs):
        self.server_url = server_url
        self.client = Client(server_url)
        self.init_kwargs = kwargs

    def _upload(self, fname, message=""):
        bot_message = _upload(self.client, fname, message=message)
        return bot_message
    
    def _chat(self, message, **kwargs):
        bot_message = _chat(self.client, message, **kwargs)
        return bot_message
    
    def _set(self, **kwargs):
        self.kwargs.update(kwargs)
    
    def chat(self, message, fname=None, **kwargs):
        try:
            if fname is not None:
                user_message = message + self._upload(fname)
            else:
                user_message = message
            bot_message = self._chat(user_message, **kwargs, **self.init_kwargs)
        except Exception as e:
            bot_message = f"ERROR: {e}"
        return bot_message

def _print(user, bot, _format='plain'):
    from utils import _reformat_message
    if _format != 'html':
        user = _reformat_message(user, _format=_format)
        bot = _reformat_message(bot, _format=_format)
    print(f'👤 user:\n{user}\n🤖 assistant:\n{bot}\n')

def test_chatbot_api_call():
    chatbot = Chatbot("http://localhost:7860/")

    # random
    user = "test_files/cat.jpg"
    bot = chatbot.chat("", fname=user)
    _print(user, bot)
    
    user = 'image'
    bot = chatbot.chat(user)
    _print(user, bot)

    user = 'card'
    bot = chatbot.chat(user)
    _print(user, bot)

    user = 'collapse'
    bot = chatbot.chat(user)
    _print(user, bot)

    # openai
    user = 'hello'
    bot = chatbot.chat(user, chat_engine='gpt-3.5-turbo')
    _print(user, bot)

    user = 'Write me a Python code to calculate Fibonacci numbers.'
    bot = chatbot.chat(user, chat_engine='gpt-3.5-turbo')
    _print(user, bot)

def chatbot_shell():
    print("""Welcom to chatbot! 
Usage:
  * Use "/stop" to stop
  * Use "/upload <filename>" to upload a file
  * Use "/reset" to reset the chatbot
""")
    chatbot = Chatbot("http://localhost:7860/")
    _format = 'json'

    while True:
        message = input('#### Type message here: ')
        if message == '/stop':
            break
        elif message == '/reset':
            chatbot = Chatbot("http://localhost:7860/")
        elif message.startswith('/set'):
            k, v = message[len('/set '):].split('=')
            chatbot.init_kwargs[k] = v
        elif message.startswith('/upload'):
            fname = message.split(' ', maxsplit=1)[1].strip()
            bot = chatbot.chat("", fname)
            _print(message, bot, _format)
        else:
            bot = chatbot.chat(message)
            _print(message, bot, _format)            

if __name__ == '__main__':

    # test_chatbot_api_call()
    chatbot_shell()
