import random
from pprint import pprint

from utils import parse_message, format_to_message

def _random_bot_fn(message, history, **kwargs):
    # Example multimodal messages
    samples = [
        format_to_message(dict(text="I love cat", images=["https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"])),
        format_to_message(dict(text="I hate cat", images=["https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/2560px-Felis_catus-cat_on_snow.jpg"])),
        format_to_message(dict(audios=["https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav"])),
        format_to_message(dict(videos=["https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"])),
        format_to_message(dict(files=["https://www.africau.edu/images/default/sample.pdf"])),
        format_to_message(dict(text="Hello, how can I assist you today?", buttons=['Primary', dict(text='Secondary', value="the second choice")])),
        format_to_message(dict(text="We found the following items:", cards=[
            dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", title="Siam Lilac Point", 
                 text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads, with silver-gray fur surrounding those points."),
            dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", 
                 title="Siam Lilac Point", text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads, with silver-gray fur surrounding those points.",
                 extra="""<a href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg" class="btn btn-primary btn-sm text-white">More</a>"""),
        ])),
    ]
    if 'pdf' in message:
        bot_message = samples[4]
    elif 'button' in message:
        bot_message = samples[5]
    elif 'card' in message:
        bot_message = samples[6]
    else:
        bot_message = random.choice(samples)
    return bot_message

def _openai_bot_fn(message, history, **kwargs):
    import openai, os
    openai.api_key = os.environ["OPENAI_API_KEY"]

    messages = []
    if 'system_prompt' in kwargs and kwargs['system_prompt']:
        messages.append({"role": "system", "content": kwargs['system_prompt']})
    messages.append({"role": "user", "content": message})

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=False,
    )

    return resp.choices[0].message.content

def _openai_stream_bot_fn(message, history, **kwargs):
    import openai, os
    openai.api_key = os.environ["OPENAI_API_KEY"]

    messages = []
    if 'system_prompt' in kwargs and kwargs['system_prompt']:
        messages.append({"role": "system", "content": kwargs['system_prompt']})
    messages.append({"role": "user", "content": message})

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    bot_message = ""
    for _resp in resp:
        if 'content' in _resp.choices[0].delta: # last resp delta is empty
            bot_message += _resp.choices[0].delta.content # need to accumulate previous message
        yield bot_message.strip() # accumulated message can easily be postprocessed


def _openai_langchain_bot_fn(message, history, **kwargs):
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    return llm.predict(message)