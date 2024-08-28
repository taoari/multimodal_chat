import random
from utils.message import render_message, get_spinner

def _random_bot_fn(message, history, **kwargs):

    # Example multimodal messages
    samples = {}
    target = dict(text="I love cat", images=["https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"])
    samples['image'] = render_message(target)
    target = dict(audios=["https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav"])
    samples['audio'] = render_message(target)
    target = dict(videos=["https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"])
    samples['video'] = render_message(target)
    target = dict(files=["https://www.africau.edu/images/default/sample.pdf"])
    samples['pdf'] = render_message(target)
    target = dict(text="Hello, how can I assist you today?", 
            buttons=['Primary', dict(text='Secondary', value="the second choice"), 
                    dict(text="More", href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")])
    samples['button'] = render_message(target)
    target = dict(text="We found the following items:", cards=[
        dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", title="Siam Lilac Point", 
                text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads.", buttons=[]),
        dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", 
                title="Siam Lilac Point", text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads.",
                buttons=[dict(text="Search", value="/search"),
                         dict(text="More", href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")])])
    samples['card'] = render_message(target)
    _message = get_spinner() + " Please be patient"
    samples['spinner'] = _message
    target = dict(text="This is a reference", references=[dict(title="Sources", sources=[
        dict(text="📁 hello.pdf", link="https://hello.com", score=0.5),
        dict(text="📁 World.pdf", link="https://world.com", score=0.3),
    ])])
    samples['reference'] = render_message(target)
    samples['markdown'] = """
Hello **World**

![This is a cat](https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg)
    """
    samples['markdown_slack'] = """
Hello *World*

*Resources*
<https://hello.com|📁 hello.pdf> `score: 0.5`
    """

    if message in samples:
        bot_message = samples[message]
    elif message == 'all':
        bot_message = '<br >'.join(samples.values())
    else:
        bot_message = random.choice(list(samples.values()))
    return bot_message