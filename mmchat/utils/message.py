from jinja2 import Template
from bs4 import BeautifulSoup
import os
import re

def _trim_local_file(url):
    return url.removeprefix('/file=')

def _prefix_local_file(url):
    if '://' not in url:  # Local file
        return '/file=' + url
    else:
        return url

def get_spinner(variant='primary'):
    return f"""<div class="spinner-border spinner-border-sm text-{variant}" role="status"></div>"""

MESSAGE_TEMPLATE = """
{% if msg.images %}
    <div class="images">
    {% for image in msg.images %}
        <img src="{{ _prefix_local_file(image) }}" alt="{{ _basename(image) }}">
    {% endfor %}
    </div>
{% endif %}
{% if msg.audios %}
    <div class="audios">
    {% for audio in msg.audios %}
        <audio controls>
            <source src="{{ _prefix_local_file(audio) }}">
            {{ _basename(audio) }}
        </audio>
    {% endfor %}
    </div>
{% endif %}
{% if msg.videos %}
    <div class="videos">
    {% for video in msg.videos %}
        <video controls>
            <source src="{{ _prefix_local_file(video) }}">
            {{ _basename(video) }}
        </video>
    {% endfor %}
    </div>
{% endif %}
{% if msg.files %}
    <div class="files">
    {% for file in msg.files %}
        <a href="{{ _prefix_local_file(file) }}">📁 {{ _basename(file) }}</a>
    {% endfor %}
    </div>
{% endif %}
{% if msg.buttons %}
    <div class="buttons">
    {% for button in msg.buttons %}
        {% if button is string %}
            <a class="btn btn-primary btn-chatbot text-white" value="{{ button }}">{{ button }}</a>
        {% else %}
            {% if button.value %}
                <a class="btn btn-primary btn-chatbot text-white" value="{{ button.value }}">{{ button.text }}</a>
            {% elif button.href %}
                <a class="btn btn-primary btn-chatbot-href text-white" href="{{ button.href }}">{{ button.text }}</a>
            {% endif %}
        {% endif %}
    {% endfor %}
    </div>
{% endif %}
{% if msg.cards %}
    <div class="card-group">
    {% for card in msg.cards %}
        <div class="card" style="width: 18rem;">
            <img src="{{ _prefix_local_file(card.image) }}" class="card-img-top" alt="{{ _basename(card.image) }}">
            <div class="card-body">
                <h5 class="card-title"><b>{{ card.title }}</b></h5>
                <p class="card-text text-primary">{{ card.text }}</p>
                {% if card.buttons %}
                    {% for button in card.buttons %}
                        {% if button.value %}
                            <a class="btn btn-primary btn-chatbot text-white" value="{{ button.value }}">{{ button.text }}</a>
                        {% elif button.href %}
                            <a class="btn btn-primary btn-chatbot-href text-white" href="{{ button.href }}">{{ button.text }}</a>
                        {% endif %}
                    {% endfor %}
                {% endif %}
            </div>
        </div>
    {% endfor %}
    </div>
{% endif %}
{% if msg.references %}
    <div class="references">
        {% for ref in msg.references %}
            <div class="reference">
            <b>{{ ref.title }}</b>
            <ol>
            {% for source in ref.sources %}
                <li><a href="{{ source.link }}">{{ source.text }}</a> 
                {% if source.score %}
                    <code>score: {{ source.score }}</code>
                {% endif %}
                </li>
            {% endfor %}
            </ol>
            <div>
        {% endfor %}
    </div>
{% endif %}
"""

def _basename(filepath):
    return os.path.basename(filepath)

def render_message(msg_dict, format='html'):
    if format == 'html':
        msg = re.sub(r'\n+', '\n', Template(MESSAGE_TEMPLATE, trim_blocks=True, lstrip_blocks=True).render(msg=msg_dict,
                _prefix_local_file=_prefix_local_file, _basename=_basename)).strip()
        msg = msg if 'text' not in msg_dict or not msg_dict['text'] else msg_dict['text'] + '\n' + msg
    elif format == 'plain':
        msg = msg_dict["text"] if "text" in msg_dict else ""

        files = []
        for key in ['images', 'audios', 'videos', 'files']:
            if key in msg_dict:
                files.extend(msg_dict[key])
        cards = []
        if 'cards' in msg_dict:
            for card in msg_dict['cards']:
                cards.append(f"**{card['title']}**:\n\t{card['text']}")
        buttons = []
        if 'buttons' in msg_dict:
            for btn in msg_dict['buttons']:
                buttons.append(btn if isinstance(btn, str) else btn['text'])

        msg = '\n\n'.join([msg] + files + cards + buttons).strip()

    elif format == 'speech':
        msg = msg_dict["text"] if "text" in msg_dict else ""
        
    elif format == 'json':
        import json
        msg = json.dumps(msg_dict, indent=2)
    else:
        raise ValueError(f"Invalid format: {format}")
    return msg

def parse_message(msg_html, cleanup=False):
    soup = BeautifulSoup(msg_html, "html.parser")
    result = dict(text="", images=[], audios=[], videos=[], files=[], buttons=[], cards=[], references=[])

    if elem := soup.find("div", class_="images"):
        for img in elem.find_all("img"):
            result["images"].append(_trim_local_file(img.get("src")))

    if elem := soup.find("div", class_="audios"):
        for audio in elem.find_all("audio"):
            result["audios"].append(_trim_local_file(audio.find("source").get("src")))

    if elem := soup.find("div", class_="videos"):
        for video in elem.find_all("video"):
            result["videos"].append(_trim_local_file(video.find("source").get("src")))

    if elem := soup.find("div", class_="files"):
        for file_link in elem.find_all("a"):
            result["files"].append(_trim_local_file(file_link.get("href")))

    if elem := soup.find("div", class_="buttons"):
        for button in elem.find_all("a"):
            if button.get("value"):
                if button.text == button.get("value"):
                    result["buttons"].append(button.text)
                else:
                    result["buttons"].append({"text": button.text, "value": button.get("value")})
            else:
                result["buttons"].append({"text": button.text, "href": button.get("href")})

    if elem := soup.find("div", class_="card-group"):
        for card in soup.find_all("div", class_="card"):
            card_dict = {
                "image": _trim_local_file(card.find("img").get("src")),
                "title": card.find("h5", class_="card-title").text.strip(),
                "text": card.find("p", class_="card-text").text.strip(),
                "buttons": []
            }
            for btn in card.find_all("a", class_="btn-chatbot"):
                card_dict["buttons"].append({"text": btn.text.strip(), "value": btn.get("value")})
            for link in card.find_all("a", class_="btn-chatbot-href"):
                card_dict["buttons"].append({"text": link.text.strip(), "href": link.get("href")})
            result["cards"].append(card_dict)

    if elem := soup.find("div", class_="references"):
        for ref in elem.find_all("div", class_="reference"):
            ref_title = ref.find("b").text.strip()
            sources = []
            for li in ref.find_all("li"):
                src = {"text": li.a.text.strip(), "link": li.a.get("href")}
                if "score: " in li.text:
                    src['score'] = float(li.text.split("score: ")[-1])
                sources.append(src)
            result["references"].append({"title": ref_title, "sources": sources})

    # extract div elements for clean text
    for unwanted in soup.select('div'):
        unwanted.extract()
    result["text"] = soup.get_text().strip()

    if cleanup:
        result = {k: v for k, v in result.items() if v}
    return result

# Test functions

def test_parse_message_images():
    target = dict(text="I love cat", images=["https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"])
    assert target == parse_message(render_message(target), cleanup=True)

def test_parse_message_audios():
    target = dict(audios=["https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav"])
    assert target == parse_message(render_message(target), cleanup=True)

def test_parse_message_videos():
    target = dict(videos=["https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"])
    assert target == parse_message(render_message(target), cleanup=True)

def test_parse_message_files():
    target = dict(files=["https://www.africau.edu/images/default/sample.pdf"])
    assert target == parse_message(render_message(target), cleanup=True)

def test_parse_message_buttons():
    target = dict(text="Hello, how can I assist you today?", 
            buttons=['Primary', dict(text='Secondary', value="the second choice"), 
                    dict(text="More", href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")])
    assert target == parse_message(render_message(target), cleanup=True)

def test_parse_message_spinner():
    message = get_spinner() + " Please be patient"
    target = dict(text="Please be patient")
    assert target == parse_message(message, cleanup=True)

def test_parse_message_cards():
    # For test pass, buttons must be left if empty, order in value-type then href-type
    target = dict(text="We found the following items:", cards=[
        dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", title="Siam Lilac Point", 
                text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads.", buttons=[]),
        dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", 
                title="Siam Lilac Point", text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads.",
                buttons=[dict(text="Search", value="/search"),
                         dict(text="More", href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")])])
    assert target == parse_message(render_message(target), cleanup=True)

def test_parse_message_references():
    target = dict(text="This is a reference", references=[dict(title="Sources", sources=[
        dict(text="Hello", link="https://hello.com", score=0.5),
        dict(text="World", link="https://world.com"),
    ])])
    assert target == parse_message(render_message(target), cleanup=True)

def _rerender_message(message, format='plain'):
    return render_message(parse_message(message), format=format)

def _rerender_history(history, format='plain'):
    res = []
    for msg in history:
        msg = {**msg}
        if isinstance(msg['content'], str):
            msg['content'] = _rerender_message(msg['content'], format=format)
        res.append(msg)
    return res