import os
from bs4 import BeautifulSoup
import gradio as gr
from jinja2 import Template


"""User message should be in the following format for multimodal input:

text
<img src="{filepath}" alt="{os.path.basename(filepath)}"/>
<audio controls><source src="{filepath}">{os.path.basename(filepath)}</audio>
<video controls><source src="{filepath}">{os.path.basename(filepath)}</video>
<a href="{filepath}">📁 {os.path.basename(filepath)}</a>

NOTE:
  1. can include multiple images, audios, videos, or files
  2. local files are prefixed with '\\file=' in gradio framework, remote files should have a "://' in the src string
    2.1 local files much be under ${TEMP}/gradio/ in order to be served by gradio

Parsed user message is in a dictionary format:

{
    "text": "...",
    "images": ["<img_url>", ...],
    "audios": ["<audio_url>", ...],
    ...
}
"""

def _trim_local_file(url):
    return url.lstrip('\\file=')

def _prefix_local_file(url):
    if '://' not in url: # localfile
        return '\\file=' + url
    else:
        return url

CARD_TEMPLATE = """  
  <div class="card" style="max-width: 18rem;">
    <img src="{image}" class="card-img-top" alt="{alt}">
    <div class="card-body text-primary">
      <h5 class="card-title">**{title}**</h5>
      <p class="card-text">{text}</p>
      <div>{extra}</div>
    </div>
  </div>
"""

# # collapse NOTE: add class="btn btn-outline-primary" for button
# COLLAPSE_TEMPLATE = """
# <p>
#   <a data-bs-toggle="collapse" href="#{id}" role="button" aria-expanded="false" aria-controls="collapseExample">
#     📝 {title}
#   </a>
# </p>
# <div class="collapse" id="{id}">
#   <div class="card card-body">
#     {text}
#   </div>
# </div>"""

# # accordin
# COLLAPSE_TEMPLATE = """
# <div class="accordion accordion-flush" id="{id}-example">
#   <div class="accordion-item">
#     <h2 class="accordion-header" id="{id}-heading">
#       <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#{id}-collapse" aria-expanded="false" aria-controls="{id}-collapse">
#          📝 {title}
#       </button>
#     </h2>
#     <div id="{id}-collapse" class="accordion-collapse collapse" aria-labelledby="{id}-heading" data-bs-parent="#{id}-example">
#       <div class="accordion-body">{text}</div>
#     </div>
#   </div>
# </div>"""

# collapse details
COLLAPSE_TEMPLATE = """
<details id="{id}">
<summary>{title}</summary>
{text}
</details>"""

REFERENCE_TEMPLATE = """
<div class="reference">
<h6><b>{{ title }}</b></h6>
<ol>
{% for ref in sources -%}
<li><a href="{{ ref.link }}">{{ ref.text }} <span class="badge badge-light text-info">score:  {{ ref.score }}</span></a></li>
{% endfor -%}
</ol>
</div>
"""

def format_to_message(msg_dict, _format='html'):
    if _format == 'html':
        msg = msg_dict["text"] if "text" in msg_dict else ""
        if "images" in msg_dict and len(msg_dict["images"]) > 0:
            for filepath in msg_dict["images"]:
                filepath = _prefix_local_file(filepath)
                msg += f'<img src="{filepath}" alt="{os.path.basename(filepath)}"/>'
        if "audios" in msg_dict and len(msg_dict["audios"]) > 0:
            for filepath in msg_dict["audios"]:
                filepath = _prefix_local_file(filepath)
                msg += f'<audio controls><source src="{filepath}">{os.path.basename(filepath)}</audio>'
        if "videos" in msg_dict and len(msg_dict["videos"]) > 0:
            for filepath in msg_dict["videos"]:
                filepath = _prefix_local_file(filepath)
                msg += f'<video controls><source src="{filepath}">{os.path.basename(filepath)}</video>'
        if "files" in msg_dict and len(msg_dict["files"]) > 0:
            for filepath in msg_dict["files"]:
                filepath = _prefix_local_file(filepath)
                msg += f'<a href="{filepath}">📁 {os.path.basename(filepath)}</a>'
        if "buttons" in msg_dict and len(msg_dict["buttons"]) > 0:
            msg += '<br />'
            for btn in msg_dict["buttons"]:
                # btn btn-primary for bootstrap formatting, btn-chatbot to indicate it is a chatbot button
                btn = dict(text=btn, value=btn) if isinstance(btn, str) else btn
                if "value" in btn:
                    msg += f""" <a class="btn btn-primary btn-chatbot text-white" value="{btn['value']}">{btn['text']}</a>"""
                else:
                    msg += f""" <a class="btn btn-primary btn-chatbot-href text-white" href="{btn['href']}">{btn['text']}</a>"""

        if "cards" in msg_dict and len(msg_dict["cards"]) > 0:
            cards_msg = ""
            for card in msg_dict["cards"]:
                _card = {}
                for key in ['image', 'title', 'text', 'extra']:
                    _card[key] = card[key] if key in card else ""
                if "buttons" in card:
                    _card["extra"] += format_to_message(dict(buttons=card["buttons"]))
                cards_msg += CARD_TEMPLATE.format(alt=os.path.basename(card["image"]), **_card)
            msg += f"""\n<div class="card-group">{cards_msg}</div>""".replace('\n', '')
        if "collapses" in msg_dict and len(msg_dict["collapses"]) > 0:
            import uuid
            collapses_msg_pre = ""
            for collapse in msg_dict["collapses"]:
                before = 'before' in collapse and collapse['before']
                _collapse = COLLAPSE_TEMPLATE.format(id=str(uuid.uuid4()) + ("-before" if before else ""), 
                        title=collapse['title'], text=collapse['text'])
                if before:
                    collapses_msg_pre += _collapse
                else:
                    msg += _collapse
            msg = collapses_msg_pre + msg # collapses are usually are the front
        if "references" in msg_dict:
            for ref in msg_dict['references']:
                msg += Template(REFERENCE_TEMPLATE).render(**ref)

    elif _format == 'plain':
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
        # ignore buttons and collapses

    elif _format == 'speech':
        msg = msg_dict["text"] if "text" in msg_dict else ""
        # only text, ignore files, cards, buttons and collapses
        
    elif _format == 'json':
        import json
        msg = json.dumps(msg_dict, indent=2)
    else:
        raise ValueError(f"Invalid format: {_format}")

    return msg

def _parse_and_delete(soup):
    res = dict(buttons=[], cards=[], collapses=[], references=[])
    # reference
    for elem in soup.find_all('div', class_='reference'):
        title = elem.h6.text
        sources = []
        for li in elem.find_all('a'):
            src = dict(link=li.get('href'), score=float(li.span.text.split(': ')[-1]), text=li.contents[0].strip())
            sources.append(src)
        res['references'].append(dict(title=title, sources=sources))
        elem.extract()
    # collapses
    for elem in soup.find_all("details"):
        collapse = dict(title=elem.summary.text.strip(),
                before='before' in elem.get("id"))
        elem.summary.extract()
        collapse["text"]=elem.text.strip()

        res['collapses'].append(collapse)
        elem.extract()
    # cards : must before buttons as can contain buttons
    for elem in soup.find_all("div", class_="card"):
        card = dict(image=_trim_local_file(elem.img.get("src")),
                    title=elem.div.h5.text.lstrip("**").rstrip("**"),
                    text=elem.div.p.text)
        buttons = _parse_and_delete(elem.div.div)
        card['buttons'] = buttons['buttons']
        res['cards'].append(card)
        elem.extract()
    # buttons
    for elem in soup.find_all('a', class_='btn-chatbot'):
        btn = dict(text=elem.text.strip(), value=elem.get("value"))
        btn = btn["text"] if btn["text"] == btn["value"] else btn
        res['buttons'].append(btn)
        elem.extract()
    for elem in soup.find_all('a', class_='btn-chatbot-href'):
        btn = dict(text=elem.text.strip(), href=elem.get("href"))
        res['buttons'].append(btn)
        elem.extract()
    return res

def parse_message(message, cleanup=False):
    """Parse user message in HTML format to Json inputs for LLMs."""
    soup = BeautifulSoup(message, 'html.parser')
    res = _parse_and_delete(soup)
    # extract img, audio, video, and general files
    # res["text"] = soup.text
    res["images"] = [_trim_local_file(img.get("src")) for img in soup.find_all('img')]
    res["audios"] = [_trim_local_file(audio.source.get("src")) for audio in soup.find_all('audio')]
    res["videos"] = [_trim_local_file(video.source.get("src")) for video in soup.find_all('video')]
    res["files"] = [_trim_local_file(a.get("href")) for a in soup.find_all('a')]
    # exclude img, audio, video, href texts in "text"
    for tag in ['img', 'audio', 'video', 'a', 'button']:
        for unwanted in soup.select(tag):
            unwanted.extract()
    res["text"] = soup.text.strip()
    if cleanup:
        res = {k: v for k, v in res.items() if v}
    return res

def test_parse_message_image():
    target = dict(text="I love cat", images=["https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"])
    assert target == parse_message(format_to_message(target), cleanup=True)

def test_parse_message_audio():
    target = dict(audios=["https://upload.wikimedia.org/wikipedia/commons/2/28/Caldhu.wav"])
    assert target == parse_message(format_to_message(target), cleanup=True)

def test_parse_message_video():
    target = dict(videos=["https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"])
    assert target == parse_message(format_to_message(target), cleanup=True)

def test_parse_message_file():
    target = dict(files=["https://www.africau.edu/images/default/sample.pdf"])
    assert target == parse_message(format_to_message(target), cleanup=True)

def test_parse_message_button():
    target = dict(text="Hello, how can I assist you today?", 
            buttons=['Primary', dict(text='Secondary', value="the second choice"), 
                    dict(text="More", href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")])
    assert target == parse_message(format_to_message(target), cleanup=True)

def test_parse_message_card():
    # For test pass, buttons must be left if empty, order in value-type then href-type
    target = dict(text="We found the following items:", cards=[
        dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", title="Siam Lilac Point", 
                text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads.", buttons=[]),
        dict(image="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg", 
                title="Siam Lilac Point", text="The lilac point Siamese cat usually has a pale pink nose and pale pink paw pads.",
                buttons=[dict(text="Search", value="/search"),
                         dict(text="More", href="https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")])])
    assert target == parse_message(format_to_message(target), cleanup=True)

def test_parse_message_spinner():
    message = get_spinner() + " Please be patient"
    target = dict(text="Please be patient")
    assert target == parse_message(message, cleanup=True)

def test_parse_message_collapse():
    target = dict(text="Final results goes here", collapses=[dict(
            title="Show progress", text="Scratch pad goes here", before=True)])
    assert target == parse_message(format_to_message(target), cleanup=True)
    target = dict(text="Final results goes here", collapses=[dict(
            title="Show progress", text="Scratch pad goes here", before=False)])
    assert target == parse_message(format_to_message(target), cleanup=True)

def test_parse_message_reference():
    target = dict(text="This is a reference", references=[dict(title="Sources", sources=[
        dict(text="Hello", link="https://hello.com", score=0.5),
        dict(text="World", link="https://world.com", score=0.3),
    ])])
    assert target == parse_message(format_to_message(target), cleanup=True)

def _reformat_message(message, _format='plain'):
    if _format is None or _format == 'auto':
        return message
    return format_to_message(parse_message(message, cleanup=True), _format=_format)

def _reformat_history(history, _format='plain'):
    if _format is None or _format == 'auto':
        return history
    return [[_reformat_message(human, _format=_format), _reformat_message(ai, _format=_format)] for human, ai in history]

def get_spinner(variant='primary'):
    return f"""<div class="spinner-border spinner-border-sm text-{variant}" role="status"></div>"""

def get_temp_file_name(prefix='gradio/app-', suffix='.png'):
    import tempfile
    fname = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix).name
    return fname

def fix_exif_orientation(filepath, outpath=None):
    from PIL import Image, ExifTags

    try:
        outpath = filepath if outpath is None else outpath
        image=Image.open(filepath)

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        exif = image._getexif()

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)

        image.save(outpath)
        image.close()
    except:
        # cases: image don't have getexif
        pass
    
def reload_javascript():
    """reload custom javascript. The following code enables bootstrap css and makes chatbot message buttons responsive.
    """
    print("Reloading javascript...")
    js = """
<!-- for bootstrap -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-9ndCyUaIbzAi2FUVXJi0CjmCapSmO7SnpJef0486qhLnuZ2cdeRhO02iuK6FUUVM" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-geWF76RCwLtnZ8qwWowPQNguL3RmwHVBC9FhGdlKrxdiJJigb/j/68SIy3Te4Bkz" crossorigin="anonymous"></script>

<!-- for message buttons to work -->
<script>
function registerMessageButtons() {
	const collection = document.querySelectorAll(".btn-chatbot");
	for (let i = 0; i < collection.length; i++) {
      // NOTE: gradio use .value instead of .innerHTML for gr.Textbox
	  collection[i].onclick=function() {
        elem = document.getElementById("inputTextBox").getElementsByTagName('textarea')[0];
        elem.value = collection[i].getAttribute("value"); // use value instead of innerHTML
        elem.dispatchEvent(new Event('input', {
            view: window,
            bubbles: true,
            cancelable: true
            }))
        };  
	}
}
// need to make sure registerMessageButtons() is executed all the time as new message can come out;
var intervalId = window.setInterval(function(){
  registerMessageButtons();
}, 1000);
</script>
"""
    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        # soup = BeautifulSoup(res.body, 'html.parser')
        # # NOTE: gradio UI is rendered by JavaScript, so can not find btn-chatbot
        # res.body = str(soup).replace('</html>', f'{js}</html>').encode('utf8')
        res.body = res.body.replace(b'</html>', f'{js}</html>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response

GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse