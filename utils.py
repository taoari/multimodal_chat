import os
from bs4 import BeautifulSoup
import gradio as gr


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

def parse_message(message):
    """Parse user message in HTML format to Json inputs for LLMs."""
    soup = BeautifulSoup(message, 'html.parser')
    res = {}
    
    # extract img, audio, video, and general files
    # res["text"] = soup.text
    res["images"] = [_trim_local_file(img.get("src")) for img in soup.find_all('img')]
    res["audios"] = [_trim_local_file(audio.source.get("src")) for audio in soup.find_all('audio')]
    res["videos"] = [_trim_local_file(video.source.get("src")) for video in soup.find_all('video')]
    res["files"] = [_trim_local_file(a.get("href")) for a in soup.find_all('a')]
    res["buttons"] = [dict(text=btn.text, value=btn.value) for btn in soup.find_all('button')]
    
    # exclude img, audio, video, href texts in "text"
    for tag in ['img', 'audio', 'video', 'a', 'button']:
        for unwanted in soup.select(tag):
            unwanted.extract()
    res["text"] = soup.text.strip()
    return res

CARD_TEMPLATE = """  
  <div class="card" style="max-width: 18rem;">
    <img src="{image}" class="card-img-top" alt="{alt}">
    <div class="card-body text-primary">
      <h5 class="card-title">**{title}**</h5>
      <p class="card-text">{text}</p>
      {extra}
    </div>
  </div>
"""

COLLAPSE_TEMPLATE = """
<p>
  <a class="btn btn-outline-primary" data-bs-toggle="collapse" href="#{id}" role="button" aria-expanded="false" aria-controls="collapseExample">
    📝 {title}
  </a>
</p>
<div class="collapse" id="{id}">
  <div class="card card-body">
    {text}
  </div>
</div>"""

COLLAPSE_TEMPLATE = """
<div class="accordion accordion-flush" id="{id}-example">
  <div class="accordion-item">
    <h2 class="accordion-header" id="{id}-heading">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#{id}-collapse" aria-expanded="false" aria-controls="{id}-collapse">
         📝 {title}
      </button>
    </h2>
    <div id="{id}-collapse" class="accordion-collapse collapse" aria-labelledby="{id}-heading" data-bs-parent="#{id}-example">
      <div class="accordion-body">{text}</div>
    </div>
  </div>"""

def format_to_message(res):
    msg = res["text"] if "text" in res else ""
    if "images" in res:
        for filepath in res["images"]:
            filepath = _prefix_local_file(filepath)
            msg += f'<img src="{filepath}" alt="{os.path.basename(filepath)}"/>'
    if "audios" in res:
        for filepath in res["audios"]:
            filepath = _prefix_local_file(filepath)
            msg += f'<audio controls><source src="{filepath}">{os.path.basename(filepath)}</audio>'
    if "videos" in res:
        for filepath in res["videos"]:
            filepath = _prefix_local_file(filepath)
            msg += f'<video controls><source src="{filepath}">{os.path.basename(filepath)}</video>'
    if "files" in res:
        for filepath in res["files"]:
            filepath = _prefix_local_file(filepath)
            msg += f'<a href="{filepath}">📁 {os.path.basename(filepath)}</a>'
    if "buttons" in res:
        msg += '<br />'
        for btn in res["buttons"]:
            # btn btn-primary for bootstrap formatting, btn-chatbot to indicate it is a chatbot button
            btn_text, btn_value = (btn, btn) if isinstance(btn, str) else (btn['text'], btn['value'])
            msg += f' <a class="btn btn-primary btn-chatbot text-white" value="{btn_value}">{btn_text}</a>'
    if "cards" in res:
        cards_msg = ""
        for card in res["cards"]:
            card = card.copy()
            for key in ['image', 'title', 'text', 'extra']:
                card[key] = card[key] if key in card else ""
            cards_msg += CARD_TEMPLATE.format(alt=os.path.basename(card["image"]), **card)
        msg += f"""\n<div class="card-group">{cards_msg}</div>""".replace('\n', '')
    if "collapses" in res:
        import uuid
        collapses_msg = ""
        for collapse in res["collapses"]:
            collapses_msg += COLLAPSE_TEMPLATE.format(id=uuid.uuid4(), 
                    title=collapse['title'], text=collapse['text'])
        msg = collapses_msg + msg # collapses are usually are the front

    return msg

def get_spinner(text="Loading...", variant='primary'):
    return f"""<div class="spinner-border spinner-border-sm text-{variant}" role="status">
  <span class="visually-hidden">{text}</span>
</div>"""

def get_temp_file_name(prefix='gradio/app-', suffix='.png'):
    import tempfile
    fname = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix).name
    return fname
    
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