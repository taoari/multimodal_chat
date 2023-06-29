import os
from bs4 import BeautifulSoup


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
    
    # exclude img, audio, video, href texts in "text"
    for tag in ['img', 'audio', 'video', 'a']:
        for unwanted in soup.select(tag):
            unwanted.extract()
    res["text"] = soup.text.strip()
    return res

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
    return msg