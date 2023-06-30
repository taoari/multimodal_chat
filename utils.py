import os
from bs4 import BeautifulSoup


"""User message should be in the following format for multimodal input:

text
<img src="{filepath}" alt="{os.path.basename(filepath)}"/>

NOTE:
  1. can include multiple image files
  2. local files are prefixed with '\\file=' in gradio framework, remote files should have a "://' in the src string
    2.1 local files much be under ${TEMP}/gradio/ in order to be served by gradio

Parsed user message is into a dictionary format:

{
    "text": "...",
    "images": ["<img_url>", ...],
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
    
    # extract images
    res["images"] = [_trim_local_file(img.get("src")) for img in soup.find_all('img')]
    
    # exclude img in "text"
    for tag in ['img']:
        for unwanted in soup.select(tag):
            unwanted.extract()
    res["text"] = soup.text.strip()
    return res

def format_to_message(res):
    msg = res["text"]
    if "images" in res:
        for filepath in res["images"]:
            filepath = _prefix_local_file(filepath)
            msg += f'\n<img src="{filepath}" alt="{os.path.basename(filepath)}"/>'
    return msg