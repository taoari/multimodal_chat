import re
from datasets import load_dataset

dataset = load_dataset('fka/awesome-chatgpt-prompts')

def _to_command(display):
    res = '_'.join(display.lower().replace('/', ' ').strip().split())
    res = re.sub(r'[^A-Za-z0-9_]+', '', res)
    return res

PROMPTS = {}
PROMPTS["<none>"] = dict(act="<none>", prompt="") # always reset to <none> after a prompt
PROMPTS["english_translator"] = dict(act="English translator", prompt="""
I want you to act as an English translator. \
I will speak to you in any language and you will detect the language and translate it into English. \
I want you to only reply the translation and nothing else, do not write explanations. \
If the input is already in English, simplify reply with the original text. My first sentence is "Aloha!".
""")
PROMPTS.update({_to_command(prompt['act']): prompt for prompt in dataset['train']})