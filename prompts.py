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
PROMPTS["rewrite_professional"] = dict(act="Rewrite professional", prompt="""
Rewrite the following in a professional manner.
""")
PROMPTS["----"] = dict(act="----", prompt='')
PROMPTS.update(sorted({_to_command(prompt['act']): prompt for prompt in dataset['train']}.items()))


def split_prompt(prompt):
    """Split original prompt into pure instruction prompt and first request/sentence/command
    
    NOTE: may not accurate for some instances due to the diverse of the prompts"""
    for stop in ['My first', 'my first', 'First inquiry', 'First request']:
        res = prompt.rsplit(stop, maxsplit=1)
        if len(res) == 2:
            return res[0].strip(), stop + res[1]
    return prompt, None


def test_split_prompt():
    for k, v in PROMPTS.items():
        instr, first = split_prompt(v['prompt'])
        print(k, first)

def test_split_prompt_no_first():
    for k, v in PROMPTS.items():
        instr, first = split_prompt(v['prompt'])
        if first is None:
            print(k, instr)
        