# app.py : Multimodal Chatbot
import logging
import time
from pprint import pprint
from dotenv import load_dotenv
import gradio as gr

from utils import parse_message, format_to_message, get_temp_file_name

################################################################
# Load .env and logging
################################################################

load_dotenv()  # take environment variables from .env.

logging.basicConfig(level=logging.WARN, format='%(asctime)-15s] %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p %Z")

def print(*args, **kwargs):
    sep = kwargs['sep'] if 'sep' in kwargs else ' '
    logging.warning(sep.join([str(val) for val in args])) # use level WARN for print, as gradio level INFO print unwanted messages

################################################################
# Global variables
################################################################

TITLE = "AI Create (v2)"

DESCRIPTION = """
# AI Create

## Generate Image from Text

* Enter description in text box and hit ENTER

## Image Editing

* Upload an image or use an image generated from text for the initial image to work on
* Enter editing instruction in text box and hit ENTER


## Inpainting

* Upload an image or drag and drop generated image to both "Image work on" and "Draw mask"
* Draw mask region for inpainting
* Leave text box empty and hit ENTER

## Local Image Editing

* Upload an image or drag and drop generated image to both "Image work on" and "Draw mask"
* Draw mask region for local editing
* Enter editing instruction in text box and hit ENTER


## TIPS:

* **Drag and drop** an image from Chatbot to Workspace to quickly change the image to work on
  * make sure "Image work on" and "Draw mask" background are synced if you want to do inpainting or local image editing
* Use `prompt_strength` to balance authenticity and creativity
* Adjust `gaussian_blur_radius` to better integrate with background if mask is used

## NOTE:

* "Image work on" and "Draw mask" should be merged, but due to a bug in gradio, currently you have to make sure that they are synced for inpainting and local image editing.
"""

IMAGE_KWARGS = {} # dict(height=512)

MASK_INVERT_CHAT_ENGINES = ['stabilityai']

ATTACHMENTS = {
    'image': dict(cls='Image', type="filepath", label="Image work on", **IMAGE_KWARGS),
    'draw_mask': dict(cls='Image', source="upload", tool="sketch", interactive=True, 
                type='pil', label='Draw mask', **IMAGE_KWARGS),
    'feathered_mask': dict(cls='Image', 
                interactive=False, label="Feathered mask preview", **IMAGE_KWARGS),
}

SETTINGS = {
    'chat_engine': dict(cls='Radio', choices=['auto', 'stabilityai', 'dalle2'], 
            value='auto', interactive=True, label="Chat engine"),
}

PARAMETERS = {
    'translate': dict(cls='Checkbox', interactive=True, 
            label="Translate", info="Translate into English may generate better results"),
    'prompt_strength': dict(cls='Slider', minimum=0, maximum=1, value=0.6, step=0.05, interactive=True, 
            label="Prompt strength", info="Low strength for authenticity; high strength for creativity"),
    'gaussian_blur_radius': dict(cls='Slider', minimum=0, maximum=100, value=10, step=1, interactive=True, 
            label="Gaussian blur radius", info="Gaussian blur radius for mask"),
}
    
KWARGS = {} # use for chatbot additional_inputs, do NOT change
    
################################################################
# utils
################################################################

def _create_from_dict(PARAMS, tabbed=False):
    params = {}
    for name, kwargs in PARAMS.items():
        cls_ = kwargs['cls']; del kwargs['cls']
        if not tabbed:
            params[name] = getattr(gr, cls_)(**kwargs)
        else:
            with gr.Tab(name):
                params[name] = getattr(gr, cls_)(**kwargs)
    return params


def _process_mask_image(mask_image, invert=True, radius=5):
    if mask_image is None:
        return None
    from PIL import ImageOps
    from PIL import ImageFilter
    mask_image = _assure_pil_image(mask_image)
    if invert:
        mask_image = ImageOps.invert(mask_image.convert('RGB'))
    if radius > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=radius))
    return mask_image

def _assure_pil_image(img):
    if isinstance(img, str):
        from PIL import Image
        img = Image.open(img)
    return img


def bot_fn(message, history, *args):
    kwargs = {name: value for name, value in zip(KWARGS.keys(), args)}
    kwargs['chat_engine'] = 'stabilityai' if kwargs['chat_engine'] == 'auto' else kwargs['chat_engine']
   
    user_message = parse_message(message)['text']
    chat_engine = kwargs['chat_engine']

    try:
        bot_message = None
        mask = kwargs['draw_mask']
        image = kwargs['image']
        mask_image = mask['mask'] if isinstance(mask, dict) else mask

        if chat_engine == 'stabilityai':

            if kwargs['translate']:
                from langchain.chat_models import ChatOpenAI
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0,
                    verbose=True,
                )
                _user_message = llm.predict(f'Translate the following sentence into English (return original text if it is already in English): {user_message}')
            else:
                _user_message = user_message

            import stability_ai

            image = stability_ai.generate(_user_message, 
                    init_image=_assure_pil_image(image), # not arbitrary resolution
                    mask_image=_process_mask_image(_assure_pil_image(mask_image), radius=kwargs['gaussian_blur_radius'], 
                            invert=True),
                    start_schedule=kwargs['prompt_strength'])
            fname = get_temp_file_name(prefix='gradio/stabilityai-', suffix='.png')
            image.save(fname)

            if kwargs['translate']:
                bot_message = format_to_message(dict(images=[fname], text=f'Translated prompt: {_user_message}'))
            else:
                bot_message = format_to_message(dict(images=[fname]))

        elif chat_engine == 'dalle2':

            from PIL import Image
            import requests
            from io import BytesIO

            import dalle2
            image_url = dalle2.generate(user_message,
                    image=_assure_pil_image(image),
                    mask=_process_mask_image(_assure_pil_image(mask_image), radius=0, 
                            invert=True), # NOTE: dalle2 should not blur on mask image, prompt_strength not work
                    prompt_strength=kwargs['prompt_strength'])
            
            # NOTE: image_url can not be shown in Image component
            fname = get_temp_file_name(prefix='gradio/dalle2-', suffix='.png')
            image = Image.open(BytesIO(requests.get(image_url).content)) # for return
            image.save(fname)

            bot_message = format_to_message(dict(images=[fname]))
        
    except Exception as e:
        import traceback
        bot_message = traceback.format_exc()
        # bot_message = 'ERROR: ' + str(e)

    return bot_message, fname

################################################################
# Gradio app
################################################################

def get_demo():

    # use css and elem_id to format
    css="""#chatbot {
min-height: 600px;
}"""

    # NOTE: can not be inside another gr.Blocks
    # _chatbot = gr.Chatbot(elem_id="chatbot", avatar_images = ("assets/user.png", "assets/bot.png"))
    # _textbox = gr.Textbox(container=False, show_label=False, placeholder="Type a message...", scale=10, elem_id='inputTextBox', min_width=300)

    with gr.Blocks(css=css) as demo:
        # title
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        # description
        with gr.Accordion("Expand to see Introduction and Usage", open=False):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            # attachements, settings, and parameters
            with gr.Column(scale=1):
                attachments = _create_from_dict(ATTACHMENTS, tabbed=True)
                with gr.Accordion("Settings", open=False) as settings_accordin:
                    settings = _create_from_dict(SETTINGS)
                with gr.Accordion("Parameters", open=False) as parameters_accordin:
                    parameters = _create_from_dict(PARAMETERS)

            with gr.Column(scale=9):
                # chatbot
                global KWARGS
                KWARGS = {**attachments, **settings, **parameters}
                import chat_interface
                chatbot = chat_interface.ChatInterface(bot_fn, # chatbot=_chatbot, textbox=_textbox,
                        additional_inputs=list(KWARGS.values()),
                        additional_outputs=[KWARGS['image']],
                        upload_btn="📁",
                        retry_btn="Retry", undo_btn="Undo", clear_btn="Clear",
                    )
                # chatbot.textbox.elem_id = 'inputTextBox'
                chatbot.chatbot.avatar_images = ("assets/user.png", "assets/bot.png")

                # examples
                with gr.Accordion("Examples", open=False) as examples_accordin:
                    create_examples = gr.Examples(
                        ['rocket ship launching from forest with flower garden under a blue sky, masterful, ghibli',
                        'crayon drawing of rocket ship launching from forest',
                        ],
                        inputs=chatbot.textbox, label="AI Create Examples",
                    )
            # additional handlers
            # for name, attach in attachments.items():
            #     if hasattr(chatbot, '_upload_fn') and isinstance(attach, gr.Image):
            #         attach.change(chatbot._upload_fn,
            #             [chatbot.textbox, attach], 
            #             [chatbot.textbox], queue=False, api_name=False)
            mask_preview = attachments['feathered_mask']
            mask = attachments['draw_mask']
            mask.edit(lambda x, r: _process_mask_image(x['mask'], radius=r, invert=True), 
                    inputs=[mask, parameters['gaussian_blur_radius']], outputs=mask_preview)
            parameters['gaussian_blur_radius'].change(lambda x, r: _process_mask_image(x['mask'], radius=r, invert=True), 
                    inputs=[mask, parameters['gaussian_blur_radius']], outputs=mask_preview)
    return demo

def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Multimodal Chatbot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-p', '--port', default=7860, type=int,
        help='port number.')
    parser.add_argument(
        '--debug', action='store_true', 
        help='debug mode.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    demo = get_demo()
    from utils import reload_javascript
    reload_javascript()
    demo.queue().launch(share=True, server_port=args.port)
