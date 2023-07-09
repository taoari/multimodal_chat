# app.py : Multimodal Chatbot
import gradio as gr
import random
import os
from pprint import pprint
import mimetypes
from dotenv import load_dotenv

from utils import parse_message, format_to_message, get_temp_file_name

load_dotenv()  # take environment variables from .env.

# Used for Radio, CheckboxGroup, Dropdown for convert between text and display text
TEXT2DISPLAY = { 
        'auto': 'Auto', 'stabilityai': 'StabilitiyAI', 'dalle2': 'DALL·E 2', # for chat engine
    }

DISPLAY2TEXT = {v:k for k,v in TEXT2DISPLAY.items()}

#################################################################################

TITLE = "AI Create"

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

SETTINGS = {
    'chat_engine': dict(cls='Radio', choices=list(map(TEXT2DISPLAY.get, ['auto', 'stabilityai', 'dalle2'])), value=TEXT2DISPLAY['auto'], 
            interactive=True, label="Chat engine"),
}

PARAMETERS = {
    'translate': dict(cls='Checkbox', interactive=True, label="Translate", info="Translate into English may generate better results"),
    'prompt_strength': dict(cls='Slider', minimum=0, maximum=1, value=0.6, step=0.05, interactive=True, label="Prompt strength",
            info="Low strength for authenticity; high strength for creativity"),
    'gaussian_blur_radius': dict(cls='Slider', minimum=0, maximum=100, value=10, step=1, interactive=True, 
            label="Gaussian blur radius", info="Gaussian blur radius for mask"),
}

ATTACHMENTS = {
    # 'image': dict(cls='Image', type="filepath"),
    # 'webcam': dict(cls='Image', type="filepath", source="webcam"),
    # 'audio': dict(cls='Audio', type="filepath"),
    # 'microphone': dict(cls="Audio", type="filepath", source="microphone"),
    # 'video': dict(cls="Video"),
    # "file": dict(cls="File", type="file"),
    # 'model3d': dict(cls="Model3D"),
}

#################################################################################

CONFIG = {
    'upload_button': False,
}

WORKSPACE = {
    'image': dict(cls='Image', type="filepath", label="Image work on", shape=(512,512), height=512, width=512),
    'mask': dict(cls='Image', source="upload", tool="sketch", interactive=True, 
                 type='pil', label='Draw mask', shape=(512,512), height=512, width=512),
}

def user(history, msg, image, *attachments):
    _attachments = {name: filepath for name, filepath in zip(ATTACHMENTS.keys(), attachments)}
    print(_attachments)
    msg_dict = dict(text=msg, images=[], audios=[], videos=[], files=[])
    for name, filepath in _attachments.items():
        if filepath is not None:
            if name in ['image', 'webcam']:
                msg_dict['images'].append(filepath)
            elif name in ['audio', 'microphone']:
                msg_dict['audios'].append(filepath)
            elif name in ['video']:
                msg_dict['videos'].append(filepath)
            else:
                msg_dict['files'].append(filepath)
    if image is not None:
        msg_dict['images'].append(image)
    return history + [[format_to_message(msg_dict), None]], gr.update(value="", interactive=False), \
        *([gr.update(value=None, interactive=False)] * len(attachments))

def user_post():
    if len(ATTACHMENTS) == 0:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=True), *([gr.update(interactive=True)] * len(ATTACHMENTS))

def user_upload_file(msg, filepath):
    # history = history + [((file.name,), None)]
    mtype = mimetypes.guess_type(filepath.name)[0]
    if mtype.startswith('image'):
        msg += f'<img src="\\file={filepath.name}" alt="{os.path.basename(filepath.name)}"/>'
    elif mtype.startswith('audio'):
        msg += f'<audio controls><source src="\\file={filepath.name}">{os.path.basename(filepath.name)}</audio>'
    elif mtype.startswith('video'):
        msg += f'<video controls><source src="\\file={filepath.name}">{os.path.basename(filepath.name)}</video>'
    else:
        msg += f'<a href="\\file={filepath.name}">📁 {os.path.basename(filepath.name)}</a>'
    return msg


def _process_mask_image(mask_image, radius=5):
    from PIL import ImageOps
    from PIL import ImageFilter
    mask_image = ImageOps.invert(mask_image.convert('RGB'))
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=radius))
    return mask_image

def _assure_pil_image(img):
    if isinstance(img, str):
        from PIL import Image
        img = Image.open(img)
    return img


def bot(history, image, mask, *args):

    _settings = {name: value for name, value in zip(SETTINGS.keys(), args[:len(SETTINGS)])}
    _parameters = {name: value for name, value in zip(PARAMETERS.keys(), args[len(SETTINGS):])}

    # Update settings for chat mode
    # 1. convert gr.Radio and gr.CheckboxGroup from display back to text
    # 2. auto select chat engine according chat mode if it is "auto"
    _settings['chat_engine'] = DISPLAY2TEXT[_settings['chat_engine']]
    _settings['chat_engine'] = 'stabilityai' if _settings['chat_engine'] == 'auto' else _settings['chat_engine']
    
    user_message = history[-1][0]
    chat_engine = _settings['chat_engine']

    try:
        # recommend to write as external functions:
        #   bot_message = <mod>.bot(user_message, **kwargs)
        bot_message = None
        if chat_engine == 'stabilityai':

            if _parameters['translate']:
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
                    mask_image=_process_mask_image(_assure_pil_image(mask['mask']), radius=_parameters['gaussian_blur_radius']) if isinstance(mask, dict) else mask,
                    start_schedule=_parameters['prompt_strength'])
            fname = get_temp_file_name(prefix='gradio/stabilityai-', suffix='.png')
            image.save(fname)

            if _parameters['translate']:
                bot_message = format_to_message(dict(images=[fname], text=f'Translated prompt: {_user_message}'))
            else:
                bot_message = format_to_message(dict(images=[fname]))
        
    except Exception as e:
        bot_message = 'ERROR: ' + str(e)

    history[-1][1] = bot_message

    print(_settings); print(_parameters)
    pprint(history)
    return history, image

def bot_undo(history, user_message, image):
    if len(history) >= 1:
        user_message = history[-1][0]
        msg_dict = parse_message(user_message)
        if len(msg_dict['images']) > 0:
            return history[:-1], msg_dict['text'], msg_dict['images'][-1]
        return history[:-1], msg_dict['text'], None
    return history, user_message, image

def clear_chat():
    # conversation_chain.memory.clear()
    return [], "", *([None] * len(ATTACHMENTS))

def get_demo():

    def _create_from_dict(PARAMS):
        params = {}
        for name, kwargs in PARAMS.items():
            cls_ = kwargs['cls']; del kwargs['cls']
            params[name] = getattr(gr, cls_)(**kwargs)
        return params
    
    # use css and elem_id to format
    css="""#chatbot {
min-height: 600px;
}
"""

    with gr.Blocks(css=css) as demo:
        gr.HTML(f"<center><h1>{TITLE}</h1></center>")
        with gr.Accordion("Expand to see Introduction and Usage", open=False):
            gr.Markdown(f"{DESCRIPTION}")
        with gr.Row():
            with gr.Column(scale=4):
                attachments = _create_from_dict(ATTACHMENTS)
                with gr.Accordion("Settings", open=False) as settings_accordin:
                    settings = _create_from_dict(SETTINGS)
                with gr.Accordion("Parameters", open=False) as parameters_accordin:
                    parameters = _create_from_dict(PARAMETERS)
                with gr.Accordion("Workspace", open=True) as workspace_accordin:
                    workspace = _create_from_dict(WORKSPACE)
                    image, mask = workspace['image'], workspace['mask']

                with gr.Accordion("Feathered mask preview", open=False) as preview_accordin:
                    # update mask preview if mask or radius is changed
                    mask_preview = gr.Image(interactive=False, label="Feathered mask preview", height=512, width=512)
                    mask.edit(lambda x, r: _process_mask_image(x['mask'], radius=r), 
                            inputs=[mask, parameters['gaussian_blur_radius']], outputs=mask_preview)
                    parameters['gaussian_blur_radius'].change(lambda x, r: _process_mask_image(x['mask'], radius=r), 
                            inputs=[mask, parameters['gaussian_blur_radius']], outputs=mask_preview)
 
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(elem_id='chatbot')
                with gr.Row():
                    if CONFIG['upload_button']:
                        with gr.Column(scale=0.5, min_width=30):
                            upload = gr.UploadButton("📁", file_types=["image", "video", "audio", "file"])
                    with gr.Column(scale=8):
                        # NOTE: elem_id for chatbot message buttons to work
                        msg = gr.Textbox(show_label=False,
                            placeholder="Enter text and press ENTER", container=False, elem_id="inputTextBox")
                    with gr.Column(scale=1, min_width=60):
                        submit = gr.Button(value="Submit")
                    with gr.Column(scale=1, min_width=60):
                        undo = gr.Button(value="Undo")
                    with gr.Column(scale=1, min_width=60):
                        # clear = gr.ClearButton([msg, chatbot])
                        clear = gr.Button("Clear") # also clear chatbot memory

        if CONFIG['upload_button']:
            upload.upload(user_upload_file, [msg, upload], [msg], queue=False)
        msg.submit(user, [chatbot, msg, image] + list(attachments.values()), [chatbot, msg] + list(attachments.values()), queue=False).then(
            bot, [chatbot, image, mask] + list(settings.values()) + list(parameters.values()), [chatbot, image],
        ).then(
            user_post, None, [msg] + list(attachments.values()), queue=False)
        submit.click(user, [chatbot, msg, image] + list(attachments.values()), [chatbot, msg] + list(attachments.values()), queue=False).then(
            bot, [chatbot, image, mask] + list(settings.values()) + list(parameters.values()), [chatbot, image],
        ).then(
            user_post, None, [msg] + list(attachments.values()), queue=False)
        undo.click(bot_undo, [chatbot, msg, image], [chatbot, msg, image])
        clear.click(clear_chat, [], [chatbot, msg] + list(attachments.values()))

        with gr.Accordion("Examples", open=False) as examples_accordin:
            create_examples = gr.Examples(
                ['rocket ship launching from forest with flower garden under a blue sky, masterful, ghibli',
                 'crayon drawing of rocket ship launching from forest',
                ],
                inputs=msg, label="AI Create Examples",
            )
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
    # WARNING: gobal variables are shared accross users, and should be avoided.

    demo = get_demo()
    from utils import reload_javascript
    reload_javascript()
    demo.queue().launch(share=True, server_port=args.port)