import io
import os
import warnings

from IPython.display import display
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation


def _preprocess_image(img):
    """init_image: image dimensions must be multiples of 64"""
    width, height = img.size
    new_width, new_height = int(width/64) * 64, int(height/64) * 64
    if new_width != width or new_height != height:
        left, top = int((width - new_width)/2), int((height-new_height)/2)
        img = img.crop((left, top, left+new_width, top+new_height))
    return img


def bot(user_message, history, refine=False, prompt_strength=0.6):
    from utils import parse_message, format_to_message

    msg_dict = parse_message(user_message)

    init_image = None
    if "images" in msg_dict and len(msg_dict["images"]) >= 1:
        init_image = msg_dict["images"][-1]

    # set init_image to last image in bot response if in refine mode
    if init_image is None and refine:
        for _, _bot_msg in history[:-1][::-1]:
            _msg_dict = parse_message(_bot_msg)
            if init_image is None and "images" in _msg_dict and len(_msg_dict["images"]) >= 1:
                init_image = _msg_dict["images"][-1]
                break

    if init_image is not None:
        print(f'Refine from {init_image}')

    img = generate(msg_dict["text"], 
            init_image=_preprocess_image(Image.open(init_image)) if init_image is not None else None,
            start_schedule=prompt_strength
            )
    if img is not None:
        import tempfile
        fname = tempfile.NamedTemporaryFile(prefix='gradio/stability_ai-', suffix='.png').name
        img.save(fname)
        bot_message = format_to_message(dict(images=[fname]))
    else:
        bot_message = "Sorry, stability.ai failed to generate image."

    return bot_message


def generate(prompt, init_image=None, start_schedule=0.6, width=512, height=512,
             engine="stable-diffusion-xl-beta-v2-2-2"):
    """
    Parameters:
        init_image: PIL Image
        start_schedule: the strength of our prompt in relation to our initial image.
            0.0 -- initial image, 1.0 -- prompt
    Returns:
        img : PIL Image or None
    """

    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        key=os.environ['STABILITY_API_KEY'], # API Key reference.
        verbose=True, # Print debug messages.
        engine=engine, # Set the engine to use for generation.
        # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
        # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-diffusion-xl-beta-v2-2-2 stable-inpainting-v1-0 stable-inpainting-512-v2-0
    )

    kwargs = dict(init_image=init_image, start_schedule=start_schedule) if init_image is not None else {}

    # Set up our initial generation parameters.
    answers = stability_api.generate(
        prompt=prompt,
        **kwargs,
        # init_image=img, # Assign our previously generated img as our Initial Image for transformation.
        # start_schedule=1.0, # Set the strength of our prompt in relation to our initial image.
        # seed=123467458, # If attempting to transform an image that was previously generated with our API,
        #                 # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
        steps=30, # Amount of inference steps performed on image generation. Defaults to 30. 
        cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
        width=width, # Generation width, defaults to 512 if not included.
        height=height, # Generation height, defaults to 512 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                    # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                    # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, display generated image.
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary)) # Set our resulting initial image generation as 'img2' to avoid overwriting our previous 'img' generation.
                return img