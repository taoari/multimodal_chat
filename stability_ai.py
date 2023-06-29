import io
import os
import warnings

from IPython.display import display
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation


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