import openai
from PIL import Image
from utils import get_temp_file_name
import numpy as np

def _preprocess_image(img, size=1024):
    if img is None:
        return img
    
    width, height = img.size
    if width <= height:
        w = size
        h = int((height/width) * w)
    else:
        h = size
        w = int((width/height) * h)

    if w != width or h != height:
        img = img.resize((w,h))

    width, height = img.size
    new_width, new_height = size, size
    if new_width != width or new_height != height:
        left, top = int((width - new_width)/2), int((height-new_height)/2)
        img = img.crop((left, top, left+new_width, top+new_height))
        print(f'Image is resized: {img.size}')
    return img

def generate(prompt, image=None, mask=None, n=1, size="1024x1024", prompt_strength=0.6):
    """
    image, mask: dalle2 Uploaded image must be a PNG and less than 4 MB.
                here we assume they are PIL images
    """
    # import pdb; pdb.set_trace()
    fname = get_temp_file_name(prefix='gradio/app-', suffix=None)
    image_fname = fname + '-image.png'
    mask_fname = fname + '-mask.png'

    if image is not None:
        image = _preprocess_image(image).convert('RGBA')
        image.save(image_fname) # convert to PNG format

        if mask is not None:
            mask_img = np.array(_preprocess_image(mask).convert('RGBA'))
            mask_img[:,:,3] = mask_img[:,:,0] # BW image to alpha channel
            mask_img[:,:,:3] = np.asarray(_preprocess_image(image).convert('RGB'))
            mask = Image.fromarray(mask_img)
            mask.save(mask_fname)
        elif prompt_strength > 0:
            image_strength = int(255*(1-prompt_strength))
            mask_img = np.array(_preprocess_image(image).convert('RGBA'))
            mask_img[:,:,3] = image_strength
            mask = Image.fromarray(mask_img)
            mask.save(mask_fname)

    if image is None and mask is None:
        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=size
        )
        image_url = response['data'][0]['url']
    else:
        response = openai.Image.create_edit(
            image=open(image_fname, "rb") if image is not None else None,
            mask=open(mask_fname, "rb") if mask is not None else None,
            prompt=prompt,
            n=n,
            size=size
        )
        image_url = response['data'][0]['url']
    return image_url

def generate_variation(image, n=1, size="1024x1024"):
    response = openai.Image.create_variation(
        image=open(image, "rb"),
        n=n,
        size=size
    )
    image_url = response['data'][0]['url']
    return image_url