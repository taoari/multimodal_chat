import pytesseract
from PIL import Image

def ocr(img_name, engine='pytesseract'):
    return pytesseract.image_to_string(Image.open(img_name))