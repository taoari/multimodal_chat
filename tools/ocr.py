import pytesseract
from PIL import Image

# NOTE: Windows: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

EASYOCR_READER = None

def setup_easyocr_reader():
    global EASYOCR_READER
    import easyocr
    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    EASYOCR_READER = reader

def ocr(img_name, engine='pytesseract'):
    if engine is None or engine == 'pytesseract':
        return pytesseract.image_to_string(Image.open(img_name))
    elif engine == 'easyocr':
        if EASYOCR_READER is None:
            setup_easyocr_reader()
        reader = EASYOCR_READER
        result = reader.readtext(img_name, detail=0)
        return '\n'.join(result)
    else:
        raise ValueError(f"Invalid OCR engine: {engine}")