import cv2
from PIL import Image, ImageDraw, ImageFont
from pyzbar.pyzbar import decode

from utils import parse_message, format_to_message

def image_resize(img, shape):
    height, width = img.shape[:2]
    if isinstance(shape, int):
        if width >= height:
            _shape = (shape, -1)
        else:
            _shape = (-1, shape)
    else:
        _shape = shape
    
    w, h = _shape
    if w == -1:
        w = int((width/height) * h)
    elif h == -1:
        h = int((height/width) * w)
    return cv2.resize(img, (w,h))

def _get_barcodes_cv2(img, draw=True):
    res = []
    for d in decode(img):
        s = d.data.decode()
        print(s)
        if draw:
            img = cv2.rectangle(img, (d.rect.left, d.rect.top),
                                    (d.rect.left + d.rect.width, d.rect.top + d.rect.height), (0, 255, 0), 3)
            img = cv2.putText(img, s, (d.rect.left, d.rect.top + d.rect.height),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        res.append(s)
    return res, img

def _get_barcodes_pil(img, draw=True):
    _draw = ImageDraw.Draw(img)
    # wget https://github.com/gasharper/linux-fonts/raw/master/arial.ttf # download for linux
    font = ImageFont.truetype('arial.ttf', size=30)  # Set 'arial.ttf' for Windows
    res = []
    for d in decode(img):
        s = d.data.decode()
        if draw:
            _draw.rectangle(((d.rect.left, d.rect.top), (d.rect.left + d.rect.width, d.rect.top + d.rect.height)),
                        outline=(0, 0, 255), width=3)
            _draw.polygon(d.polygon, outline=(0, 255, 0), width=3)
            _draw.text((d.rect.left, d.rect.top + d.rect.height), d.data.decode(),
                    (255, 0, 0), font=font)
        print(s)
        res.append(s)
    return res, img


def _get_barcodes(img, draw=True):
    if isinstance(img, Image):
        return _get_barcodes_pil(img, draw=draw)
    else:
        return _get_barcodes_cv2(img, draw=draw)


def get_barcodes(img_name, draw=True):
    img = Image.open(img_name)
    return _get_barcodes_pil(img, draw=draw)

def get_barcodes_tool(img_name):
    barcodes = get_barcodes(img_name, draw=False)[0]
    if len(barcodes) > 0:
        return f'Detected {len(barcodes)} barcodes or QR codes: {", ".join(barcodes)}'
    else:
        return 'No barcodes or QR codes detected.'