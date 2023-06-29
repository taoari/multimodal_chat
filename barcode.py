import cv2
import tempfile
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

def bot(user_message, DEBUG=False):
    """ Input user_message and output bot_message
    """
    barcodes, annotated_img = get_barcodes_from_message(user_message)
    if barcodes is not None and len(barcodes) > 0:
        if DEBUG:
            fname = tempfile.NamedTemporaryFile(prefix='gradio/barcode-', suffix='.jpg').name
            cv2.imwrite(fname, annotated_img)
            res = {'text': "**DBUG**: We have detected {} barcode/QR code:\n* {}\n".format(len(barcodes), '\n* '.join(barcodes)),
                "images": [fname]}
            bot_message = [format_to_message(res)]
        else:
            bot_message = []

        from walgreens import walgreens_search_by_barcode
        for barcode in barcodes:
            import re
            if re.match('\d+', barcode):
                df_match = walgreens_search_by_barcode(barcode)
                if df_match is not None:
                    product = df_match.iloc[0]
                    # str(product[['title', 'price', 'image_link', 'link', 'availability', 'InStock']])
                    _bot_msg = f"We found **{product['title']}** in our store that is *{product['availability']}* at price *{product['price']}*."
                    _bot_msg += f' You can buy this product from {product["link"]}'
                    _bot_msg += f'\n<img src="{product["image_link"]}" />'
                    
                    bot_message.append(_bot_msg)
                else:
                    bot_message.append(f"Sorry, we can not found this product with barcode *{barcode}* in our store!")
            else:
                bot_message.append(f"Sorry, *{barcode}* is not a product barcode!")
    else:
        bot_message = "Sorry, we did not find any barcode."
    return bot_message


def get_barcodes_from_message(message):
    parsed_msg = parse_message(message)
    if len(parsed_msg["images"]) == 0:
        return None, None
    img = cv2.imread(parsed_msg["images"][-1]) # only use the last image
    # if max(img.shape[:2]) >= 640:
    #     img = image_resize(img, 640) # NOTE: zbar does not work pretty good for large resolution images
    #     print(img.shape)
    return get_barcodes(img)


def get_barcodes(frame):
    res = []
    for d in decode(frame):
        s = d.data.decode()
        print(s)
        frame = cv2.rectangle(frame, (d.rect.left, d.rect.top),
                                (d.rect.left + d.rect.width, d.rect.top + d.rect.height), (0, 255, 0), 3)
        frame = cv2.putText(frame, s, (d.rect.left, d.rect.top + d.rect.height),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        res.append(s)
    return res, frame


def get_barcodes_pil(img):
    res = []
    for d in decode(img):
        s = d.data.decode()
        # TODO: draw on pil image
        print(s)
        res.append(s)
    return res, img
