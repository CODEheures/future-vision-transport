import numpy as np
import io
import json
import base64
from PIL import Image


def toBase64(img):
    with open(img, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def serialize(img):
    memfile = io.BytesIO()
    np.save(memfile, img)
    serialized = memfile.getvalue()
    return json.dumps(serialized.decode('latin-1'))


def deserialize(json_data):
    memfile = io.BytesIO()
    memfile.write(json.loads(json_data).encode('latin-1'))
    memfile.seek(0)
    return np.load(memfile)


def numpy_tobase64(img):
    path = 'mask_predict.png'
    image = Image.fromarray(img, mode="RGB")
    image.save(path)
    return toBase64(path)
