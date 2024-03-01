import numpy as np
import io
import json


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
