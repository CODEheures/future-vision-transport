import tensorflow as tf
import serializer
from data import Data
import numpy as np
from losses import DiceLoss, JaccardLoss  # noqa: F401
from flask import Flask, request

model_h5 = 'model.h5'
model = tf.keras.models.load_model(model_h5)
app = Flask(__name__)


def infer(img):
    predict = model.predict(np.array([img], dtype=np.uint8))
    return predict[0]


@app.route("/", methods=['POST'])
def predict():
    json = request.json
    img = serializer.deserialize(json['image'])
    infered = infer(img)
    dt = Data(make_df=False)
    mask_pred = dt.y_pred_to_mask(infered)
    serialized_mask = serializer.serialize(mask_pred)
    response = {'predicted_mask': serialized_mask}
    return response
