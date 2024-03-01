import tensorflow as tf
import serializer
from data import Data
import numpy as np
import keras


@keras.saving.register_keras_serializable(package="Loss")
class DiceLoss(tf.keras.losses.Loss):

    def __init__(self, name='Sørensen–Dice-Loss', smooth=1e-6):
        super().__init__()
        self.name = name
        self.smooth = smooth

    def __str__(self):
        return self.name

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        intersection = tf.reduce_sum( y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_pred**2, axis=(1, 2, 3)) + tf.reduce_sum(y_true**2, axis=(1, 2, 3))
        dice = 2 * intersection/(denominator + self.smooth)
        return 1 - dice

    def get_config(self):
        return {"name": self.name, "smooth": self.smooth}


@keras.saving.register_keras_serializable(package="Loss")
class JaccardLoss(tf.keras.losses.Loss):

    def __init__(self, name='Jaccard-Loss', smooth=1e-6):
        super().__init__()
        self.name = name
        self.smooth = smooth

    def __str__(self):
        return self.name

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        intersection = tf.reduce_sum( y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_pred**2, axis=(1, 2, 3)) + tf.reduce_sum(y_true**2, axis=(1, 2, 3))
        dice = 2 * intersection/(denominator + self.smooth)
        jaccard = dice / (2 - dice)
        return 1 - jaccard

    def get_config(self):
        return {"name": self.name, "smooth": self.smooth}


def infer(img):
    model_h5 = 'model.h5'
    model = tf.keras.models.load_model(model_h5)
    predict = model.predict(np.array([img], dtype=np.uint8))
    return predict[0]


def predict(json):
    img = serializer.deserialize(json)
    infered = infer(img)
    dt = Data(make_df=False)
    return dt.y_pred_to_mask(infered)
