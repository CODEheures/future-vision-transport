import keras
import tensorflow as tf


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
