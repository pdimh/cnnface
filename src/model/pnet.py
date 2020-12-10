import numpy as np
import tensorflow as tf


@tf.autograph.experimental.do_not_convert
def loss_box(yt, yp):

    idx = yt[:, 2] * yt[:, 3] != 0
    yt_n = tf.cast(yt, 'float32')
    yp_n = tf.squeeze(yp)
    res = (yp_n - yt_n) ** 2
    res = tf.reduce_sum(res, axis=1)
    return tf.where(idx, res, tf.zeros(yt.shape[0])) * 0.5


@tf.autograph.experimental.do_not_convert
def loss_class(yt, yp):

    yt_n = tf.squeeze(tf.cast(yt, 'float'))
    yp_n = tf.clip_by_value(tf.squeeze(
        yp), clip_value_min=0.0001, clip_value_max=0.9999)

    idx = yt_n < 2

    res = -(yt_n * tf.math.log(yp_n[:, 1]) +
            (1-yt_n) * tf.math.log(yp_n[:, 0]))
    return tf.where(idx, res, tf.zeros(yt.shape[0]))


def model():
    inputs = tf.keras.layers.Input((12, 12, 3))

    conv1 = tf.keras.layers.Conv2D(
        10, (3, 3), strides=1, input_shape=(12, 12, 3))(inputs)
    prelu1 = tf.keras.layers.PReLU()(conv1)
    maxpool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(prelu1)

    conv2 = tf.keras.layers.Conv2D(16, (3, 3), strides=1)(maxpool1)
    prelu2 = tf.keras.layers.PReLU()(conv2)

    conv3 = tf.keras.layers.Conv2D(32, (3, 3), strides=1)(prelu2)
    prelu3 = tf.keras.layers.PReLU()(conv3)

    conv4_1 = tf.keras.layers.Conv2D(
        2, (1, 1), strides=1, activation=tf.nn.softmax, name='class_output')(prelu3)
    conv4_2 = tf.keras.layers.Conv2D(
        4, (1, 1), strides=1, name='box_output')(prelu3)

    model = tf.keras.models.Model(inputs=inputs, outputs=[conv4_1, conv4_2])
    model.compile(
        optimizer='adam',
        loss={
            'class_output': loss_class,
            'box_output': loss_box,
        },
        metrics=['accuracy'],
        run_eagerly=True
    )

    return model
