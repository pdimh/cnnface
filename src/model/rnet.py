import tensorflow as tf


# @tf.autograph.experimental.do_not_convert
def loss_box(yt, yp):

    idx = yt[:, 2] * yt[:, 3] != 0
    yt_n = tf.cast(yt, 'float32')
    yp_n = yp
    res = (yp_n - yt_n) ** 2
    res = tf.reduce_sum(res, axis=1)
    return tf.where(idx, res, tf.zeros(tf.shape(yt)[0])) * 0.5


# @tf.autograph.experimental.do_not_convert
def loss_class(yt, yp):

    yt_n = tf.squeeze(tf.cast(yt, 'float'), axis=1)
    yp_n = tf.clip_by_value(yp, clip_value_min=0.0001, clip_value_max=0.9999)

    idx = yt_n < 2

    res = -(yt_n * tf.math.log(yp_n[:, 1]) +
            (1-yt_n) * tf.math.log(yp_n[:, 0]))
    return tf.where(idx, res, tf.zeros(tf.shape(yt)[0]))


def model():
    inputs = tf.keras.layers.Input((24, 24, 3))

    conv1 = tf.keras.layers.Conv2D(
        28, (3, 3), strides=1)(inputs)
    prelu1 = tf.keras.layers.PReLU()(conv1)
    maxpool1 = tf.keras.layers.MaxPooling2D(
        (3, 3), strides=2, padding='same')(prelu1)

    conv2 = tf.keras.layers.Conv2D(48, (3, 3), strides=1)(maxpool1)
    prelu2 = tf.keras.layers.PReLU()(conv2)
    maxpool2 = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(prelu2)

    conv3 = tf.keras.layers.Conv2D(64, (2, 2), strides=1)(maxpool2)
    prelu3 = tf.keras.layers.PReLU()(conv3)

    flatten4 = tf.keras.layers.Flatten()(prelu3)
    dense4 = tf.keras.layers.Dense(128)(flatten4)
    prelu4 = tf.keras.layers.PReLU()(dense4)

    dense5_1 = tf.keras.layers.Dense(
        2, activation=tf.nn.softmax, name='class_output')(prelu4)

    dense5_2 = tf.keras.layers.Dense(
        4, name='box_output')(prelu4)

    model = tf.keras.models.Model(inputs=inputs, outputs=[dense5_1, dense5_2])
    model.compile(
        optimizer='adam',
        loss={
            'class_output': loss_class,
            'box_output': loss_box,
        },
        # run_eagerly=True
    )

    return model
