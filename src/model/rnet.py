import tensorflow as tf

from model import loss_box, loss_class


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
