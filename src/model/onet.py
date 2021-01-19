import tensorflow as tf

from model import loss_box, loss_class


def model():
    inputs = tf.keras.layers.Input((48, 48, 3))

    conv1 = tf.keras.layers.Conv2D(
        32, (3, 3), strides=1)(inputs)
    prelu1 = tf.keras.layers.PReLU()(conv1)
    maxpool1 = tf.keras.layers.MaxPooling2D(
        (3, 3), strides=2, padding='same')(prelu1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=1)(maxpool1)
    prelu2 = tf.keras.layers.PReLU()(conv2)
    maxpool2 = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(prelu2)

    conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=1)(maxpool2)
    prelu3 = tf.keras.layers.PReLU()(conv3)
    maxpool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(prelu3)

    conv4 = tf.keras.layers.Conv2D(128, (2, 2), strides=1)(maxpool3)
    prelu4 = tf.keras.layers.PReLU()(conv4)

    flatten5 = tf.keras.layers.Flatten()(prelu4)
    dense5 = tf.keras.layers.Dense(256)(flatten5)
    prelu5 = tf.keras.layers.PReLU()(dense5)

    dense6_1 = tf.keras.layers.Dense(
        2, activation=tf.nn.softmax, name='class_output')(prelu5)

    dense6_2 = tf.keras.layers.Dense(
        4, name='box_output')(prelu5)

    model = tf.keras.models.Model(inputs=inputs, outputs=[dense6_1, dense6_2])
    model.compile(
        optimizer='adam',
        loss={
            'class_output': loss_class,
            'box_output': loss_box,
        },
        # run_eagerly=True
    )

    return model
