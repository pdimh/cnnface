import tensorflow as tf

from model import loss_box, loss_class


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
        # run_eagerly=True
    )

    return model
