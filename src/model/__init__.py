import tensorflow as tf


# @tf.autograph.experimental.do_not_convert
def loss_box(yt, yp):

    idx = yt[:, 2] * yt[:, 3] != 0
    yt_n = tf.cast(yt, 'float32')
    yp_n = tf.reshape(yp, [tf.shape(yp)[0], tf.shape(yp)[-1]])
    res = (yp_n - yt_n) ** 2
    res = tf.reduce_sum(res, axis=1)
    return tf.where(idx, res, tf.zeros(tf.shape(yt)[0])) * 0.5


# @tf.autograph.experimental.do_not_convert
def loss_class(yt, yp):

    yt_n = tf.squeeze(tf.cast(yt, 'float'), axis=1)
    yp_n = tf.clip_by_value(tf.reshape(
        yp, [tf.shape(yp)[0], tf.shape(yp)[-1]]), clip_value_min=0.0001, clip_value_max=0.9999)
    idx = yt_n < 2
    res = -(yt_n * tf.math.log(yp_n[:, 1]) +
            (1-yt_n) * tf.math.log(yp_n[:, 0]))
    return tf.where(idx, res, tf.zeros(tf.shape(yt)[0]))
