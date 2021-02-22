import numpy as np
import os
import sys
import tensorflow as tf

from utils.face_class import FaceClass
from preprocessing.fddb import FddbPics
from utils.pic_adapter import PicAdapter
from preprocessing.picture import Picture
from preprocessing.widerface import WFPics


def save_img(pic, output_path, model_type, sample_type, face_class, filename):
    path = os.path.join(output_path, model_type, sample_type,
                        str(face_class.name.lower()))
    os.makedirs(path, exist_ok=True)
    i = 0
    tf.keras.preprocessing.image.save_img(
        os.path.join(path, f'{filename}.jpg'), pic.data
    )

    if face_class is not FaceClass.NEGATIVE:
        np.save(os.path.join(
            path, f'{filename}_box'), pic.box, allow_pickle=False)


def get_picture(preConfig):
    pic_adapter = preConfig.adapter
    if pic_adapter == PicAdapter.WIDERFACE:
        wfpics = WFPics(preConfig.widerface.annotation,
                        preConfig.widerface.training,
                        preConfig.widerface.validation)
        return wfpics.pics
    elif pic_adapter == PicAdapter.FDDB:
        fddb = FddbPics(preConfig.fddb.annotation,
                        preConfig.fddb.binary)
        return fddb.pics
    else:
        sys.exit('PicAdapter is not valid.')


def get_pyramid(data, levels=10, min_size=12):

    factor = np.power(min_size / min(data.shape[0:2]), 1/levels)
    n_sizes = data.shape[0:2] * \
        np.transpose([np.power(factor, range(0, levels))])
    n_sizes = np.floor(n_sizes).astype(int)

    pyramid = []
    for i in range(0, levels):
        img_r = tf.cast(tf.image.resize(data, n_sizes[i]), tf.uint8).numpy()
        pyramid.append((img_r, data.shape[0] / img_r.shape[0]))

    return pyramid


def slide(net, data, window_size=12, threshold=0.5):

    shape = tf.constant((data.shape[1], data.shape[0]))
    prediction = net(tf.expand_dims(
        data / tf.constant(255, dtype=tf.uint8), axis=0))
    fclass = tf.reshape(
        prediction[0], [tf.shape(prediction[0])[1] * tf.shape(prediction[0])[2], tf.shape(prediction[0])[-1]])
    fbox = tf.reshape(
        prediction[1], [tf.shape(prediction[1])[1] * tf.shape(prediction[1])[2], tf.shape(prediction[1])[-1]])

    positive = tf.squeeze(
        tf.cast(tf.where(fclass[:, 1] > threshold), dtype=tf.int32), axis=[1])

    stride = 2
    ix = positive * stride
    bbox = tf.gather(tf.cast(fbox, dtype=tf.int32), positive)
    bbox_list = tf.transpose(tf.convert_to_tensor([(ix % (shape[0] - window_size + 1)
                                                    ) + bbox[:, 0],
                                                   (ix // (shape[0] - window_size + 1)) *
                                                   stride + bbox[:, 1],
                                                   bbox[:, 2],
                                                   bbox[:, 3]]))
    score_list = tf.gather(fclass, positive)[:, FaceClass.POSITIVE.value]
    return Picture(bbox_list.numpy(), data, score=np.array(score_list.numpy()))
