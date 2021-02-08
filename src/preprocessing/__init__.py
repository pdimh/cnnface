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


def get_pyramid(data, factor=0.7, min_size=12):
    pyramid = []

    c_data = data
    c_shape_o = np.array([data.shape[0], data.shape[1]])
    c_shape = c_shape_o
    while all(np.array(c_shape * factor) >= (min_size, min_size)):
        new_data = np.array(tf.cast(tf.image.resize(
            c_data, c_shape * factor), tf.uint8))

        c_data = new_data
        c_shape = np.array([new_data.shape[0], new_data.shape[1]])

        pyramid.append((new_data, c_shape_o[0] / c_shape[0]))

    return pyramid


def slide(net, data, batch_size, window_size=12, stride=1, threshold=0.5):

    shape = tf.constant((data.shape[1], data.shape[0]))
    patches = tf.squeeze(tf.image.extract_patches(images=tf.convert_to_tensor([data], dtype=tf.uint8),
                                                  sizes=[1, window_size,
                                                         window_size, 1],
                                                  strides=[
                                                      1, stride, stride, 1],
                                                  rates=[1, 1, 1, 1],
                                                  padding='VALID'), 0)
    patches = tf.reshape(patches, shape=(-1, window_size, window_size, 3))
    prediction = net.predict(
        patches / tf.constant(255, dtype=tf.uint8), batch_size=batch_size)
    fclass = tf.reshape(
        prediction[0], [tf.shape(prediction[0])[0], tf.shape(prediction[0])[-1]])

    positive = tf.squeeze(
        tf.cast(tf.where(fclass[:, 1] > threshold), dtype=tf.int32))

    ix = positive * stride
    bbox = tf.gather(tf.cast(tf.reshape(
        prediction[1], [tf.shape(prediction[1])[0], tf.shape(prediction[1])[-1]]),
        dtype=tf.int32), positive)
    bbox_list = tf.transpose(tf.convert_to_tensor([(ix % (shape[0] - window_size + 1)
                                                    ) + bbox[:, 0],
                                                   (ix // (shape[0] - window_size + 1)) *
                                                   stride + bbox[:, 1],
                                                   bbox[:, 2],
                                                   bbox[:, 3]]))
    score_list = tf.gather(fclass, positive)[:, FaceClass.POSITIVE.value]
    return Picture(bbox_list.numpy(), data, score=np.array(score_list.numpy()))
