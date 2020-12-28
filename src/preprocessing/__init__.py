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


def get_picture(config):
    pic_adapter = PicAdapter.WIDERFACE if config['ADAPTER'] == 'WIDERFACE' else PicAdapter.FDDB
    if pic_adapter == PicAdapter.WIDERFACE:
        wfpics = WFPics(os.path.relpath(config['WIDER_ANNOT']),
                        os.path.relpath(config['WIDER_TRAIN']),
                        os.path.relpath(config['WIDER_VALIDATION']))
        return wfpics.pics
    elif pic_adapter == PicAdapter.FDDB:
        fddb = FddbPics(os.path.relpath(config['FDDB_ANNOT']),
                        os.path.relpath(config['FDDB_BIN']))
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


def slide(net, picture, window_size=12, stride=1, threshold=0.5):

    data = picture.data
    shape = (data.shape[1], data.shape[0])

    patches = []

    xp = yp = 0
    while yp <= shape[1] - window_size:
        while xp <= shape[0] - window_size:
            ndata = data[yp:yp + window_size, xp:xp + window_size, :]
            xp += stride
            patches.append(ndata)
        xp = 0
        yp += stride
    prediction = net(np.array(patches) / 255, training=False)
    fclass = np.squeeze(prediction[0], axis=(1, 2))
    bbox = np.squeeze(prediction[1], axis=(1, 2))

    bbox_list = []
    score_list = []

    for i in range(len(fclass)):
        if(fclass[i][FaceClass.POSITIVE.value] > threshold):
            ix = i * stride
            bbox_list.append(
                np.around([
                    (ix % (shape[0] - window_size + 1)) + bbox[i][0],
                    (ix // (shape[0] - window_size + 1)) * stride + bbox[i][1],
                    bbox[i][2],
                    bbox[i][3]]).astype(int))
            score_list.append(fclass[i][FaceClass.POSITIVE.value])
    return Picture(np.array(bbox_list), data, score=np.array(score_list))
