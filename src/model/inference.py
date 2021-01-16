import numpy as np
import tensorflow as tf

import preprocessing

from preprocessing.picture import Picture


def stage1(pnet_model, picture, pyr_factor, stride, iou_threshold, min_score):

    pyramid = preprocessing.get_pyramid(
        picture.data, factor=pyr_factor)

    bbox = np.array([], dtype=int).reshape(0, 4)
    score = np.array([], dtype='float32')
    for pyr_item in pyramid:
        pic_ex = preprocessing.slide(pnet_model, Picture(
            None, pyr_item[0]), window_size=12, stride=stride)
        if(pic_ex.box.shape[0] > 0):
            bbox = np.concatenate(
                (bbox, np.around(pic_ex.box * pyr_item[1]).astype(int)))
            score = np.concatenate((score, pic_ex.score))
    bbox_nms = np.column_stack(
        [bbox[:, 0], bbox[:, 1], bbox[:, 0]+bbox[:, 2], bbox[:, 1]+bbox[:, 3]])
    idx_nms = tf.image.non_max_suppression(
        bbox_nms, score, len(bbox), iou_threshold, min_score)
    sboxes = tf.gather(bbox, idx_nms)

    sboxes = tf.random.shuffle(sboxes)

    return Picture(sboxes, picture.data)


def stage2(rnet_model, picture, iou_threshold, min_score):

    w_size = 24
    patches = []

    if not len(picture.box):
        return picture

    for box in picture.box:
        patches.append(picture.extract(box, resize=(w_size, w_size))[0])

    r_data = np.array([p.data for p in patches]) / 255
    prediction = rnet_model(r_data, training=False)

    idx_face = np.where(np.array(prediction[0][:, 1]) > 0.5)

    bbox = np.array([], dtype=int).reshape(0, 4)
    score = []
    for i in idx_face[0]:
        o_box = picture.box[i]
        c_box = np.array(prediction[1])[i]
        n_box = np.around([o_box[0] + c_box[0], o_box[1] + c_box[1], c_box[2]
                           * o_box[2] / w_size, c_box[3] * o_box[3] / w_size]).astype(int)
        bbox = np.concatenate((bbox, n_box.reshape(1, 4)))
        score.append(np.array(prediction[0][:, 1])[i])

    bbox_nms = np.column_stack(
        [bbox[:, 0], bbox[:, 1], bbox[:, 0]+bbox[:, 2], bbox[:, 1]+bbox[:, 3]])
    idx_nms = tf.image.non_max_suppression(
        bbox_nms, score, len(bbox), iou_threshold, min_score)
    sboxes = tf.gather(bbox, idx_nms)
    return Picture(sboxes, picture.data)
