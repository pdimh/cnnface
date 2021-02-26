import numpy as np
import tensorflow as tf

from utils.face_class import FaceClass
from preprocessing.picture import Picture


def stage1(pnet_model, picture, pyr_levels, iou_threshold, min_score):

    pyramid = _get_pyramid(
        picture.data, levels=pyr_levels)

    bbox = np.array([], dtype=int).reshape(0, 4)
    score = np.array([], dtype='float16')
    for pyr_item in pyramid:
        pic_ex = _slide(
            pnet_model, pyr_item[0])
        if(pic_ex.box.shape[0] > 0):
            bbox = np.concatenate(
                (bbox, np.around(pic_ex.box * pyr_item[1]).astype(int)))
            score = np.concatenate((score, pic_ex.score))
    bbox_nms = np.column_stack(
        [bbox[:, 0], bbox[:, 1], bbox[:, 0]+bbox[:, 2], bbox[:, 1]+bbox[:, 3]])
    idx_nms = tf.image.non_max_suppression(
        bbox_nms, score, len(bbox), iou_threshold, min_score)
    sboxes = tf.gather(bbox, idx_nms)
    sboxes = tf.random.shuffle(sboxes).numpy()
    return Picture(sboxes, picture.data)


def stage2(rnet_model, picture, iou_threshold, min_score):

    w_size = 24
    patches = []

    if not len(picture.box):
        return picture

    for box in picture.box:
        patches.append(picture.extract(box, resize=(w_size, w_size))[0])

    r_data = np.true_divide(
        np.array([p.data for p in patches]), 255, dtype=np.float16)
    prediction = rnet_model(r_data, training=False)

    idx_face = np.where(np.array(prediction[0][:, 1]) > min_score)

    bbox = np.array([], dtype=int).reshape(0, 4)
    score = []
    for i in idx_face[0]:
        o_box = picture.box[i]
        c_box = np.array(prediction[1])[i]
        n_box = np.around([o_box[0] + c_box[0], o_box[1] +
                           c_box[1], o_box[2], o_box[3]]).astype(int)
        bbox = np.concatenate((bbox, n_box.reshape(1, 4)))
        score.append(np.array(prediction[0][:, 1])[i])

    bbox_nms = np.column_stack(
        [bbox[:, 0], bbox[:, 1], bbox[:, 0]+bbox[:, 2], bbox[:, 1]+bbox[:, 3]])
    idx_nms = tf.image.non_max_suppression(
        bbox_nms, score, len(bbox), iou_threshold, min_score)
    sboxes = tf.gather(bbox, idx_nms).numpy()
    return Picture(sboxes, picture.data)


def stage3(onet_model, picture, iou_threshold, min_score):

    w_size = 48
    patches = []

    if not len(picture.box):
        return picture

    for box in picture.box:
        patches.append(picture.extract(box, resize=(w_size, w_size))[0])

    r_data = np.true_divide(
        np.array([p.data for p in patches]), 255, dtype=np.float16)
    prediction = onet_model(r_data, training=False)

    idx_face = np.where(np.array(prediction[0][:, 1]) > min_score)

    bbox = np.array([], dtype=int).reshape(0, 4)
    score = []
    for i in idx_face[0]:
        o_box = picture.box[i]
        c_box = np.array(prediction[1])[i]
        n_box = np.around([o_box[0] + c_box[0], o_box[1] +
                           c_box[1], o_box[2], o_box[3]]).astype(int)
        bbox = np.concatenate((bbox, n_box.reshape(1, 4)))
        score.append(np.array(prediction[0][:, 1])[i])

    bbox_nms = np.column_stack(
        [bbox[:, 0], bbox[:, 1], bbox[:, 0]+bbox[:, 2], bbox[:, 1]+bbox[:, 3]])
    idx_nms = tf.image.non_max_suppression(
        bbox_nms, score, len(bbox), iou_threshold, min_score)
    sboxes = tf.gather(bbox, idx_nms).numpy()
    return Picture(sboxes, picture.data)


def _slide(net, data, threshold=0.5):

    prediction = net(tf.expand_dims(
        data / tf.constant(255, dtype=tf.uint8), axis=0))
    fclass = tf.squeeze(prediction[0], axis=[0])
    fbox = tf.squeeze(prediction[1], axis=[0])

    positive = tf.where(fclass[:, :, 1] > threshold)

    ix = positive * 2  # In pnet, stride is 2
    bbox = tf.gather_nd(tf.cast(fbox, dtype=tf.int64), positive)
    bbox_list = tf.transpose(tf.convert_to_tensor([ix[:, 1] + bbox[:, 0],
                                                   ix[:, 0] + bbox[:, 1],
                                                   bbox[:, 2],
                                                   bbox[:, 3]]))
    score_list = tf.gather_nd(fclass, positive)[:, FaceClass.POSITIVE.value]
    return Picture(bbox_list.numpy(), data, score=np.array(score_list.numpy()))


def _get_pyramid(data, levels=10, min_size=12):

    factor = np.power(min_size / min(data.shape[0:2]), 1/levels)
    n_sizes = data.shape[0:2] * \
        np.transpose([np.power(factor, range(0, levels))])
    n_sizes = np.floor(n_sizes).astype(int)

    pyramid = []
    for i in range(0, levels):
        img_r = tf.cast(tf.image.resize(data, n_sizes[i]), tf.uint8).numpy()
        pyramid.append((img_r, data.shape[0] / img_r.shape[0]))

    return pyramid
