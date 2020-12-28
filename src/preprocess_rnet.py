import numpy as np
import os
import tensorflow as tf
import time

import utils.config as config_utils
import utils.gpu as gpu
import model.pnet as pnet
import preprocessing


from utils.face_class import FaceClass
from utils.model_type import ModelType
from preprocessing.picture import Picture
from utils.sample_type import SampleType


def extract_samples(output_path, pics, sample_type):

    if not len(pics):
        return

    counter = 0

    tf.print(
        f'=> Starting Extraction ({ModelType.RNET}): {sample_type} samples'
    )

    progbar = tf.keras.utils.Progbar(
        len(pics), width=50, verbose=1, interval=0.05, stateful_metrics=None,
        unit_name='step'
    )
    progbar.update(0)

    for t_pic in pics:
        pic = t_pic.get_as_picture()
        # pic.draw()
        if not pic:
            continue

        if len(pic.box):
            pyramid = preprocessing.get_pyramid(
                pic.data, factor=float(config['PYRAMID_FACTOR']))

            bbox = np.array([], dtype=int).reshape(0, 4)
            score = np.array([], dtype='float32')
            for pyr_item in pyramid:
                pic_ex = preprocessing.slide(pnet_model, Picture(
                    None, pyr_item[0]), window_size=12, stride=int(config['STRIDE']))
                if(pic_ex.box.shape[0] > 0):
                    bbox = np.concatenate(
                        (bbox, np.around(pic_ex.box * pyr_item[1]).astype(int)))
                    score = np.concatenate((score, pic_ex.score))
            bbox_nms = np.column_stack(
                [bbox[:, 0], bbox[:, 1], bbox[:, 0]+bbox[:, 2], bbox[:, 1]+bbox[:, 3]])
            idx_nms = tf.image.non_max_suppression(
                bbox_nms, score, len(bbox), 0.5, float(config['MIN_SCORE']))
            sboxes = tf.gather(bbox, idx_nms)
            # Picture(bbox, pic.data).draw()
            # Picture(sboxes, pic.data).draw()

            pos = neg = part = 0
            sboxes = tf.random.shuffle(sboxes)
            for sbox in sboxes:

                if sbox[2]*sbox[3] <= 0:
                    continue

                [spic, iou] = pic.extract(sbox, (24, 24))

                if iou < 0.3 and neg < 3 * (pos + 1):
                    neg += 1
                    preprocessing.save_img(
                        spic, output_path, ModelType.RNET, sample_type, FaceClass.NEGATIVE, f'{counter:04}')
                elif iou >= 0.4 and iou <= 0.65 and part < pos + 1:
                    part += 1
                    preprocessing.save_img(
                        spic, output_path, ModelType.RNET, sample_type, FaceClass.PART_FACE, f'{counter:04}')
                elif iou > 0.65:
                    pos += 1
                    preprocessing.save_img(
                        spic, output_path, ModelType.RNET, sample_type, FaceClass.POSITIVE, f'{counter:04}')
                else:
                    continue
                counter += 1
        progbar.add(1)


start_time = time.time()
config = config_utils.get_preprocessing(ModelType.RNET)
gpu.configure(config)

OUTPUT_PATH = os.path.relpath(config['OUTPUT_PATH'])
pics = preprocessing.get_picture(config)

pnet_model = tf.keras.models.load_model(
    os.path.join(config['MODEL_PATH'], 'pnet'),
    custom_objects={'loss_class': pnet.loss_class, 'loss_box': pnet.loss_box})

np.random.shuffle(pics)
train_len = len(pics) * int(config['TRAIN_PERCENT']) // 100
val_len = len(pics) * int(config['VAL_PERCENT']) // 100
train_pics = pics[0:train_len]
val_pics = pics[train_len:train_len + val_len]
test_pics = pics[train_len + val_len:]

extract_samples(OUTPUT_PATH, train_pics, SampleType.TRAIN)
extract_samples(OUTPUT_PATH, val_pics, SampleType.VALIDATION)
extract_samples(OUTPUT_PATH, test_pics, SampleType.TEST)

print(f'\nElapsed Time: {(time.time() - start_time):.2f} (s)')
