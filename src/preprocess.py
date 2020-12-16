import numpy as np
import os
import sys
import tensorflow as tf

import utils.config as config_utils

from utils.face_class import FaceClass
from preproc.fddb import FddbPics
from utils.pic_adapter import PicAdapter
from utils.sample_type import SampleType
from preproc.widerface import WFPics


def save_img(pic, sample_type, face_class, filename):
    path = os.path.join(OUTPUT_PATH, sample_type, str(face_class.name.lower()))
    os.makedirs(path, exist_ok=True)
    i = 0
    tf.keras.preprocessing.image.save_img(
        os.path.join(path, f'{filename}.jpg'), pic.data
    )

    if face_class is not FaceClass.NEGATIVE:
        np.save(os.path.join(
            path, f'{filename}_box'), pic.box, allow_pickle=False)


def extract_samples(pics, sample_type):
    counter = 0

    tf.print(f"=> Starting Extraction: {sample_type} samples")

    progbar = tf.keras.utils.Progbar(
        len(pics), width=50, verbose=1, interval=0.05, stateful_metrics=None,
        unit_name='step'
    )
    progbar.update(0)

    for t_pic in pics:
        pic = t_pic.get_as_picture()
        count = [30, 10, 10]
        pic.filter_boxes(12)
        if len(pic.box):
            while count[0] > 0:
                (crop, iou) = pic.extract_rnd((12, 12),
                                              min_cropfactor=0.1, max_cropfactor=2)
                if iou < 0.3:
                    if(count[0] > 0):
                        count[0] -= 1
                        save_img(crop, sample_type,
                                 FaceClass.NEGATIVE, f'{counter:04}')
                elif iou >= 0.4 and iou <= 0.65:
                    if(count[1] > 0):
                        count[1] -= 1
                        save_img(crop, sample_type,
                                 FaceClass.PART_FACE, f'{counter:04}')
                elif iou > 0.65:
                    if(count[2] > 0):
                        count[2] -= 1
                        save_img(crop, sample_type,
                                 FaceClass.POSITIVE, f'{counter:04}')
                else:
                    continue
                counter += 1

            cropfaces = pic.crop_rng(min_cropfactor=1.1, max_cropfactor=1.2)
            err = 0
            while any(j > 0 for j in count):
                rndpic = cropfaces[np.random.randint(0, len(cropfaces))]
                (crop, iou) = rndpic.extract_rnd(
                    (12, 12), min_cropfactor=0.7)

                if iou >= 0.4 and iou <= 0.65:
                    if(count[1] > 0):
                        count[1] -= 1
                        save_img(crop, sample_type,
                                 FaceClass.PART_FACE, f'{counter:04}')
                elif iou > 0.65:
                    if(count[2] > 0):
                        count[2] -= 1
                        save_img(crop, sample_type,
                                 FaceClass.POSITIVE, f'{counter:04}')
                else:
                    if err < 20:
                        err += 1
                    else:
                        break
                    continue
                counter += 1
        progbar.add(1)


def get_picture(pic_adapter):
    if pic_adapter == PicAdapter.WIDERFACE:
        return _get_picture_widerface()
    elif pic_adapter == PicAdapter.FDDB:
        return _get_picture_fddb()
    else:
        sys.exit('PicAdapter is not valid.')


def _get_picture_fddb():
    fddb = FddbPics(os.path.relpath(config['FDDB_ANNOT']),
                    os.path.relpath(config['FDDB_BIN']))
    return fddb.pics


def _get_picture_widerface():
    wfpics = WFPics(os.path.relpath(config['WIDER_ANNOT']),
                    os.path.relpath(config['WIDER_TRAIN']),
                    os.path.relpath(config['WIDER_VALIDATION']))
    return wfpics.pics


config = config_utils.get_preprocess()

if int(config['FORCE_CPU']):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

OUTPUT_PATH = os.path.relpath(config['OUTPUT_PATH'])
pics = get_picture(PicAdapter.WIDERFACE)

np.random.shuffle(pics)
train_len = len(pics)*int(config['FDDB_TRAIN_PERCENT'])//100
val_len = len(pics)*int(config['FDDB_VAL_PERCENT'])//100
train_pics = pics[0:train_len]
val_pics = pics[train_len:train_len + val_len]
test_pics = pics[train_len + val_len:]

extract_samples(train_pics, SampleType.TRAIN)
extract_samples(val_pics, SampleType.VALIDATION)
extract_samples(test_pics, SampleType.TEST)
