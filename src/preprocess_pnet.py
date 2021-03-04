import numpy as np
import tensorflow as tf
import time

import utils.config as config_utils
import utils.gpu as gpu
import preprocessing

from utils.face_class import FaceClass
from utils.model_type import ModelType
from utils.sample_type import SampleType


def extract_samples(output_path, pics, sample_type):

    if not len(pics):
        return

    counter = 0

    tf.print(
        f'=> Starting Extraction ({ModelType.PNET}): {sample_type} samples'
    )

    progbar = tf.keras.utils.Progbar(
        len(pics), width=50, verbose=1, interval=0.05, stateful_metrics=None,
        unit_name='step'
    )
    progbar.update(0)

    for t_pic in pics:
        pic = t_pic.get_as_picture()
        if not pic:
            continue

        count = [30, 10, 10]
        pic.filter_boxes(12)
        if len(pic.box):
            while count[0] > 0:
                (crop, iou) = pic.extract_rnd((12, 12),
                                              min_cropfactor=0.1, max_cropfactor=2)
                if iou < 0.3:
                    if(count[0] > 0):
                        count[0] -= 1
                        preprocessing.save_img(crop, output_path, ModelType.PNET, sample_type,
                                               FaceClass.NEGATIVE, f'{counter:04}')
                elif iou >= 0.4 and iou <= 0.65:
                    if(count[1] > 0):
                        count[1] -= 1
                        preprocessing.save_img(crop, output_path, ModelType.PNET, sample_type,
                                               FaceClass.PART_FACE, f'{counter:04}')
                elif iou > 0.65:
                    if(count[2] > 0):
                        count[2] -= 1
                        preprocessing.save_img(crop, output_path, ModelType.PNET, sample_type,
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
                        preprocessing.save_img(crop, output_path, ModelType.PNET, sample_type,
                                               FaceClass.PART_FACE, f'{counter:04}')
                elif iou > 0.65:
                    if(count[2] > 0):
                        count[2] -= 1
                        preprocessing.save_img(crop, output_path, ModelType.PNET, sample_type,
                                               FaceClass.POSITIVE, f'{counter:04}')
                else:
                    if err < 20:
                        err += 1
                    else:
                        break
                    continue
                counter += 1
        progbar.add(1)


start_time = time.time()
config = config_utils.get_config()
preconfig = config.preprocessing
gpu.configure(preconfig.force_cpu, config.gpu_mem_limit)

OUTPUT_PATH = config.sample_path
pics = preprocessing.get_picture(preconfig)

np.random.shuffle(pics)
train_len = len(pics) * preconfig.percentage.training // 100
val_len = len(pics) * preconfig.percentage.validation // 100
train_pics = pics[0:train_len]
val_pics = pics[train_len:train_len + val_len]
test_pics = pics[train_len + val_len:]

extract_samples(OUTPUT_PATH, train_pics, SampleType.TRAIN)
extract_samples(OUTPUT_PATH, val_pics, SampleType.VALIDATION)
extract_samples(OUTPUT_PATH, test_pics, SampleType.TEST)

print(f'\nElapsed Time: {(time.time() - start_time):.2f} (s)')
