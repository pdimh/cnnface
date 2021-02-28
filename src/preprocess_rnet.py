import numpy as np
import os
import tensorflow as tf
import time

import model.inference as inference
import utils.config as config_utils
import utils.gpu as gpu
import preprocessing


from utils.face_class import FaceClass
from model import loss_box, loss_class
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
        if not pic:
            continue

        if len(pic.box):
            pic_stage1 = inference.stage1(pnet_model,
                                          pic,
                                          preconfig.stage1.pyramid_levels,
                                          preconfig.stage1.iou_threshold,
                                          preconfig.stage1.min_score,
                                          preconfig.stage1.min_face_size)

            pos = neg = part = 0
            for sbox in pic_stage1.box:
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
config = config_utils.get_config()
preconfig = config.preprocessing
gpu.configure(preconfig.force_cpu, config.gpu_mem_limit)

OUTPUT_PATH = preconfig.output_path
pics = preprocessing.get_picture(preconfig)

pnet_model = tf.keras.models.load_model(
    os.path.join(config.model_path, ModelType.PNET),
    custom_objects={'loss_class': loss_class, 'loss_box': loss_box})

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
