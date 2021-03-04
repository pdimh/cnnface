import argparse
import os

from PIL import Image
import numpy as np

import tensorflow as tf

from preprocessing.picture import Picture

from model import loss_box, loss_class

import model.inference as inference
import utils.gpu as gpu
import utils.config as config_utils
from utils.model_type import ModelType


parser = argparse.ArgumentParser(
    description='Detect faces using trained models')
parser.add_argument('path',
                    help='path of the picture')
args = parser.parse_args()

try:
    data = np.array(Image.open(args.path))
except IOError:
    print('Error: File not found.')
    exit(1)

config = config_utils.get_config()
infConfig = config.inference
gpu.configure(infConfig.force_cpu, config.gpu_mem_limit)


pnet_model = tf.keras.models.load_model(
    os.path.join(config.model_path, ModelType.PNET),
    custom_objects={'loss_class': loss_class, 'loss_box': loss_box})
rnet_model = tf.keras.models.load_model(
    os.path.join(config.model_path, ModelType.RNET),
    custom_objects={'loss_class': loss_class, 'loss_box': loss_box})
onet_model = tf.keras.models.load_model(
    os.path.join(config.model_path, ModelType.ONET),
    custom_objects={'loss_class': loss_class, 'loss_box': loss_box})

pic_stage1 = inference.stage1(pnet_model,
                              Picture(None, data),
                              infConfig.stage1.pyramid_levels,
                              infConfig.stage1.iou_threshold,
                              infConfig.stage1.min_score,
                              infConfig.stage1.min_face_size)
pic_stage2 = inference.stage2(rnet_model,
                              pic_stage1,
                              infConfig.stage2.iou_threshold,
                              infConfig.stage2.min_score)
pic_stage3 = inference.stage3(onet_model,
                              pic_stage2,
                              infConfig.stage3.iou_threshold,
                              infConfig.stage3.min_score)
pic_stage3.draw()
