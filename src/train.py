from PIL import Image
import numpy as np
import os
from random import shuffle
import tensorflow as tf

from preproc.picture import Picture
import model.pnet as pnet
import utils.config as config_utils
from utils.face_class import FaceClass
from utils.sample_type import SampleType


def list(sample_type, face_class):
    res = []

    folder = os.path.join(os.path.relpath(
        PATH), sample_type, str(face_class.name.lower()))

    files = sorted([file for file in os.listdir(
        folder) if file.endswith('.jpg')])

    for file in files:
        path = os.path.join(folder, file)
        data = np.array(Image.open(path))
        box = None
        if len(data.shape) == 3 and face_class != FaceClass.NEGATIVE:
            box = np.load(path[0:-4] + '_box.npy')
        res.append(Picture(box, data, face_class))
    return res


config = config_utils.get_training()

if int(config['FORCE_CPU']):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    if config['GPU_MEM_LIMIT']:
        tf.config.experimental.set_virtual_device_configuration(
            tf.config.experimental.list_physical_devices('GPU')[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(config['GPU_MEM_LIMIT']))])

PATH = config['PATH']

train_positive = list(SampleType.TRAIN, FaceClass.POSITIVE)
train_part = list(SampleType.TRAIN, FaceClass.PART_FACE)
train_negative = list(SampleType.TRAIN, FaceClass.NEGATIVE)

trainpics = train_positive + train_part + train_negative

train_data = np.array([pic.data for pic in trainpics]) / 255
train_class = np.array([pic.face.value for pic in trainpics], dtype=int)
train_box = np.array([pic.box[0].flatten() if len(pic.box) != 0 else [
    0, 0, 0, 0] for pic in trainpics])

model = pnet.model()
model.summary()
model.fit(train_data,
          {
              "class_output": train_class,
              "box_output": train_box
          },
          batch_size=200,
          epochs=1000)
