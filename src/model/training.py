import numpy as np
import os
import time

from PIL import Image
from random import shuffle

import utils.config as config_utils
import utils.gpu as gpu
import model.onet as onet
import model.pnet as pnet
import model.rnet as rnet


from utils.face_class import FaceClass
from preprocessing.picture import Picture
from utils.model_type import ModelType
from utils.sample_type import SampleType


def list(path, model_type, sample_type, face_class):
    res = []

    folder = os.path.join(path, model_type, sample_type,
                          str(face_class.name.lower()))

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


def accuracy_class(model, test_data, test_class):
    predicts = model.predict(test_data)[0]
    predicts = np.reshape(predicts, [predicts.shape[0], predicts.shape[-1]])

    err = 0
    total = len(test_data)
    for idx in range(0, len(test_data)):
        pred = predicts[idx]
        truth = test_class[idx]

        if truth == FaceClass.NEGATIVE.value and pred[0] < pred[1]:
            err += 1
        elif truth == FaceClass.POSITIVE.value and pred[0] > pred[1]:
            err += 1
        elif truth == FaceClass.PART_FACE.value:
            total -= 1

    print(f'Accuracy: {1 - err / total}')


def fit(model_type):
    start_time = time.time()
    config = config_utils.get_config()
    tconfig = config.training
    model_config = getattr(tconfig, model_type)
    gpu.configure(model_config.force_cpu, config.gpu_mem_limit)

    PATH = os.path.relpath(tconfig.path)

    train_positive = list(PATH, model_type,
                          SampleType.TRAIN, FaceClass.POSITIVE)
    train_part = list(PATH, model_type,
                      SampleType.TRAIN, FaceClass.PART_FACE)
    train_negative = list(PATH, model_type,
                          SampleType.TRAIN, FaceClass.NEGATIVE)

    trainpics = train_positive + train_part + train_negative
    shuffle(trainpics)

    train_data = np.array([pic.data for pic in trainpics]) / 255
    train_class = np.array([pic.face.value for pic in trainpics], dtype=int)
    train_box = np.array([pic.box[0].flatten() if len(pic.box) != 0 else np.array([
        0, 0, 0, 0]) for pic in trainpics])

    model = None
    if model_type == ModelType.PNET:
        model = pnet.model()
    elif model_type == ModelType.RNET:
        model = rnet.model()
    elif model_type == ModelType.ONET:
        model = onet.model()

    model.summary()
    model.fit(train_data,
              {
                  "class_output": train_class,
                  "box_output": train_box
              },
              batch_size=model_config.batch_size,
              epochs=model_config.epochs)

    model.save(os.path.join(config.model_path, model_type))

    print(f'\nElapsed Time: {(time.time() - start_time):.2f} (s)')

    # Uncomment to check accuracy

    accuracy_class(model, train_data, train_class)

    test_positive = list(PATH, model_type, SampleType.TEST, FaceClass.POSITIVE)
    test_part = list(PATH, model_type, SampleType.TEST, FaceClass.PART_FACE)
    test_negative = list(PATH, model_type, SampleType.TEST, FaceClass.NEGATIVE)

    testpics = test_positive + test_part + test_negative

    test_data = np.array([pic.data for pic in testpics]) / 255
    test_class = np.array([pic.face.value for pic in testpics], dtype=int)

    accuracy_class(model, test_data, test_class)
