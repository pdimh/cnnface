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


def get_picture(preConfig):
    pic_adapter = preConfig.adapter
    if pic_adapter == PicAdapter.WIDERFACE:
        wfpics = WFPics(preConfig.widerface.annotation,
                        preConfig.widerface.training,
                        preConfig.widerface.validation)
        return wfpics.pics
    elif pic_adapter == PicAdapter.FDDB:
        fddb = FddbPics(preConfig.fddb.annotation,
                        preConfig.fddb.binary)
        return fddb.pics
    else:
        sys.exit('PicAdapter is not valid.')
