import os

from matplotlib.patches import Ellipse, Rectangle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from preproc.picture import Picture

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FddbPic:
    def __init__(self, path, qt, coord):
        self.path = path
        self.qt = qt
        self.coord = coord
        self.coord_rect = np.column_stack([
            coord[:, 3]-coord[:, 1], coord[:, 4]-coord[:, 0], 2*coord[:, 1], 2*coord[:, 0]])
        self._data = None
        self._data_res = {}

    @property
    def data(self):
        if self._data is None:
            self._data = np.array(Image.open(self.path))
        return self._data


class FddbPics:

    def __init__(self, annot_folder, bin_folder):
        self.annot_folder = annot_folder
        self.bin_folder = bin_folder
        self._process()

    def _process(self):
        pics = []

        files = sorted([os.path.join(self.annot_folder, file) for file in os.listdir(
            self.annot_folder) if 'ellipseList' in file])
        for file in files:
            with open(file, 'r') as reader:
                lines = [line.rstrip() for line in reader.readlines()]

                i = 0
                while i < len(lines):
                    path = os.path.join(self.bin_folder, lines[i] + '.jpg')
                    qt = int(lines[i+1])
                    coord = np.array([np.fromstring(c, dtype=np.float, sep=' ')
                                      for c in lines[i+2:i+2+qt]])
                    coord = coord[:, 0:-1]
                    pics.append(FddbPic(path, qt, coord))
                    i += 2+qt
        self.pics = pics

    def get(self, qt=None):
        if qt is None:
            return self.pics
        else:
            return np.array(list(filter(lambda pic: pic.qt == qt, self.pics)))

    def get_as_picture(self):
        return [Picture(pic.coord_rect, pic.data) for pic in self.get() if len(pic.data.shape) == 3]
