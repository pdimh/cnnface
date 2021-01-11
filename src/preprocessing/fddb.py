from PIL import Image

import numpy as np
import os

from preprocessing.picture import Picture


class FddbPic:
    def __init__(self, path, qt, coord):
        self.path = path
        self.qt = qt
        self.coord = coord
        self.coord_rect = np.column_stack([
            coord[:, 3]-coord[:, 1], coord[:, 4]-coord[:, 0], 2*coord[:, 1], 2*coord[:, 0]])

    def get_data(self):
        return np.array(Image.open(self.path))

    def get_as_picture(self):
        try:
            return Picture(self.coord_rect, self.get_data())
        except:
            return None


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
