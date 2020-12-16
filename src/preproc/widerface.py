from PIL import Image

import numpy as np
import os

from preproc.picture import Picture


class WFPic:
    def __init__(self, path,  coord):
        self.path = path
        self.coord = coord
        self._data = None
        self._data_res = {}

    @property
    def data(self):
        if self._data is None:
            self._data = np.array(Image.open(self.path))
        return self._data

    def get_as_picture(self):
        return Picture(self.coord, self.data)


class WFPics:

    def __init__(self, annot_folder, bin_train_folder, bin_val_folder):
        self.annot_folder = annot_folder
        self.bin_train_folder = bin_train_folder
        self.bin_val_folder = bin_val_folder
        self._process()

    def _process(self):
        pics = []

        files = sorted([os.path.join(self.annot_folder, file) for file in os.listdir(
            self.annot_folder) if 'ellipseList' in file])

        files = ((os.path.join(self.annot_folder, 'wider_face_train_bbx_gt.txt'), os.path.relpath(self.bin_train_folder)),
                 (os.path.join(self.annot_folder, 'wider_face_val_bbx_gt.txt'), os.path.relpath(self.bin_val_folder)))
        for file in files:
            with open(file[0], 'r') as reader:
                lines = [line.rstrip() for line in reader.readlines()]

                i = 0
                while i < len(lines):
                    path = os.path.join(file[1], lines[i])
                    qt = int(lines[i+1])
                    coord = np.array([np.fromstring(c, dtype=np.float, sep=' ')
                                      for c in lines[i+2:i+2+qt] if c[7] != '1'])
                    if len(coord) > 0:
                        coord = coord[:, 0:4]
                    pics.append(WFPic(path, coord))
                    i += 2+qt if qt > 0 else 3
        self.pics = pics
