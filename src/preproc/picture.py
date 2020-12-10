import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import tensorflow as tf

from utils.face_class import FaceClass


class Picture:
    def __init__(self, box, data, face_class=FaceClass.POSITIVE):
        self.data = data

        if len(self.data.shape) < 3:
            raise ValueError('Data must be RGB')

        self.size = np.array([data.shape[1], data.shape[0]]).astype(int)
        self.box = box.astype(int) if box is not None and len(
            box) > 0 else np.array([])

        if(self.box.size > 0):
            self.box = self.box.clip(min=0)
            self.box[:, 2] = self.box[:, 2].clip(
                max=self.size[0]-self.box[:, 0])
            self.box[:, 3] = self.box[:, 3].clip(
                max=self.size[1]-self.box[:, 1])
            self.face = face_class
        else:
            self.face = FaceClass.NEGATIVE

    def draw(self):

        plt.imshow(self.data)
        ax = plt.gca()

        for c in self.box:
            ax.add_patch(Rectangle((c[0], c[1]), width=c[2], height=c[3]-1,
                                   fill=False, color='b'))
        plt.show()

    def crop_rng(self, min_cropfactor=0.7, max_cropfactor=1.2):
        cropbox = self.box.copy()
        crop_factor = np.random.randint(
            min_cropfactor * 100, max_cropfactor * 100 + 1) / 100
        delta = cropbox[:, 2:4] * (crop_factor - 1) // 2

        cropbox[:, 0] = cropbox[:, 0] - delta[:, 0]
        cropbox[:, 1] = cropbox[:, 1] - delta[:, 1]
        cropbox[:, 2] = cropbox[:, 2] + (delta[:, 0] * 2) + 1
        cropbox[:, 3] = cropbox[:, 3] + (delta[:, 1] * 2) + 1
        cropbox = cropbox.clip(min=0)
        cropbox[:, 2] = cropbox[:, 2].clip(max=self.size[0]-cropbox[:, 0])
        cropbox[:, 3] = cropbox[:, 3].clip(max=self.size[1]-cropbox[:, 1])

        new_data = [tf.image.crop_to_bounding_box(
            self.data, crop[1], crop[0], crop[3], crop[2]) for crop in cropbox]
        newbox = np.column_stack((
            self.box[:, 0] - cropbox[:, 0],
            self.box[:, 1] - cropbox[:, 1],
            self.box[:, 2],
            self.box[:, 3]
        ))

        return [Picture(np.array([newbox[i]]), new_data[i]) for i in range(0, len(self.box))]

    def filter_boxes(self, min_size):
        self.box = np.delete(self.box, np.array(
            np.where(self.box[:, 2:4] <= min_size))[0, :], 0)

    # Extract all patches from picture
    def extract_patches(self, size, strides):

        patches = []

        xp = np.arange(0, self.size[0], strides[0])
        yp = np.arange(0, self.size[1], strides[1])

        data = np.pad(self.data, ((0, yp[-1]+size[1]-self.size[1]),
                                  (0, xp[-1]+size[0]-self.size[0]),
                                  (0, 0)), 'constant', constant_values=1)
        for yi in yp:
            for xi in xp:
                cropbox = [xi, yi, *size]
                new_data = np.array(tf.image.crop_to_bounding_box(
                    data, cropbox[1], cropbox[0], cropbox[3], cropbox[2]).numpy())

                # Get bounding box
                boxes = self.get_new_boxes(cropbox)
                newpic = Picture(boxes, new_data)
                patches.append(newpic)

        return patches, len(xp), len(yp)

    def extract_rnd(self, resize=None, min_cropfactor=0.3, max_cropfactor=1):
        idx = np.random.randint(0, self.box.shape[0])
        crop_factor = np.random.randint(
            min_cropfactor * 100, max_cropfactor * 100 + 1) / 100
        size = (self.box[idx, 2:4] *
                crop_factor).astype('int').clip(min=1, max=self.size)
        coord = np.random.randint(0, self.size - size + 1)

        data = np.array(tf.image.crop_to_bounding_box(
            self.data, coord[1], coord[0], size[1], size[0]))

        pic = Picture(self.get_new_boxes((*coord, *size)), data)
        return pic if resize is None else pic.resize(resize), self.ioc(coord, size)

    def ioc(self, crop, size):
        boxa = np.column_stack(
            (self.box[:, 0], self.box[:, 1], self.box[:, 0]+self.box[:, 2], self.box[:, 1]+self.box[:, 3]))
        boxb = np.append(crop, crop+size)

        xa = np.maximum(boxa[:, 0], boxb[0])
        ya = np.maximum(boxa[:, 1], boxb[1])
        xb = np.minimum(boxa[:, 2], boxb[2])
        yb = np.minimum(boxa[:, 3], boxb[3])

        intersec = np.maximum(0, xb-xa + 1) * np.maximum(0, yb-ya + 1)

        boxa_area = (boxa[:, 2] - boxa[:, 0] + 1) * \
            (boxa[:, 3] - boxa[:, 1] + 1)
        boxb_area = (boxb[2] - boxb[0] + 1) * \
            (boxb[3] - boxb[1] + 1)
        iou = intersec / (boxa_area + boxb_area - intersec)

        return max(iou)

    def get_new_boxes(self, cropbox):
        croparray = np.tile(cropbox, (self.box.shape[0], 1))
        box = self.box

        newbox = np.column_stack(
            [box[:, 0] - croparray[:, 0],
             box[:, 1] - croparray[:, 1],
             np.zeros(box.shape[0]),
             np.zeros(box.shape[0])])
        newbox[:, 0] = newbox[:, 0].clip(min=0, max=croparray[:, 2])
        newbox[:, 1] = newbox[:, 1].clip(min=0, max=cropbox[3])

        idx = np.where(croparray[:, 0] < box[:, 0])
        newbox[:, 2] = box[:, 0] + box[:, 2] - croparray[:, 0]
        newbox[idx, 2] = croparray[idx, 0] + croparray[idx, 2] - box[idx, 0]

        idx = np.where(croparray[:, 1] < box[:, 1])
        newbox[:, 3] = box[:, 1] + box[:, 3] - croparray[:, 1]
        newbox[idx, 3] = croparray[idx, 1] + croparray[idx, 3] - box[idx, 1]

        newbox = np.delete(newbox, np.array(
            np.where(newbox[:, 2:4] <= 0))[0, :], 0)
        return newbox

    def resize(self, length):
        new_data = np.array(tf.cast(tf.image.resize(
            self.data, length, method='area', preserve_aspect_ratio=False), tf.uint8))
        new_size = np.array(
            [new_data.shape[1], new_data.shape[0]])
        new_box = self.box
        if(len(self.box) > 0):
            new_box = np.column_stack([
                new_size[0]*self.box[:, 0]/self.size[0],
                new_size[1]*self.box[:, 1]/self.size[1],
                new_size[0]*self.box[:, 2]/self.size[0],
                new_size[1]*self.box[:, 3]/self.size[1]])
        return Picture(new_box, new_data)

    @ staticmethod
    def draw_patches(patches, xp, yp):
        axes = plt.subplots(yp, xp, figsize=(20, 20))[1]
        axes = axes.flatten()
        i = 0
        for ax in axes:
            data = patches[i].data
            box = patches[i].box
            face = patches[i].face
            ax.imshow(data)
            ax.axis('off')

            for c in box:
                if not face:
                    ax.add_patch(Rectangle((c[0], c[1]), width=c[2], height=c[3],
                                           fill=False, color='r'))
                else:
                    ax.add_patch(Rectangle((c[0], c[1]), width=c[2], height=c[3],
                                           fill=False, color='b'))
            i += 1
        plt.tight_layout()
        plt.show()
