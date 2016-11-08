
import os
import random
import numpy as np
import cv2
import math

colorWindow = (63, 127, 191, 255)
colorSegment = [tuple([b,g,r]) for b in colorWindow for g in colorWindow for r in colorWindow]
scales = [3, 6, 13, 28]
filters = []
# initializing 4*6 gabor kernals
for ksize in scales:
    for theta in np.arange(0, np.pi, np.pi / 6):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)


class BGRHistogram:
    """
    creates 4*4*4 vector for color feature
    """
    def __init__(self, image=None):
        self.histogram = {i:0 for i in colorSegment}
        if image:
            image = cv2.imread(image)
            for y in image:
                for px in y:
                    bin = (self.classify(px[0]), self.classify(px[1]), self.classify(px[2]))
                    self.histogram[bin] += 1
        else:
            print('Invalid path or no image provided')
            return

    def classify(self, intensity):
        for interval in colorWindow:
            if intensity <= interval:
                return interval

    def bgrVector(self):
        rtn = np.array([self.histogram[bin] for bin in colorSegment])
        return rtn/np.linalg.norm(rtn)


class GaborTexture:
    """
    creates gabor texture vector
    """
    def __init__(self, image=None, partition=3):
        if not image:
            print('Invalid path or no image provided')
            return
        image = cv2.imread(image, 0) #load gray-scale
        self.roi = self.partitionImage(image, partition)

    def partitionImage(self, image, partition):
        rows, columns = image.shape[0], image.shape[1]
        rowIntervalSize, columnIntervalSize = rows/partition, columns/partition
        roi = []
        r1, r2 = 0, 0
        for i in range(partition):
            r1 = r2
            r2 += rowIntervalSize
            c1, c2 = 0, 0
            for j in range(partition):
                c1 = c2
                c2 += columnIntervalSize
                roi.append(image[r1:r2, c1:c2])
        return roi

    def textureVector(self):
        rtn = []
        for img in self.roi:
            for fil in filters:
                fimg, val = cv2.filter2D(img, cv2.CV_8UC3, fil), 0
                for row in fimg:
                    val += np.sum([abs(i) for i in row]) #enery of the convolved image
                rtn.append(val)
        rtn = np.array(rtn)
        return rtn/np.linalg.norm(rtn)


class MinibatchGenerator(object):
    """
    Mini image batch generator from directory path
    may be extended for other training purposes
    """
    def __init__(self, data_path, batch_size=100, training=0.7, validation=0, test=0.3):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size should be a positive integer, %r given" % batch_size)
        self._data_path = data_path
        self._batch_size = batch_size
        self._image_map, self._class_map, self._test_image_map = self.setDictInit(data_path)

    def setDictInit(self, path, training=0.7, validation=0, test=0.3):
        encode_offset = 0
        types = os.listdir(path)
        image_map, class_map, test_image_map = {}, {}, {}
        for t in types:
            if t == '.DS_Store': continue
            training_set, image_names = set(), os.listdir(path + "/" + t)
            size, count = len(image_names), 0
            for img in image_names:
                if count < training*size:
                    s.add(img)
                else:
                    if t in test_image_map:
                        test_image_map[t].add(img)
                    else:
                        test_image_map[t] = set([img])
            image_map[t] = s
            cl = np.zeros(101) # one-hot-encoded, TODO: dynamic sizing
            cl[encode_offset] = 1
            class_map[t] = cl
            encode_offset += 1
        return image_map, class_map, test_image_map

    def next(self):
        remaining = self._batch_size
        type_size = remaining/20 # TODO: dynamic class sizing
        image_tensor, class_tensor = np.array([]), np.array([])
        while remaining - type_size > 0:
            cl = random.sample(self._image_map.keys(), 1)
            type_images = random.sample(self._image_map[cl], type_size)
            for img in type_images:
                image_path = self._data_path + "/" + cl + "/" + img
                feature_v = np.array(BGRHistogram(image_path).bgrVector() + GaborTexture(image_path).textureVector())
                feature_v = feature_v/np.linalg.norm(v)
                image_tensor = np.vstack((image_tensor, feature_v))
                class_tensor = np.vstack((class_tensor, self._class_map[cl]))
            remaining -= type_size
        return (np.array(image_tensor), np.array(class_tensor))

    def testData(self):
        return self._test_image_map

    def classTesnsorMap(self):
        return self._class_map

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()







