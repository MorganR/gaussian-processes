import numpy as np
import struct
import time

def _import_data(list_to_add_to, filename):
    with open(filename, "rb") as image_file:
        num = struct.unpack(">I",image_file.read(4))[0]
        rows = struct.unpack(">I",image_file.read(4))[0]
        cols = struct.unpack(">I",image_file.read(4))[0]

    list_to_add_to.append(np.memmap(filename, '>B', 'c', offset=12, shape=(num, rows, cols)))

class Galaxies:
    '''A class for importing and accessing galaxy image data'''

    labels = {'spiral': 0, 'elliptical': 1}
    labels_by_idx = [0, 1]
    labels_by_value = {0: 'spiral', 1: 'elliptical'}
    train_images = []
    test_images = []

    num_images_per_file = 1000
    base_files = ['spirals', 'ellipticals']
    folder = 'galaxies'

    def __init__(self):
        for bf in self.base_files:
            _import_data(self.train_images, self.folder+'/'+str(self.num_images_per_file)+'-train-'+bf+'.ubyte')
            _import_data(self.test_images, self.folder+'/'+str(self.num_images_per_file)+'-test-'+bf+'.ubyte')

    def __str__(self):
        return '{} classes with labels '.format(len(self.train_images))+\
            str(self.labels)+\
            '\n{} training images per class'.format(self.train_images[0].size)+\
            '\n{} testing images per class'.format(self.test_images[0].size)

    def get_flattened_train_data(self, num_per_class):
        num = min(num_per_class, self.train_images[0].shape[0])
        num_classes = len(self.train_images)
        num_rows = self.train_images[0].shape[1]
        num_cols = self.train_images[0].shape[2]
        x = self.train_images[0][0:num,:,:].reshape((num, num_rows*num_cols))
        y = np.ones(num)*self.labels_by_idx[0]
        for i in range(1,num_classes):
            x = np.concatenate((x, self.train_images[i][0:num,:,:].reshape((num, num_rows*num_cols))))
            y = np.concatenate((y, np.ones(num)*self.labels_by_idx[i]))
        print(x.shape)
        return x.astype(np.float64)/255, y.astype(np.int32)

    def get_flattened_test_data(self):
        num = self.test_images[0].shape[0]
        num_classes = len(self.test_images)
        num_rows = self.test_images[0].shape[1]
        num_cols = self.test_images[0].shape[2]
        x = self.test_images[0][0:num,:,:].reshape((num, num_rows*num_cols))
        y = np.ones(num)*self.labels_by_idx[0]
        for i in range(1,num_classes):
            x = np.concatenate((x, self.test_images[i][0:num,:,:].reshape((num, num_rows*num_cols))))
            y = np.concatenate((y, np.ones(num)*self.labels_by_idx[i]))
        return x.astype(np.float64)/255, y.astype(np.int32)
