import numpy as np
import struct
import time

class MNIST:
    '''A class for importing and accessing MNIST data'''

    def __init__(self):
        with open("train-images.idx3-ubyte", "rb") as image_file:
            self.train_magic = struct.unpack(">I",image_file.read(4))[0]
            self.train_num = struct.unpack(">I",image_file.read(4))[0]
            self.train_rows = struct.unpack(">I",image_file.read(4))[0]
            self.train_cols = struct.unpack(">I",image_file.read(4))[0]

        self.train_images = np.memmap('train-images.idx3-ubyte', '>B', 'c', offset=16, shape=(self.train_num, self.train_rows, self.train_cols))
        self.train_labels = np.memmap('train-labels.idx1-ubyte', '>B', 'r', offset=8, shape=(self.train_num))
        self._order_training_sets()

        with open("t10k-images.idx3-ubyte", "rb") as test_file:
            self.test_magic = struct.unpack(">I",test_file.read(4))[0]
            self.test_num = struct.unpack(">I",test_file.read(4))[0]
            self.test_rows = struct.unpack(">I",test_file.read(4))[0]
            self.test_cols = struct.unpack(">I",test_file.read(4))[0]

        self.test_images = np.memmap('t10k-images.idx3-ubyte', '>B', 'c', offset=16, shape=(self.test_num, self.test_rows, self.test_cols))
        self.test_labels = np.memmap('t10k-labels.idx1-ubyte', '>B', 'r', offset=8, shape=(self.test_num))

        print('Done importing.')

        self.X, self.Y = self.get_flattened_train_data()
        self.X_test, self.Y_test = self.get_flattened_test_data()

    def __str__(self):
        return ('Training Set Info\n' +
            '\tMagic: {}, Number of Images: {}, Image Shape: ({}, {})\n'.format(
            self.train_magic, self.train_num, self.train_rows, self.train_cols) +
            'Test Set Info\n' +
            '\tMagic: {}, Number of Images: {}, Image Shape: ({}, {})\n'.format(
                self.test_magic, self.test_num, self.test_rows, self.test_cols)
            )

    def rotate_each_image(self):
        for i in range(0, 10):
            times_90 = 0
            for j in range(0, len(self.train_ordered[i])):
                self.train_ordered[i][j,:,:] = np.rot90(self.train_ordered[i][j,:,:], times_90)
                times_90 = (times_90 + 1) % 4
        times_90 = 0
        for i in range(0, self.test_num):
            self.test_images[i,:,:] = np.rot90(self.test_images[i,:,:], times_90)
            times_90 = (times_90 + 1) % 4


    def undo_image_rotation(self):
        for i in range(0, 10):
            times_90 = 0
            for j in range(0, len(self.train_ordered[i])):
                self.train_ordered[i][j,:,:] = np.rot90(self.train_ordered[i][j,:,:], -times_90)
                times_90 = (times_90 + 1) % 4
        times_90 = 0
        for i in range(0, self.test_num):
            self.test_images[i,:,:] = np.rot90(self.test_images[i,:,:], -times_90)
            times_90 = (times_90 + 1) % 4

    def get_flattened_train_data(self, num_per_digit=0):
        if num_per_digit == 0:
            return self._get_all_flattened(self.train_images, self.train_rows * self.train_cols, self.train_labels)
        num_digits = np.unique(self.train_labels).size
        X = np.zeros((num_digits * num_per_digit, self.train_rows * self.train_cols), dtype=np.float64)
        Y = np.zeros((num_digits * num_per_digit), dtype=np.int32)
        for i in range(0, num_per_digit):
            for y in range(0, num_digits):
                Y[i + i*y] = y
                X[i + i*y] = self.train_ordered[y][i, :, :].flatten().astype(np.float64) / 255
        return X, Y
    
    def get_flattened_test_data(self):
        return self._get_all_flattened(self.test_images, self.test_rows*self.test_cols, self.test_labels)

    def _order_training_sets(self):
        self.train_ordered = []
        for i in range(0, 10):
            indices = np.where(self.train_labels == i)
            self.train_ordered.append(self.train_images[indices])

    def _get_all_flattened(self, images, image_size, labels):
        num = images.shape[0]
        X = images[0:num, :, :].reshape((num),
            image_size) \
            .astype(np.float64) / 255
        Y = labels[0:num].astype(np.int32)
        return X, Y
