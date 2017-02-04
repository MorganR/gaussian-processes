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

        self.train_images = np.memmap('train-images.idx3-ubyte', '>B', 'r', offset=16, shape=(self.train_num, self.train_rows, self.train_cols))
        self.train_labels = np.memmap('train-labels.idx1-ubyte', '>B', 'r', offset=8, shape=(self.train_num))
        self._order_training_sets()

        with open("t10k-images.idx3-ubyte", "rb") as test_file:
            self.test_magic = struct.unpack(">I",test_file.read(4))[0]
            self.test_num = struct.unpack(">I",test_file.read(4))[0]
            self.test_rows = struct.unpack(">I",test_file.read(4))[0]
            self.test_cols = struct.unpack(">I",test_file.read(4))[0]

        self.test_images = np.memmap('t10k-images.idx3-ubyte', '>B', 'r', offset=16, shape=(self.test_num, self.test_rows, self.test_cols))
        self.test_labels = np.memmap('t10k-labels.idx1-ubyte', '>B', 'r', offset=8, shape=(self.test_num))

        print('Done importing.')

    def __str__(self):
        return ('Training Set Info\n' +
            '\tMagic: {}, Number of Images: {}, Image Shape: ({}, {})\n'.format(
            self.train_magic, self.train_num, self.train_rows, self.train_cols) +
            'Test Set Info\n' +
            '\tMagic: {}, Number of Images: {}, Image Shape: ({}, {})\n'.format(
                self.test_magic, self.test_num, self.test_rows, self.test_cols)
            )

    def _order_training_sets(self):
        self.train_ordered = []
        for i in range(0, 10):
            indices = np.where(self.train_labels == i)
            self.train_ordered.append(self.train_images[indices])
