import numpy as np
import struct
import time

class Mnist:
    '''A class for importing and accessing MNIST data'''
    def __init__(self):
        with open("train-images.idx3-ubyte", "rb") as image_file:
            self.train_magic = struct.unpack(">I",image_file.read(4))[0]
            self.train_num = struct.unpack(">I",image_file.read(4))[0]
            self.train_rows = struct.unpack(">I",image_file.read(4))[0]
            self.train_cols = struct.unpack(">I",image_file.read(4))[0]

        self.train_images = np.memmap('train-images.idx3-ubyte', '>B', 'r', offset=16, shape=(self.train_num, self.train_rows, self.train_cols))
        self.train_labels = np.memmap('train-labels.idx1-ubyte', '>B', 'r', offset=8, shape=(self.train_num))
        print('Done importing.')

    def __str__(self):
        return ('Training Set Info\n' +
            'Magic: {}, Number of Images: {}, Image Shape: ({}, {})\n'.format(
            self.train_magic, self.train_num, self.train_rows, self.train_cols))
