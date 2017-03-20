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

        with open("t10k-images.idx3-ubyte", "rb") as test_file:
            self.test_magic = struct.unpack(">I",test_file.read(4))[0]
            self.test_num = struct.unpack(">I",test_file.read(4))[0]
            self.test_rows = struct.unpack(">I",test_file.read(4))[0]
            self.test_cols = struct.unpack(">I",test_file.read(4))[0]

        self.test_images = np.memmap('t10k-images.idx3-ubyte', '>B', 'c', offset=16, shape=(self.test_num, self.test_rows, self.test_cols))
        self.test_labels = np.memmap('t10k-labels.idx1-ubyte', '>B', 'r', offset=8, shape=(self.test_num))

        self._order_training_sets()
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

    def get_flattened_train_data(self, digits=np.arange(0,10), num_per_digit=0):
        if num_per_digit == 0:
            if len(digits) == 10:
                return self._get_all_flattened(
                    self.train_images,
                    self.train_rows*self.train_cols,
                    self.train_labels)
            return self._get_all_per_digit_flattened(self.train_ordered,
                self.train_rows * self.train_cols, digits)
        num_digits = digits.size
        rows = range(0, num_digits)
        X = np.zeros((num_digits * num_per_digit, self.train_rows * self.train_cols), dtype=np.float64)
        Y = np.zeros((num_digits * num_per_digit), dtype=np.int32)
        for d,r in zip(digits, rows):
            base = r*num_per_digit
            for i in range(0, num_per_digit):
                Y[i + base] = d
                X[i + base] = self.train_ordered[d][i, :, :].flatten().astype(np.float64) / 255
        return X, Y
    
    def get_flattened_test_data(self, digits=np.arange(0,10)):
        if len(digits) == 10:
            return self._get_all_flattened(
                self.test_images, self.test_rows*self.test_cols, self.test_labels)
        return self._get_all_per_digit_flattened(
            self.test_ordered, self.test_rows*self.test_cols, digits)

    def _order_training_sets(self):
        self.train_ordered = []
        self.test_ordered = []
        for i in range(0, 10):
            indices = np.where(self.train_labels == i)
            self.train_ordered.append(self.train_images[indices])
            test_indices = np.where(self.test_labels == i)
            self.test_ordered.append(self.test_images[test_indices])

    def _get_all_per_digit_flattened(self, images, image_size, digits):
        num = 0
        for d in digits:
            num += images[d].shape[0]
        X = np.zeros((num, image_size), dtype=np.float64)
        Y = np.zeros((num), dtype=np.int32)
        d_i = -1
        max_i = 0
        base = 0
        digit = 0
        for i in range(0, num):
            if i == max_i:
                d_i += 1
                digit = digits[d_i]
                base = max_i
                max_i += images[digit].shape[0]
            X[i] = images[digit][i-base,:,:].flatten() / 255
            Y[i] = digit
            
        return X, Y

    def _get_all_flattened(self, images, image_size, labels):
        num = labels.size
        X = images.reshape((num, image_size)).astype(np.float64) / 255
        Y = labels.astype(np.int32)
        return X, Y