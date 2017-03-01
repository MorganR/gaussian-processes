import numpy as np
import struct
import time

class MnistPca:
    '''A class for managing a PCA version of MNIST data'''

    def __init__(self, mnist, dim, num_per_digit=0):
        '''Initialize with a regular version of MNIST'''
        self._mnist = mnist
        self.dim = dim
        self._flat_train_images, self._train_labels = mnist.get_flattened_train_data(num_per_digit)
        self.num_images = self._train_labels.size

        self._flat_test_images, self._test_labels = mnist.get_flattened_test_data()
        self.Y = self._train_labels
        self.Y_test = self._test_labels

        self._performPCA()

    def _performPCA(self):
        evecs, evals = np.linalg.eigh(np.cov(self._flat_train_images.T))
        self._i = np.argsort(evecs)[::-1]
        self._W = evals[:, self._i]
        self._W = self._W[:, :self.dim]
        self.X = (self._flat_train_images - self._flat_train_images.mean(0)).dot(self._W)
        self.X_test = (self._flat_test_images - self._flat_test_images.mean(0)).dot(self._W)