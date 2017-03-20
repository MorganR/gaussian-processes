from mnist import MNIST
from mnist_pca import MnistPca
import numpy as np
import matplotlib.cm as cm

mnist = MNIST()

def get_mnist_data(digits, num_per_digit):
    x, y = mnist.get_flattened_train_data(digits, num_per_digit)
    x_test, y_test = mnist.get_flattened_test_data(digits)
    return DataHolder(
        x,
        y,
        x_test,
        y_test
    )

def get_rotated_mnist_Data(digits, num_per_digit):
    mnist.rotate_each_image()
    data = get_mnist_data()
    mnist.undo_image_rotation()
    return data

class DataHolder():
    def __init__(self, x, y, x_test, y_test):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test

    def get_pca_data(self, num_dimensions):
        evecs, evals = np.linalg.eigh(np.cov(self.x.T))
        _i = np.argsort(evecs)[::-1]
        _W = evals[:, _i]
        _W = _W[:, :num_dimensions]
        x = (self.x - self.x.mean(0)).dot(_W)
        x_test = (self.x_test - self.x_test.mean(0)).dot(_W)
        return DataHolder(x, self.y.copy(), x_test, self.y_test.copy())

    def plot_pca_data(self, ax, num_per_digit):
        colors = cm.rainbow(np.linspace(0, 1, np.unique(self.y).size), alpha=0.5)
        for i,c in zip(np.unique(self.y), colors):
            images_per_digit = self.x[self.y==i].shape[0]
            if num_per_digit < images_per_digit:
                iterations = int(round(images_per_digit / num_per_digit))
                ax.plot(self.x[self.y==i][::iterations,0],self.x[self.y==i][::iterations,1], linestyle='', marker='o', c=c, label=i)
            else:
                ax.plot(self.x[self.y==i,0],self.x[self.y==i,1], linestyle='', marker='o', c=c, label=i)
            ax.set_yticks([])
            ax.set_xticks([])