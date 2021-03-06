import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from data.image import get_xyz_space

def plot_image(image):
    plt.imshow(image)
    plt.colorbar()
    plt.show()

def plot_image_3d(image):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xyz = get_xyz_space(image)

    ax.plot_trisurf(xyz[0], xyz[1], xyz[2])

    plt.show()

def plot_image_and_model(image, model):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax_im = ax.imshow(image)
    fig.colorbar(ax_im)

    height, width = image.shape

    xx = np.linspace(0, width-1, width*4)
    yy = np.linspace(0, height-1, height*4)
    xy = np.meshgrid(xx, yy)
    xx = xy[0].flatten()
    yy = xy[1].flatten()
    xy = np.stack((xx, yy), axis=-1)

    mean, var = model.predict_y(xy)

    mean = np.reshape(mean, (height*4, width*4))

    ax = fig.add_subplot(212)
    ax_im = ax.imshow(mean)
    fig.colorbar(ax_im)
    plt.show()

def plot_image_and_model_3d(image, model):
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')

    xyz = get_xyz_space(image)

    ax.plot_trisurf(xyz[0], xyz[1], xyz[2])

    height, width = image.shape

    xx = np.linspace(0, width-1, width*4)
    yy = np.linspace(0, height-1, height*4)
    xy = np.meshgrid(xx, yy)
    xx = xy[0].flatten()
    yy = xy[1].flatten()
    xy = np.stack((xx, yy), axis=-1)

    mean, var = model.predict_y(xy)

    ax = fig.add_subplot(212, projection='3d')
    ax.plot_trisurf(xx, yy, mean.flatten(), color='g')

    plt.show()
