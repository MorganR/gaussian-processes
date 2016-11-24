
import numpy as np
import GPflow

from data.image import get_xyz_space

def get_model_x(image_xyz):
    return np.stack((image_xyz[0], image_xyz[1]), axis=-1)

def get_model_y(image_z):
    return np.reshape(image_z, (image_z.size, 1))

def fit_model(image, kern=GPflow.kernels.RBF(2)):
    flat_x, flat_y, flat_z = get_xyz_space(image)

    pixel_xy = get_model_x([flat_x, flat_y])
    pixel_vals = get_model_y(flat_z)

    m = GPflow.gpr.GPR(pixel_xy, pixel_vals, kern=kern)

    m.optimize()

    return m
