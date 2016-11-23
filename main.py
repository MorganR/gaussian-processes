# Main test file

import data.cartesian
import data.image
import data.shapes
from kernels.hough_space import get_line_accumulator_array
import numpy as np
import GPflow
import gp.models.linear
import tensorflow as tf

l = data.shapes.generate_line(0, 0)
print(l)
im = data.image.get_line_image(l, 10, 10)
print(im)
acc_array = get_line_accumulator_array(im)
print(acc_array)

x_axis = np.arange(10.0)
y_axis = np.arange(10.0)
print(type(y_axis[0]))
xy_grid = np.meshgrid(x_axis, y_axis)
print(type(xy_grid[0][0,0]))
#flat_x = xy_grid[0].flatten().astype(np.int32, copy=False)
flat_x = xy_grid[0].flatten()
print(type(flat_x[0]))
#flat_y = xy_grid[1].flatten().astype(np.int32, copy=False)
flat_y = xy_grid[1].flatten()
pixel_xy = np.stack((flat_x, flat_y), axis=-1)
print(type(pixel_xy[0,0]))
pixel_vals = im.flatten()

model = gp.models.linear.Linear(pixel_xy, pixel_vals)
model.optimize()

model

c = data.shapes.generate_circle(2, 4, 0, 2)
print(c)
c_im = data.image.get_circle_image(c, 11, 11)
print(c_im)
