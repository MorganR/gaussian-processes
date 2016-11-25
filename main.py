# Main test file
import GPflow
import numpy as np

import data.cartesian
import data.image
import data.shapes
from kernels.hough_space import get_line_accumulator_array
import gp.image_fitting
import utils.plots

l = data.shapes.generate_line(0, 0)
print(l)
im = data.image.get_line_image(l, 10, 10)
print(im)
acc_array = get_line_accumulator_array(im)
print(acc_array)

model = gp.image_fitting.fit_model(im)

print(model)

utils.plots.plot_image_and_model(im, model)

model2 = gp.image_fitting.fit_linear_model(im)

print(model2)

utils.plots.plot_image_and_model(im, model2)

c = data.shapes.generate_circle(2,4,0,2)
print(c)
c_im = data.image.get_circle_image(c, 11, 11)
print(c_im)

c_model = gp.image_fitting.fit_model(c_im, GPflow.kernels.RBF(2, variance=0.3))

utils.plots.plot_image_and_model(c_im, c_model)
