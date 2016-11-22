# Main test file

import data.cartesian
import data.image
import data.shapes
from kernels.hough_space import get_line_accumulator_array

l = data.shapes.generate_line(2,4)
print(l)
im = data.image.get_line_image(l, 10, 10)
print(im)
acc_array = get_line_accumulator_array(im)
print(acc_array)

c = data.shapes.generate_circle(2, 4, 0, 2)
print(c)
c_im = data.image.get_circle_image(c, 11, 11)
print(c_im)
