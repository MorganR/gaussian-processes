# Main test file

import data.cartesian
import data.image
from kernels.hough_space import get_line_accumulator_array

l = data.cartesian.generate_line(0,3)
print(l)

im = data.image.get_line_image(l, 15, 15)
print(im)
acc_array = get_line_accumulator_array(im)
print(acc_array)
