# Main test file

import data.cartesian
import data.matrix_image

l = data.cartesian.generate_line(0,0)
print("d:", l.d, "theta:", l.theta)

im = data.matrix_image.line_image(l, 5, 5)
print(im)
