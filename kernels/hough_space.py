# Perform operations in hough space to discover shapes in images

from math import sqrt, radians

from numpy import array, zeros, int32

from data.shapes import Line
from data.image import get_line_image

def get_line_accumulator_array(image):
    shape = image.shape
    image_h = shape[0]
    image_w = shape[1]

    d_max = sqrt(image_h*image_h + image_w*image_w)/2
    d_max = int(round(d_max))
    acc_array = zeros((d_max, 36), int32)

    for d in range(d_max):
        for alpha in range(0, 36):
            theta = radians(alpha*10)
            line = Line(d, theta)
            line_image = get_line_image(line, image_w, image_h)
            for y in range(image_h):
                for x in range(image_w):
                    if line_image[y, x] == 1 and image[y, x] == 1:
                        acc_array[d, alpha] = acc_array[d, alpha] + 1
            # if acc_array.A[d][alpha] > 3 or alpha == 9 or alpha == 18 or alpha == 271:
            #     print("Line image for d = %d and alpha = %d (theta: %f)" % (d, alpha*10, theta))
            #     print(line_image)
    return acc_array
