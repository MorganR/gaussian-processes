# Test file

import math

import data.shapes
import data.image

def test_line_and_image(d, theta, w, h):
    l = data.shapes.Line(d, theta)
    im = data.image.get_line_image(l, w, h)
    print(l)
    print(im)

def test_vertical_lines():
    test_line_and_image(1, 0, 7, 7)
    test_line_and_image(1, math.pi, 7, 7)

def test_horizontal_lines():
    test_line_and_image(1, math.pi/2, 7, 7)
    test_line_and_image(1, 3*math.pi/2, 7, 7)

if __name__ == "__main__":
    test_vertical_lines()
    test_horizontal_lines()
    # test_line_and_image(1, math.pi*3/2, 10, 10)
