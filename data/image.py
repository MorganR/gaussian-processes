# Generate a 2D-matrix image

from math import sin, cos
from numpy import array, zeros, int8

from data.cartesian import Point
import data.shapes as shapes

def get_point_from_center(x, y, w, h):
    return Point(x - (w - 1)/2, (h-1)/2 - y)

def get_dist_from_line(x, y, w, h, line):
    p = get_point_from_center(x, y, w, h)
    return line.get_dist_from_line(p)

def is_part_of_line(x, y, w, h, line):
    d = get_dist_from_line(x, y, w, h, line)
    return abs(d) <= 0.5

def get_line_image(line, w=5, h=5):
    m = zeros((h, w), int8)

    for x in range(0, w):
        for y in range(0, h):
            if is_part_of_line(x, y, w, h, line):
                m[y, x] = 1
    return m

def is_part_of_circle(x, y ,w, h, circle):
    p = get_point_from_center(x, y, w, h)
    d = (p - circle.center).mag()
    return abs(d - circle.r) <= 0.5

def get_circle_image(circle, w=5, h=5):
    im = zeros((h, w), int8)

    for x in range(0, w):
        for y in range(0, h):
            if is_part_of_circle(x, y, w, h, circle):
                im[y, x] = 1
    return im
