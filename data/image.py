# Generate a 2D-matrix image

from math import sin, cos
import numpy as np

from data.cartesian import Point

def get_center_point(image):
    h, w = image.shape
    return Point((w-1)/2, (h-1)/2)

def get_point_from_center(x, y, w, h):
    return Point(x - (w - 1)/2, y - (h-1)/2)

def get_dist_from_line(x, y, w, h, line):
    p = get_point_from_center(x, y, w, h)
    return line.get_dist_from_line(p)

def is_part_of_line(x, y, w, h, line):
    d = get_dist_from_line(x, y, w, h, line)
    return abs(d) <= 0.5

def get_line_image(line, w=5, h=5):
    m = np.zeros((h, w), np.float64)

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
    im = np.zeros((h, w), np.float64)

    for x in range(0, w):
        for y in range(0, h):
            if is_part_of_circle(x, y, w, h, circle):
                im[y, x] = 1
    return im

def get_xyz_space(image):
    x_axis = np.arange(image.shape[1])
    y_axis = np.arange(image.shape[0])
    xy_grid = np.meshgrid(x_axis, y_axis)
    flat_x = xy_grid[0].flatten().astype(np.float64)
    flat_y = xy_grid[1].flatten().astype(np.float64)
    flat_z = image.flatten().astype(np.float64)

    return flat_x, flat_y, flat_z
