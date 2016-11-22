# Generate a 2D-matrix image

from math import sin, cos
from numpy import matrix

from data.cartesian import Point, get_perp_line_through_point

def _generate_2d_matrix(width=5, height=5, init_val=0):
	return matrix([[init_val for x in range(width)] for y in range(height)])

def is_part_of_line(x, y, w, h, line):
	p = Point(x - (w - 1)/2, y - (h-1)/2)
	perp_line = get_perp_line_through_point(line, p)
	intercept = line.get_intercept(perp_line)
	return p.calc_distance(intercept) < 0.5

def line_image(line, w=5, h=5):
	m = _generate_2d_matrix(w, h, 0)

	for x in range(0, w):
		for y in range(0, h):
			if is_part_of_line(x, y, w, h, line):
				m.A[x][y] = 1
	return m
