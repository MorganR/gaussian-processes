# This class will be used to generate random straight lines

import random
from math import pi, sin, cos, sqrt, inf, degrees, isclose

from data.errors import InterceptError

def generate_line(d_min, d_max):
	random.seed()
	d = random.randint(d_min, d_max)
	theta = random.uniform(0, 2*pi)
	return Line(d, theta)

class Point:
	"""Represents a 2d cartesian point"""

	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y

	def calc_distance(self, point):
		x_dist = self.x - point.x
		y_dist = self.y - point.y
		return sqrt(x_dist*x_dist + y_dist*y_dist)

def get_perp_line_through_point(line, point):
	inv_l = Line(0, (line.theta + pi/2) % (2*pi))
	inv_l.d = point.x*cos(inv_l.theta) + point.y*sin(inv_l.theta)
	return inv_l

class Line:
	"""Represents a straight line as distance 'd' from the origin and angle
	'theta' from the x-axis"""

	THETA_EQUIVALENCE = 1e-5

	def __init__(self, d=0, theta=0):
		self.d = d
		self.theta = theta % (2*pi)

	def get_slope(self):
		return sin(self.theta) / cos(self.theta)

	def calc_x(self, y):
		if cos(self.theta) == 0:
			if y == d/sin(self.theta):
				return inf
			return None
		return (self.d - y*sin(self.theta))/cos(self.theta)

	def calc_y(self, x):
		if sin(self.theta) == 0:
			if x == d/cos(self.theta):
				return inf
			return None
		return (self.d - x*cos(self.theta))/sin(self.theta)

	def _solve_with_x(self, line):
		x =   (line.d/sin(line.theta) - self.d/sin(self.theta)) \
			/ (cos(line.theta)/sin(line.theta) - cos(self.theta)/sin(self.theta))
		y = self.calc_y(x)
		return Point(x, y)

	def _solve_with_y(self, line):
		y =   (line.d/cos(line.theta) - self.d/cos(self.theta)) \
			/ (sin(line.theta)/cos(line.theta) - sin(self.theta)/cos(self.theta))
		x = self.calc_x(y)
		return Point(x, y)

	def get_intercept(self, line):
		if (isclose(line.theta, self.theta) or isclose(line.theta, ((self.theta + pi) % (2*pi)))):
			raise InterceptError("Cannot find intercept of parallel lines")
		if (not isclose(sin(self.theta), 0, abs_tol=Line.THETA_EQUIVALENCE) \
				and not isclose(sin(line.theta), 0, abs_tol=Line.THETA_EQUIVALENCE)):
			# print("solving with x since sin(s.theta): %f  and sin(l.theta): %f" % (sin(self.theta), sin(line.theta)))
			return self._solve_with_x(line)
		elif (not isclose(cos(self.theta), 0, abs_tol=Line.THETA_EQUIVALENCE) \
				and not isclose(cos(line.theta),0, abs_tol=Line.THETA_EQUIVALENCE)):
			# print("solving with y")
			return self._solve_with_y(line)
		# Else these lines are exactly horizontal and vertical
		if isclose(sin(self.theta), 0, abs_tol=Line.THETA_EQUIVALENCE):
			# self is vertical
			# print("solving for vertical line")
			return Point(self.d/cos(self.theta), line.d/sin(line.theta))
		else:
			# print("solving for horizontal line")
			return Point(line.d/cos(line.theta), self.d/sin(self.theta))

	def __str__(self):
		return "d: %d\ttheta (degrees): %f" % (self.d, degrees(self.theta))
