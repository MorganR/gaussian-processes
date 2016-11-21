# This class will be used to generate random straight lines

import random

def generate_line(d_min, d_max):
	random.seed()
	d = random.randint(d_min, d_max)
	theta = random.randint(0, 359)
	return Line(d, theta)

class Line:
	"""Represents a straight line as distance 'd' from the origin and angle 'theta' from the x-axis"""

	def __init__(self):
		self.d = 0
		self.theta = 0

	def __init__(self, d, theta):
		self.d = d
		self.theta = theta

