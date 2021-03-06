
import random
from math import sin, cos, sqrt, inf, degrees, pi, isclose, atan

from data.cartesian import Point, Vector
from utils.nums import is_zero

class Line:
    """Represents a straight line as distance 'd' from the origin and angle
    'theta' from the x-axis"""

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

    def get_dist_from_line(self, point):
        return point.x*cos(self.theta) + point.y*sin(self.theta) - self.d

    def is_parallel(self, line):
        return isclose(line.theta, self.theta) \
            or isclose(line.theta, ((self.theta + pi) % (2*pi)))

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
        if (self.is_parallel(line)):
            return None
        if (not is_zero(sin(self.theta)) \
                and not is_zero(sin(line.theta))):
            # print("solving with x since sin(s.theta): %f  and sin(l.theta): %f" % (sin(self.theta), sin(line.theta)))
            return self._solve_with_x(line)
        elif (not is_zero(cos(self.theta)) \
                and not is_zero(cos(line.theta))):
            # print("solving with y")
            return self._solve_with_y(line)
        # Else these lines are exactly horizontal and vertical
        if is_zero(sin(self.theta)):
            # self is vertical
            # print("solving for vertical line")
            return Point(self.d/cos(self.theta), line.d/sin(line.theta))
        else:
            # print("solving for horizontal line")
            return Point(line.d/cos(line.theta), self.d/sin(self.theta))

    def __str__(self):
        return "Line with d: %d\ttheta (degrees): %f" \
            % (self.d, degrees(self.theta))

def generate_line(d_min, d_max):
    random.seed()
    d = random.randint(d_min, d_max)
    theta = random.uniform(0, 2*pi)
    return Line(d, theta)

def get_perp_line_through_point(line, point):
    inv_l = Line(0, (line.theta + pi/2) % (2*pi))
    inv_l.d = point.x*cos(inv_l.theta) + point.y*sin(inv_l.theta)
    return inv_l

class Circle:
    """Represents a circle as a center Point and distance r from the center"""

    def __init__(self, r, center):
        self.r = r
        # The center as a Point
        self.center = center

    def get_point(self, theta):
        d_y = r*sin(theta)
        d_x = r*cos(theta)
        return Point(self.center.x + d_x, self.center.y + d_y)

    def get_theta(self, point):
        if point == self.center:
            return None
        d_vect = point - self.center
        return atan(d_vect.y, d_vect.x)

    def __str__(self):
        return "Circle with radius %f and center %s" \
            % (self.r, str(self.center))

def generate_circle(r_min, r_max, offset_min=0, offset_max=0):
    """Generate a random circle with integer radius between r_min and r_max,
    and with a center with abs(x) and abs(y) between offset_min and offset_max"""

    if (r_min < 0 or r_max < 0 or offset_min < 0 or offset_max < 0):
        raise AssertionError("All args to generate_circle must be >= 0")

    random.seed()
    r = random.randint(r_min, r_max)
    x_off = random.randint(0, offset_max - offset_min) + offset_min
    y_off = random.randint(0, offset_max - offset_min) + offset_min

    if random.randint(0, 1) == 1:
        x_off = x_off*(-1)
    if random.randint(0, 1) == 1:
        y_off = y_off*(-1)

    return Circle(r, Point(x_off, y_off))
