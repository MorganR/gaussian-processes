# This class will be used to generate random straight lines

from math import pi, sin, cos, sqrt, inf, degrees, isclose, atan

class Vector:
    """Represents a 2d cartesian vector"""

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def mag(self):
        return sqrt(self.x*self.x + self.y*self.y)

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        else:
            raise TypeError("Can only add a Vector to a Vector")

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        else:
            raise TypeError("Can only subtract a Vector from a Vector")

    def __eq__(self, other):
        if isinstance(other, Vector):
            return isclose(self.x, other.x) and isclose(self.y, other.y)
        else:
            return False

    def __ne__(self, other):
        return not self == other

class Point(Vector):
    """Represents a 2d cartesian point"""

    def __init__(self, x=0, y=0):
        super().__init__(x, y)

    def __add__(self, other):
        if isinstance(other, Vector) and not isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, Vector) and not isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return super().__sub__(other)
