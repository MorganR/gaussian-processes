# Number Utilities

import math

ZERO_THRESHOLD = 1e-6

def is_zero(val):
    return math.isclose(val, 0, abs_tol=ZERO_THRESHOLD)
