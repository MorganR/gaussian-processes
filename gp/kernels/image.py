import GPflow
import tensorflow as tf
from math import sqrt

class Image(GPflow.kernels.RBF):
    """Performs RBF with a 2D covariance"""

    def __init__(self, input_dim,
        variance=1.0, lengthscales=None, active_dims=None, ARD=False):
        GPflow.kernels.RBF.__init__(self, input_dim, variance,
            lengthscales, active_dims, ARD) # always call the parent constructor

        self.input_shape = int(sqrt(input_dim))

    def square_dist(self, X, X2):
        """Calculate the square distance between points, accounting for
        the shape of the input image."""

        # X = X / self.lengthscales


        return GPflow.kernels.RBF.square_dist(self, X, X2)
