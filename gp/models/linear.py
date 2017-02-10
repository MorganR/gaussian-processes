import GPflow
import tensorflow as tf
from math import pi, sin, cos
import numpy as np

class Linear(GPflow.model.Model):
    """Fits a 2D image to a line in Hough-space."""

    def __init__(self, X, Y, center):
        GPflow.model.Model.__init__(self) # always call the parent constructor
        self.X = X.copy() # X is an array of 2d-pixels in the format y, x
        self.Y = Y.copy() # Y is an array of pixel values
        self.center = center; # The x and y values of the center point of the image

        self.num_data, self.input_dim = X.shape
        if self.num_data is not Y.size:
            raise AssertionError("X and Y did not have the same number of data \
            points")

        # make some parameters
        self.d = GPflow.param.Param(0, transform=GPflow.transforms.positive)
        self.d.prior = GPflow.priors.Uniform(0,1)
        self.d.fixed = True
        self.theta = GPflow.param.Param(0.01, transform=GPflow.transforms.positive)
        self.theta.prior = GPflow.priors.Uniform(0,0.5)

    def build_likelihood(self): # takes no arguments
        """Use tensorflow to calculate the likelihood for the given theta
        and d values. This function will be maximized during optimize()."""

        trig_vect = [tf.cos(self.theta), tf.sin(self.theta)]
        d = tf.matmul(self.X - [self.center.x, self.center.y], trig_vect)
        estimates = tf.exp(-tf.squared_difference(d, self.d))
        return tf.reduce_prod(tf.exp(-tf.squared_difference(self.Y, estimates))) # be sure to return a scalar

    def predict_y(self, X):
        """Predict the Y values that correspond to the given 2D pixels in X
        Returns: mean, variance"""
        w = [[cos(self.theta.value[0])], [sin(self.theta.value[0])]]
        y = np.matmul(X - [self.center.x, self.center.y], w)
        outY = np.exp(-np.square(y-self.d.value[0]))
        return outY, 0 # Return 0 variance for now
