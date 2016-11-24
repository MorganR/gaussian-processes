import GPflow
import tensorflow as tf
from math import pi, sin, cos
import numpy as np

class Linear(GPflow.model.Model):
    def __init__(self, X, Y):
        GPflow.model.Model.__init__(self) # always call the parent constructor
        self.X = X.copy() # X is an array of 2d-pixels in the format y, x
        self.Y = Y.copy() # Y is an array of pixel values

        self.num_data, self.input_dim = X.shape
        if self.num_data is not Y.size:
            raise AssertionError("X and Y did not have the same number of data \
            points")

        # make some parameters
        self.d = GPflow.param.Param(0)
        self.theta = GPflow.param.Param(pi/2)
        self.d.prior = GPflow.priors.Uniform(0,5)
        self.theta.prior = GPflow.priors.Uniform(0, pi)

    def build_likelihood(self): # takes no arguments
        trig_vect = [tf.cos(self.theta), tf.sin(self.theta)]
        d = tf.matmul(self.X, trig_vect)
        dist_from_line = d  - self.d
        zeros = tf.zeros_like(dist_from_line)
        ones = tf.ones_like(dist_from_line)
        condition = tf.less_equal(tf.abs(dist_from_line), \
                tf.fill(dist_from_line.get_shape(), tf.to_double(0.5)))
        estimates = tf.select(condition, ones, zeros)
        return tf.reduce_prod(tf.exp(-tf.squared_difference(self.Y, estimates))) # be sure to return a scalar
