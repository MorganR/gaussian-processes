# Main test file
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import GPflow
import time
from model_tester import ModelTester
from gp.kernels.image import Image as my_image

mnist = MNIST()

print(mnist)

# mnist.rotate_each_image()

# for i in range(0, 5):
#     plt.imshow(mnist.train_ordered[5][i], 'Greys')
#     plt.title(mnist.train_labels[i])
#     plt.show()

# Setup GP data
num_digits = 10
num_images_per_digit = 25
X = np.zeros((num_digits * num_images_per_digit, mnist.train_rows * mnist.train_cols), dtype=np.float64)
Y = np.zeros((num_digits * num_images_per_digit), dtype=np.int32)
for i in range(0, num_images_per_digit):
    for y in range(0, num_digits):
        Y[i + i*y] = y
        X[i + i*y] = mnist.train_ordered[y][i, :, :].flatten().astype(np.float64) / 255

constant_mean = GPflow.mean_functions.Constant(c=0.1)
constant_mean.fixed = False

print('Testing against {} images for each digit.'.format(num_images_per_digit))

m = GPflow.vgp.VGP(
    X,
    Y,
    kern=GPflow.kernels.RBF(input_dim=(mnist.train_rows * mnist.train_cols)),
    likelihood=GPflow.likelihoods.MultiClass(num_digits),
    mean_function=constant_mean,
    num_latent=num_digits)

vgp_tester = ModelTester(mnist, m)
vgp_tester.optimize()
vgp_tester.test(10000)

# m2 = GPflow.vgp.VGP(
#     X,
#     Y,
#     kern=my_image(input_dim=(mnist.train_rows * mnist.train_cols)),
#     likelihood=GPflow.likelihoods.MultiClass(num_digits),
#     mean_function=constant_mean,
#     num_latent=num_digits)
#
# my_tester = ModelTester(mnist,m2)
# my_tester.optimize()
# my_tester.test(5)
