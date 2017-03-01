# Main test file
from mnist import MNIST
from mnist_pca import MnistPca
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
num_images_per_digit = 0

num_dimensions = 100
mnist_pca = MnistPca(mnist, num_dimensions)
X = mnist_pca.X
Y = mnist_pca.Y
# num_dimensions = mnist.train_rows*mnist.train_cols
# X, Y = mnist.get_flattened_train_data(num_images_per_digit)

constant_mean = GPflow.mean_functions.Constant(c=0.1)
constant_mean.fixed = False

inducing_offset = 500
inducing_inputs = X[::inducing_offset].copy()
num_images = X.shape[0]
num_inducing_inputs = int(num_images / inducing_offset)

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.cm as cm
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# colors = cm.rainbow(np.linspace(0, 1, num_digits))

# for i,c in zip(np.unique(Y), colors):
#     ax.plot(X[Y==i,0], X[Y==i,1], zs=X[Y==i,2], linestyle='', marker='o', label=i, c=c)

# plt.legend()
# plt.show()


print('Training against {} images using {} inducing inputs.'.format(num_images, inducing_inputs.shape[0]))

m = GPflow.svgp.SVGP(
    X,
    Y,
    kern=GPflow.kernels.RBF(input_dim=num_dimensions),
    likelihood=GPflow.likelihoods.MultiClass(num_digits),
    # mean_function=constant_mean,
    Z=inducing_inputs,
    num_latent=num_digits)
m.Z.fixed = True

# m = GPflow.vgp.VGP(
#     X,
#     Y,
#     kern=GPflow.kernels.RBF(input_dim=num_dimensions),
#     likelihood=GPflow.likelihoods.MultiClass(num_digits),
#     mean_function=constant_mean,
#     num_latent=num_digits)

vgp_tester = ModelTester(mnist_pca, m)
vgp_tester.optimize()
vgp_tester.test(10000)
