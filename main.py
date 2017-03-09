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
num_images_per_digit = 30

num_dimensions = 2
mnist_pca = MnistPca(mnist, num_dimensions, num_images_per_digit)
X = mnist_pca.X
Y = mnist_pca.Y
# num_dimensions = mnist.train_rows*mnist.train_cols
# X, Y = mnist.get_flattened_train_data(num_images_per_digit)

constant_mean = GPflow.mean_functions.Constant(c=0.1)
constant_mean.fixed = False

inducing_offset = 10
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

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, num_digits), alpha=0.5)

num_rows = 2
num_cols = 5

fig, axes = plt.subplots(num_rows, num_cols)

lines = []
for r in np.arange(0, num_rows):
    for col in np.arange(0, num_cols):
        for i,c in zip(np.unique(Y), colors):
            lines.append(axes[r,col].plot(X[Y==i,0][:15], X[Y==i,1][:15], linestyle='', marker='o', c=c, label=i))
            axes[r,col].set_yticks([])
            axes[r,col].set_xticks([])
line_labels = range(0, 10)
# fig.legend((lines[0], lines[1], lines[2], lines[3], lines[4], lines[5], lines[6], lines[7], lines[8], lines[9]),
#     ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'), loc='lower center')

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

for r in np.arange(0, num_rows):
    for c in np.arange(0, num_cols):
        axes[r,c].plot(m.Z.value[:,0], m.Z.value[:,1], 'kx')

# m = GPflow.vgp.VGP(
#     X,
#     Y,
#     kern=GPflow.kernels.RBF(input_dim=num_dimensions),
#     likelihood=GPflow.likelihoods.MultiClass(num_digits),
#     mean_function=constant_mean,
#     num_latent=num_digits)

m_test = ModelTester(mnist_pca, m)
m_test.optimize()
m_test.test(10000)
m_test.visualize_density(axes)

fig.suptitle('GP Classification Using {} Images and {} Inducing Inputs'.format(
    num_images, num_inducing_inputs
))
fig.set_size_inches(18, 8)
fig.subplots_adjust(top=0.91, bottom=0.05)
fig.savefig('{}-{}.png'.format(num_images, num_inducing_inputs))
# plt.show()