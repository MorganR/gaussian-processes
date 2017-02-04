# Main test file
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import GPflow

mnist = MNIST()

print(mnist)

# plt.imshow(mnist.train_images[0,:,:], 'Greys')
# plt.title(mnist.train_labels[0])
# plt.show()
#
# plt.imshow(mnist.train_ordered[2][0,:,:], 'Greys')
# plt.show()

# Setup GP data
num_digits = 10
num_images_per_digit = 25
X = np.zeros((num_digits * num_images_per_digit, mnist.train_rows * mnist.train_cols), dtype=np.float64)
Y = np.zeros((num_digits * num_images_per_digit), dtype=np.int32)
for i in range(0, num_images_per_digit):
    for y in range(0, num_digits):
        Y[i + i*y] = y
        X[i + i*y] = mnist.train_ordered[y][i, :, :].flatten().astype(np.float64) / 255

print(Y.size, Y.shape)
print(X.size, X.shape)

m = GPflow.vgp.VGP(
    X,
    Y,
    kern=GPflow.kernels.RBF(mnist.train_rows * mnist.train_cols),
    likelihood=GPflow.likelihoods.MultiClass(num_digits),
    num_latent=num_digits)

m.optimize(maxiter=10000)

print('Optimization complete.')

num_test = 5
X_test = mnist.test_images[0:num_test, :, :].reshape((num_test), mnist.test_rows*mnist.test_cols) \
    .astype(np.float64) / 255
Y_test = mnist.test_labels[0:num_test].astype(np.int32)

mu, var = m.predict_f(X_test)
p, _ = m.predict_y(X_test)

for i in range(0, num_test):
    print("Image {}, Digit {}".format(i, Y_test[i]))
    for c in range(m.likelihood.num_classes):
        print('\tmean_{}: {}, variance_{}: {}'.format(c, mu[i,c], c, var[i,c]))
        print('\tprobability_{}: {}'.format(c, p[i,c]))
