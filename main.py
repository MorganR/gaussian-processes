# Main test file
from mnist import MNIST
import matplotlib.pyplot as plt

mnist = MNIST()

print(mnist)

plt.imshow(mnist.train_images[0,:,:], 'Greys')
plt.title(mnist.train_labels[0])
plt.show()

plt.imshow(mnist.train_ordered[2][0,:,:], 'Greys')
plt.show()
