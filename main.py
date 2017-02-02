# Main test file
from mnist import Mnist
import matplotlib.pyplot as plt
import matplotlib.colors as colors

test = Mnist()

print(test)

plt.imshow(test.train_images[0,:,:], 'Greys')
plt.title(test.train_labels[0])
plt.show()
plt.imshow(test.train_images[1,:,:], 'Greys')
plt.title(test.train_labels[1])
plt.show()