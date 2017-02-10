'''Class for testing models'''

import time
import numpy as np

class ModelTester():
    def __init__(self, mnist, model):
        self.mnist = mnist
        self.model = model

    def optimize(self):
        t_start = time.time()

        self.model.optimize(maxiter=5000)

        t_end = time.time()
        print('Optimization completed in {} seconds'.format(t_end - t_start))

    def test(self, num_test=5):
        X_test = self.mnist.test_images[0:num_test, :, :].reshape((num_test),
            self.mnist.test_rows*self.mnist.test_cols) \
            .astype(np.float64) / 255
        Y_test = self.mnist.test_labels[0:num_test].astype(np.int32)

        # mu, var = self.model.predict_f(X_test)
        p, _ = self.model.predict_y(X_test)

        for i in range(0, num_test):
            print("Image {}, Digit {}".format(i, Y_test[i]))
            for c in range(self.model.likelihood.num_classes):
                # print('\tmean_{}: {}, variance_{}: {}'.format(c, mu[i,c], c, var[i,c]))
                print('\tprobability_{}: {}'.format(c, p[i,c]))
