'''Class for testing models'''

import time
import numpy as np

class ModelTester():
    def __init__(self, mnist, model):
        self.mnist = mnist
        self.model = model

    def optimize(self):
        t_start = time.time()

        self.model.optimize(maxiter=1000)

        t_end = time.time()
        print('Optimization completed in {} seconds'.format(t_end - t_start))

    def test(self, num_test=5):
        X_test = self.mnist.test_images[0:num_test, :, :].reshape((num_test),
            self.mnist.test_rows*self.mnist.test_cols) \
            .astype(np.float64) / 255
        Y_test = self.mnist.test_labels[0:num_test].astype(np.int32)

        # mu, var = self.model.predict_f(X_test)
        p, _ = self.model.predict_y(X_test)

        Y_guess = np.argmax(p, axis=1)
        wrong_guesses = 0
        for i in range(0, num_test):
            if (Y_guess[i] != Y_test[i]):
                wrong_guesses += 1

        print('Tested against {} digits with {:.2f}% accuracy'.format(
            num_test, 100*(num_test-wrong_guesses)/num_test))
