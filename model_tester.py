'''Class for testing models'''

import time
import numpy as np
import matplotlib.pyplot as plt

class ModelTester():
    def __init__(self, data, model):
        self.data = data
        self.model = model

    def optimize(self):
        t_start = time.time()

        self.model.optimize(maxiter=1000)

        t_end = time.time()
        print('Optimization completed in {} seconds'.format(t_end - t_start))

    def test(self, num_test=5):
        # mu, var = self.model.predict_f(X_test)
        p, var = self.model.predict_y(self.data.X_test[:num_test])

        Y_guess = np.argmax(p, axis=1)
        all_guesses = np.zeros(10, dtype=np.int32)
        wrong_guesses = np.zeros(10, dtype=np.int32)
        almost_correct = np.zeros(10, dtype=np.int32)
        almost_wrong = np.zeros(10, dtype=np.int32)
        for i in range(0, num_test):
            if (i == 0):
                print(p[i,:])
                print(var[i,:])
            Y_true = self.data.Y_test[i]
            all_guesses[Y_true] += 1
            true_prob = p[i, Y_true]
            true_var = var[i, Y_true]
            if (Y_guess[i] != Y_true):
                wrong_guesses[Y_true] += 1
                highest_prob = p[i, Y_guess[i]]
                if highest_prob < (true_prob + true_var):
                    almost_correct[Y_true] += 1
            else: # guess is correct
                if (np.any(p[i,np.arange(10)!=Y_true] > (true_prob - true_var))):
                    almost_wrong[Y_true] += 1


        print('Tested against {} digits with {:.2f}% accuracy'.format(
            num_test, 100*(num_test-wrong_guesses.sum())/num_test))
        for i in range(0,10):
            print('\tTested {:d} {}s with {:.2f}% + {:.2f}% - {:.2f}% accuracy'.format(
                all_guesses[i], 
                i,
                100*(all_guesses[i] - wrong_guesses[i])/all_guesses[i],
                100*(almost_correct[i]/all_guesses[i]),
                100*(almost_wrong[i]/all_guesses[i])))

    def visualize_density(self, axes):
        # mu, var = self.model.predict_f(self.data.X_test[:1])
        # samples = self.model.predict_f_samples(self.data.X_test[:1], 3)
        # print(mu)
        # print(var)
        # print(samples)
        xlim = axes[0,0].get_xlim()
        ylim = axes[0,0].get_ylim()
        nGrid = 50
        xspaced = np.linspace( xlim[0], xlim[1], nGrid )
        yspaced = np.linspace( ylim[0], ylim[1], nGrid )
        xx, yy = np.meshgrid( xspaced, yspaced )
        Xplot = np.vstack((xx.flatten(),yy.flatten())).T
        p, var = self.model.predict_y(Xplot)
        for i in np.arange(0, 10):
            r = int((i) / axes.shape[1])
            c = i - r*(axes.shape[1])
            axes[r,c].contour(xx, yy, p[:,i].reshape(*xx.shape), [0.33, 0.66], colors='k', linewidths=1.2, zorder=100)
            axes[r,c].set_title('{}'.format(i))
