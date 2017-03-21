'''Class for testing models'''

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import os.path

def import_model_tester(model):
    with open('models/'+model+'.pkl', 'rb') as model_file:
        m_test = pickle.load(model_file)
        return m_test

class ModelTester():
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.num_classes = np.unique(data.y).size
        self.num_test_classes = np.unique(data.y_test).size
        self.has_densities = False

    def export(self, title):
        filename = 'models/'+title+'.pkl'
        num_existing = 0
        while os.path.isfile(filename):
            num_existing += 1
            filename = 'models/'+title+'-v{}.pkl'.format(num_existing)
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def optimize(self):
        t_start = time.time()

        self.model.optimize(maxiter=1000)

        t_end = time.time()
        self.optimization_time = t_end-t_start
        print('Optimization completed in {} seconds'.format(self.optimization_time))
        print(self.model.kern)

    def test(self, num_test=5):
        # mu, var = self.model.predict_f(X_test)
        if (num_test > self.data.y_test.size):
            num_test = self.data.y_test.size

        p, var = self.model.predict_y(self.data.x_test[:num_test])

        Y_guess = np.argmax(p, axis=1)
        all_guesses = np.zeros(self.num_test_classes, dtype=np.int32)
        wrong_guesses = np.zeros(self.num_test_classes, dtype=np.int32)
        almost_correct = np.zeros(self.num_test_classes, dtype=np.int32)
        almost_wrong = np.zeros(self.num_test_classes, dtype=np.int32)
        unique_y = np.sort(np.unique(self.data.y))
        for i in range(0, num_test):
            Y_true = self.data.y_test[i]
            Y_idx = np.argwhere(unique_y==Y_true)
            Y_idx = Y_idx[0]
            all_guesses[Y_idx] += 1
            true_prob = p[i, Y_idx]
            true_var = var[i, Y_idx]
            if (Y_guess[i] != Y_true):
                wrong_guesses[Y_idx] += 1
                highest_prob = p[i, Y_guess[i]]
                if highest_prob < (true_prob + true_var):
                    almost_correct[Y_idx] += 1
            else: # guess is correct
                if (np.any(p[i,np.arange(self.num_test_classes)!=Y_idx] > (true_prob - true_var))):
                    almost_wrong[Y_idx] += 1


        print('Tested against {} digits with {:.2f}% accuracy'.format(
            num_test, 100*(num_test-wrong_guesses.sum())/num_test))
        for i in range(0,self.num_test_classes):
            print('\tTested {:d} {}s with {:.2f}% + {:.2f}% - {:.2f}% accuracy'.format(
                all_guesses[i], 
                unique_y[i],
                100*(all_guesses[i] - wrong_guesses[i])/all_guesses[i],
                100*(almost_correct[i]/all_guesses[i]),
                100*(almost_wrong[i]/all_guesses[i])))

    def visualize_density(self, ax, digit_index):
        if not self.has_densities:
            nGrid = 50
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xspaced = np.linspace( xlim[0], xlim[1], nGrid )
            yspaced = np.linspace( ylim[0], ylim[1], nGrid )
            self.xx, self.yy = np.meshgrid( xspaced, yspaced )
            self.Xplot = np.vstack((self.xx.flatten(), self.yy.flatten())).T
            self.p, self.var = self.model.predict_y(self.Xplot)
            self.has_densities = True
        self._plot_contour(ax, self.xx, self.yy, self.p[:,digit_index], [0.5])
        

    # def visualize_density(self, axes):
    #     if self.data.x.shape[1] != 2:
    #         return

    #     if len(axes.shape) > 1:
    #         xlim = axes[0,0].get_xlim()
    #         ylim = axes[0,0].get_ylim()
    #     elif len(axes.shape) == 1:
    #         xlim = axes[0].get_xlim()
    #         ylim = axes[0].get_ylim()
    #     else:
    #         xlim = axes.get_xlim()
    #         ylim = axes.get_ylim()
    #     nGrid = 50
    #     xspaced = np.linspace( xlim[0], xlim[1], nGrid )
    #     yspaced = np.linspace( ylim[0], ylim[1], nGrid )
    #     xx, yy = np.meshgrid( xspaced, yspaced )
    #     Xplot = np.vstack((xx.flatten(),yy.flatten())).T
    #     p, var = self.model.predict_y(Xplot)
    #     if self.num_classes == 2:
    #         return
    #     for i in np.arange(0, self.num_classes):
    #         if len(axes.shape) > 1:
    #             r = int((i) / axes.shape[1])
    #             c = i - r*(axes.shape[1])
    #             self._plot_contour(axes[r,c], xx, yy, p[:,i], [0.5])
    #             axes[r,c].set_title('{}'.format(i))
    #         else:
    #             self._plot_contour(axes[i], xx, yy, p[:,i], [0.5])
    #             axes[i].set_title('{}'.format(i))

    def _plot_contour(self, ax, xx, yy, probability, contours):
        ax.contour(xx, yy, probability.reshape(*xx.shape), contours, colors='k', linewidths=1.2, zorder=100)
