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

    def import_old(self, model_str):
        model_shaped_thing = import_model_tester(model_str)
        self.data = model_shaped_thing.data
        self.model = model_shaped_thing.model
        self.num_classes = np.unique(self.data.y).size
        self.num_test_classes = np.unique(self.data.y_test).size
        self.has_densities = False
        self.optimization_time = model_shaped_thing.optimization_time

    def export(self, title):
        base_filename = 'models/'+self.data.name+'/'+title
        filename = base_filename+'.pkl'
        num_existing = 0
        while os.path.isfile(filename):
            num_existing += 1
            filename = base_filename+'-v{}.pkl'.format(num_existing)
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def optimize(self):
        t_start = time.time()

        self.model.optimize(maxiter=1000)

        t_end = time.time()
        self.optimization_time = t_end-t_start
        print('Optimization completed in {} seconds'.format(self.optimization_time))

    def test(self, num_test=0):
        # mu, var = self.model.predict_f(X_test)
        if (num_test == 0 or num_test > self.data.y_test.size):
            num_test = self.data.y_test.size

        p, var = self.model.predict_y(self.data.x_test[:num_test])

        Y_guess = np.argmax(p, axis=1)
        all_guesses = np.zeros(self.num_classes, dtype=np.int32)
        wrong_guesses = np.zeros(self.num_classes, dtype=np.int32)
        almost_correct = np.zeros(self.num_classes, dtype=np.int32)
        almost_wrong = np.zeros(self.num_classes, dtype=np.int32)
        unique_y = np.sort(np.unique(self.data.y))
        for i in range(0, num_test):
            Y_true = self.data.y_test[i]
            if Y_true in unique_y:
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
                    if (np.any(p[i,np.arange(self.num_classes)!=Y_idx] > (true_prob - true_var))):
                        almost_wrong[Y_idx] += 1

        print('all:', all_guesses)
        print('wrong:', wrong_guesses)

        self.right_ratios = almost_correct/all_guesses
        self.wrong_ratios = almost_wrong/all_guesses
        num_test_for_accuracy = np.sum(all_guesses)
        self.accuracy_ratios = (all_guesses - wrong_guesses)/all_guesses
        self.total_accuracy = (num_test_for_accuracy-wrong_guesses.sum())/num_test_for_accuracy
        self.total_positive_error = np.sqrt(np.sum((all_guesses*self.right_ratios/num_test_for_accuracy)**2))
        self.total_negative_error = np.sqrt(np.sum((all_guesses*self.wrong_ratios/num_test_for_accuracy)**2))

        print('Tested against {} digits with {:.2f}% + {:.2f}% - {:.2f}% accuracy'.format(
            num_test, 100*self.total_accuracy, 100*self.total_positive_error, 100*self.total_negative_error))
        for i in range(0,self.num_classes):
            print('\tTested {:d} {}s with {:.2f}% + {:.2f}% - {:.2f}% accuracy'.format(
                all_guesses[i], 
                unique_y[i],
                100*self.accuracy_ratios[i],
                100*self.right_ratios[i],
                100*self.wrong_ratios[i]))

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
