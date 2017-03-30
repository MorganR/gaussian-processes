import numpy as np
import matplotlib.pyplot as plt
from model_tester import ModelTester
from data_holder import DataHolder

class AccuracyVisualizer():
    def __init__(self, filenames, x):
        self.x = x
        self.accuracies = np.empty(x.size)
        self.pos_errors = np.empty((1, x.size))
        self.neg_errors = np.empty((1, x.size))
        self.times = np.empty(x.size)

        i = 0
        for filename in filenames:
            m_test = ModelTester(DataHolder([0], [1], [0], [1], ''), None)
            m_test.import_old(filename)
            m_test.test()
            self.accuracies[i] = m_test.total_accuracy
            self.pos_errors[0,i] = m_test.total_positive_error
            self.neg_errors[0,i] = m_test.total_negative_error
            self.times[i] = m_test.optimization_time
            i += 1
        
        self.errors = np.concatenate((self.neg_errors, self.pos_errors))

    def plot(self, ax, title, x_label, y_label):
        ax.errorbar(self.x, 100*self.accuracies, yerr=100*self.errors, fmt='o')
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    def plot_against_t(self, ax, title):
        for i in range(0, len(self.times)):
            ax.errorbar(self.times[i], 100*self.accuracies[i], yerr=100*self.errors[:,i].reshape(2,1), fmt='o', label=str(self.x[i]))
        ax.set_title(title)
        ax.set_xlabel('Training Time (s)')
        ax.set_ylabel('Accuracy (%)')
        
    def __str__(self):
        string = ''
        for i in range(0, len(self.x)):
            string += '{:.2f}: {:.1f}+{:.1f}-{:.1f}\n'.format(
                self.x[i],
                100*self.accuracies[i],
                100*self.pos_errors[0,i],
                100*self.neg_errors[0,i]
            )
        return string
