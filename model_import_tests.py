import numpy as np
import matplotlib.pyplot as plt
from model_tester import ModelTester
from data_holder import DataHolder

models = [  'vgp-10c-2d-250i',
            'vgp-10c-3d-250i',
            'vgp-10c-5d-250i',
            'vgp-10c-15d-250i',
            'vgp-10c-25d-250i',
            'vgp-10c-50d-250i',
            'vgp-10c-100d-250i',
            'vgp-10c-150d-250i',
            'vgp-10c-250d-250i'
        ]

dimensions = np.array([2, 3, 5, 15, 25, 50, 100, 150, 250])

accuracies = np.empty(dimensions.size)
pos_errors = np.empty((1, dimensions.size))
neg_errors = np.empty((1, dimensions.size))
errors = np.concatenate((neg_errors, pos_errors))

i = 0
for m_str in models:
    m_test = ModelTester(DataHolder([0], [1], [0], [1]), None)
    m_test.import_old(m_str)
    m_test.test(10000)
    accuracies[i] = m_test.total_accuracy
    pos_errors[0,i] = m_test.total_positive_error
    neg_errors[0,i] = m_test.total_negative_error
    i += 1

print(neg_errors)
print(pos_errors)
errors = np.concatenate((neg_errors, pos_errors))

plt.errorbar(dimensions, 100*accuracies, yerr=100*errors, fmt='o')
plt.title('The Effect of Input Dimensionality on Accuracy')
plt.xlabel('Number of Dimensions in X')
plt.ylabel('Accuracy (%)')
fig = plt.gcf()
fig.savefig('dimension-compare.png')
plt.show()