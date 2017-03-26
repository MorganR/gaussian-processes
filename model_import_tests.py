import numpy as np
import matplotlib.pyplot as plt
from model_tester import ModelTester
from data_holder import DataHolder
from accuracy_visualizer import AccuracyVisualizer

# compare dimensions
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
# dimension_visualizer = AccuracyVisualizer(models, dimensions)
# ax = plt.gca()
# dimension_visualizer.plot(
#     ax,
#     'The Effect of Input Dimensionality on Accuracy',
#     'Number of Dimensions in X',
#     'Accuracy (%)')
# plt.show()
# print(dimension_visualizer)

# compare vgp training times
models = [  'vgp-10c-784d-50i-nopca-3000iterations',
            'vgp-10c-784d-100i-nopca-3000iterations',
            'vgp-10c-784d-150i-nopca-3000iterations',
            'vgp-10c-784d-250i-nopca-3000iterations',
            'vgp-10c-784d-500i-nopca-3000iterations',
            'vgp-10c-784d-750i-nopca-3000iterations',
            'vgp-10c-784d-1500i-nopca-3000iterations'
        ]
i = np.array([50, 100, 150, 250, 500, 750, 1500])
# i_visualizer = AccuracyVisualizer(models, i)
# print(i_visualizer)

models = [  'svgp-10c-75d-500i-20z',
            'svgp-10c-75d-500i-50z',
            'svgp-10c-75d-500i-100z',
            'svgp-10c-75d-500i-167z',
            'svgp-10c-75d-500i-250z'
]
z = np.array([20, 50, 100, 167, 250])
z_visualizer = AccuracyVisualizer(models, z)
ax = plt.gca()
z_visualizer.plot(
    ax,
    'The Effect of Inducing Input Number on Accuracy',
    r'Number of Inducing Inputs, $m$',
    'Accuracy (%)'
)
fig = plt.gcf()
fig.savefig('inducing-compare.png')
plt.show()
ax = plt.gca()
z_visualizer.plot_against_t(
    ax,
    'The Effect of Inducing Input Number on SVGP Classification'
)
plt.legend(loc='lower right')
fig = plt.gcf()
fig.savefig('inducing-time-compare.png')
plt.show()
print(z_visualizer)
