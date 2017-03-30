import numpy as np
import matplotlib.pyplot as plt
from model_tester import ModelTester
from data_holder import DataHolder
from accuracy_visualizer import AccuracyVisualizer

# compare dimensions
models = [  'mnist/vgp-10c-2d-250i',
            'mnist/vgp-10c-3d-250i',
            'mnist/vgp-10c-5d-250i',
            'mnist/vgp-10c-15d-250i',
            'mnist/vgp-10c-25d-250i',
            'mnist/vgp-10c-50d-250i',
            'mnist/vgp-10c-100d-250i',
            'mnist/vgp-10c-150d-250i',
            'mnist/vgp-10c-250d-250i'
        ]
dimensions = np.array([2, 3, 5, 15, 25, 50, 100, 150, 250])
dimension_visualizer = AccuracyVisualizer(models, dimensions)
# ax = plt.gca()
# dimension_visualizer.plot(
#     ax,
#     'The Effect of Input Dimensionality on Accuracy',
#     'Number of Dimensions in X',
#     'Accuracy (%)')
# dimension_visualizer.plot_against_t(
#     ax,
#     'The Effect of Input Dimensionality on MNIST Classification'
# )
# plt.legend(loc='lower right')
# fig = plt.gcf()
# fig.savefig('dimension-time-compare.png')
# plt.show()

# compare vgp training times
models = [  'mnist/vgp-10c-784d-50i-nopca-3000iterations',
            'mnist/vgp-10c-784d-100i-nopca-3000iterations',
            'mnist/vgp-10c-784d-150i-nopca-3000iterations',
            'mnist/vgp-10c-784d-250i-nopca-3000iterations',
            'mnist/vgp-10c-784d-500i-nopca-3000iterations',
            'mnist/vgp-10c-784d-750i-nopca-3000iterations',
            'mnist/vgp-10c-784d-1500i-nopca-3000iterations'
        ]
i = np.array([50, 100, 150, 250, 500, 750, 1500])
# i_visualizer = AccuracyVisualizer(models, i)
# print(i_visualizer)

models = [  'mnist/svgp-10c-75d-500i-20z',
            'mnist/svgp-10c-75d-500i-50z',
            'mnist/svgp-10c-75d-500i-100z',
            'mnist/svgp-10c-75d-500i-167z',
            'mnist/svgp-10c-75d-500i-250z'
]
z = np.array([20, 50, 100, 167, 250])
# z_visualizer = AccuracyVisualizer(models, z)
# ax = plt.gca()
# z_visualizer.plot(
#     ax,
#     'The Effect of Inducing Input Number on Accuracy',
#     r'Number of Inducing Inputs, $m$',
#     'Accuracy (%)'
# )
# fig = plt.gcf()
# fig.savefig('inducing-compare.png')
# plt.show()
# ax = plt.gca()
# z_visualizer.plot_against_t(
#     ax,
#     'The Effect of Inducing Input Number on SVGP Classification'
# )
# plt.legend(loc='lower right')
# fig = plt.gcf()
# fig.savefig('inducing-time-compare.png')
# plt.show()
# print(z_visualizer)

models = [  'galaxies/vgp-2c-2d-500i-rbf',
            'galaxies/vgp-2c-5d-500i-rbf',
            'galaxies/vgp-2c-10d-500i-rbf',
            'galaxies/vgp-2c-50d-500i-rbf',
            'galaxies/vgp-2c-100d-500i-rbf',
            'galaxies/vgp-2c-250d-500i-rbf',
            'galaxies/vgp-2c-500d-500i-rbf',
            'galaxies/vgp-2c-1000d-500i-rbf'
    ]
dimensions = np.array([2, 5, 10, 50, 100, 250, 500, 1000])
# dimension_visualizer = AccuracyVisualizer(models, dimensions)
# ax = plt.gca()
# fig = plt.gcf()
# dimension_visualizer.plot_against_t(
#     ax,
#     'The Effect of Input Dimensionality on Galaxy Classification'
# )
# plt.legend(loc='lower right')
# fig.savefig('galaxy-dimension-compare.png')
# plt.show()

models = [  'galaxies/svgp-2c-100d-10000i-10z-unfixed-rbf-gpu',
            'galaxies/svgp-2c-100d-10000i-25z-unfixed-rbf-gpu',
            'galaxies/svgp-2c-100d-10000i-100z-unfixed-rbf-gpu',
            'galaxies/svgp-2c-100d-10000i-250z-unfixed-rbf-gpu',
            'galaxies/svgp-2c-100d-10000i-1000z-unfixed-rbf-gpu'
]
z = np.array([10, 25, 100, 250, 1000])
# z_visualizer = AccuracyVisualizer(models, z)
# ax = plt.gca()
# fig = plt.gcf()
# z_visualizer.plot_against_t(
#     ax,
#     'The Effect of Inducing Input Numer on Galaxy Classification'
# )
# plt.legend(loc='lower right')
# fig.savefig('galaxy-inducing-compare.png')
# plt.show()