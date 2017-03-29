# Galaxy main test file
import numpy as np
import matplotlib.pyplot as plt
from model_tester import ModelTester, import_model_tester
from data_holder import DataHolder, get_mnist_data, get_galaxy_data
from svgp_tester import SvgpTester
from vgp_tester import VgpTester
from mcmc_tester import McmcTester
from galaxy_tester import GalaxyTester

num_inducing_inputs = [250]

num_per_class = 5000

data = get_galaxy_data(num_per_class)
data = data.get_pca_data(100)

# m_test = VgpTester(data, 'rbf')
# m_test.train()
# m_test.test()

for num_inducing_input in num_inducing_inputs:
    m_test = SvgpTester(data, 'rbf', num_inducing_input, False)
    m_test.train()
    m_test.test()
    g_test = GalaxyTester(m_test.data, m_test.model)
    g_test.test()


# filename = 'svgp-2c-100d-10000i-250z-unfixed-rbf-gpu-v1'
# m_test = ModelTester(DataHolder([0], [1], [0], [1], 'dummy'), None)
# m_test.import_old(filename)
# print(m_test.optimization_time)
# m_test.test()
# g_test = GalaxyTester(m_test.data, m_test.model)
# g_test.test()

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.cm as cm
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# colors = cm.rainbow(np.linspace(0, 1, num_digits))

# for i,c in zip(np.unique(Y), colors):
#     ax.plot(X[Y==i,0], X[Y==i,1], zs=X[Y==i,2], linestyle='', marker='o', label=i, c=c)

# num_inducing=100
# m_test = SvgpTester(data, num_inducing)
# m_test.plot_z(axes, num_rows, num_cols)


# num_cols = 1
# num_rows = 1

# fig, axes = plt.subplots(num_rows, num_cols)
# if num_rows == 1 and num_cols == 1:
#     data.plot_pca_data(axes, num_per_class)
#     m_test.plot_z(axes)
#     m_test.visualize_density(axes, 0)
#     axes.legend()
# elif num_rows > 1:
#     for r in np.arange(0, num_rows):
#         for col in np.arange(0, num_cols):
#             data.plot_pca_data(axes[r,col], 15)
#             m_test.visualize_density(axes[r,col], r*num_cols + col)
#             axes[r,col].set_title('{}'.format(r*num_cols + col))
# else:
#     for col in np.arange(0, num_cols):
#         data.plot_pca_data(axes[col], 15)
#         m_test.visualize_density(axes[col], col)
#         axes[col].set_title('{}'.format(col))

# fig.suptitle('GP Classification of Galaxies Using {} Images\nand {} Inducing Inputs when Z was Fixed'.format(
#     data.y.size, m_test.inducing_inputs.shape[0]
# ))
# fig.subplots_adjust(top=0.9, bottom=0.05)
# fig.savefig('svgp-galaxies-{}-{}-fixed.png'.format(data.y.size, m_test.inducing_inputs.shape[0]))
# plt.show()
# fig.suptitle('GP Classification on 2D Data\nUsing {} Images with VGP'.format(
#     data.y.size
# ))
# fig.set_size_inches(7, 4)
# fig.subplots_adjust(top=0.8, bottom=0.05)
# fig.savefig('svgp-{}c-{}.png'.format(len(digits), data.y.size))
# plt.show()