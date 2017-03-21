# Main test file
import numpy as np
import matplotlib.pyplot as plt
from model_tester import ModelTester, import_model_tester
from data_holder import DataHolder, get_mnist_data
from svgp_tester import SvgpTester
from vgp_tester import VgpTester
from mcmc_tester import McmcTester

num_per_digits = [50]
# num_inducing_inputs = [20, 50, 100, 150, 250, 500]

for num_per_digit in num_per_digits:

    digits = np.array([0, 1, 2])
    data = get_mnist_data(digits, num_per_digit)
    data = data.get_pca_data(2)

    m_test = VgpTester(data, 'm12')
    m_test.train()
    m_test.test(10000)

    # for num_inducing_input in num_inducing_inputs:
    #     m_test = SvgpTester(data, num_inducing_input)
    #     m_test.train()
    #     m_test.test(10000)
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


    num_cols = min([len(digits), 5])
    num_rows = int(np.ceil(len(digits)/num_cols))

    fig, axes = plt.subplots(num_rows, num_cols)
    if num_rows == 1 and num_cols == 1:
        data.plot_pca_data(axes, num_per_digit)
        m_test.visualize_density(axes, 0)
        axes.legend()
    elif num_rows > 1:
        for r in np.arange(0, num_rows):
            for col in np.arange(0, num_cols):
                data.plot_pca_data(axes[r,col], 15)
                m_test.visualize_density(axes[r,col], r*num_cols + col)
                axes[r,col].set_title('{}'.format(digits[r*num_cols + col]))
    else:
        for col in np.arange(0, num_cols):
            data.plot_pca_data(axes[col], 15)
            m_test.visualize_density(axes[col], col)
            axes[col].set_title('{}'.format(digits[col]))

    # fig.suptitle('GP Classification Using {} Images and {} Inducing Inputs'.format(
    #     data.y.size, m_test.inducing_inputs.shape[0]
    # ))
    # fig.set_size_inches(18, 8)
    # fig.subplots_adjust(top=0.91, bottom=0.05)
    # fig.savefig('{}-{}.png'.format(data.y.size, m_test.inducing_inputs.shape[0]))
    fig.suptitle('GP Classification on 2D Data\nUsing {} Images with VGP'.format(
        data.y.size
    ))
    fig.set_size_inches(7, 4)
    fig.subplots_adjust(top=0.8, bottom=0.05)
    fig.savefig('vgp-{}c-{}.png'.format(len(digits), data.y.size))
    plt.show()