import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

import GPflow

import math

N = 16
X = np.random.rand(N,1)
Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*0.1 + 3

k = GPflow.kernels.Matern52(1, lengthscales=0.3)

m = GPflow.gpr.GPR(X, Y, kern=k)
m.likelihood.variance = 0.01

m.optimize()

xx = np.linspace(-0.1, 1.1, 100)[:,None]
mean, var = m.predict_y(xx)
plt.figure(figsize=(12, 6))
plt.plot(X, Y, 'kx', mew=2)
plt.plot(xx, mean, 'b', lw=2)
plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)
plt.xlim(-0.1, 1.1)

print(m)

plt.show()
