import numpy as np
import matplotlib.pyplot as plt
from curve_fitting import n_cubed, fit_n_cubed

x = np.array([50, 100, 150, 250, 500, 750, 1500])
y = np.array([13.3, 78.2, 157.9, 450.4, 2026.1, 5052.9, 27476])

coeffs, err, chi2 = fit_n_cubed(x, y)

print('coeffs:', coeffs)
print('err:', err)
print('chi2:', chi2)

# q_coeffs = []
# for i in range(0, len(coeffs)):
#     q_coeffs.append(q.Measurement(coeffs[i], err[i]))
# print(q_coeffs[0])
est = n_cubed(60000, coeffs[0], coeffs[1], coeffs[2], coeffs[3])

print('Estimated time for {} images: {:.0f} seconds ({:.2f} hours) ({:.2f} days) ({:.2f} years)'.format(60000, est, est/3600, est/3600/24, est/3600/24/365))

x_test = np.linspace(0, max(x) + 500)
y_test = n_cubed(x_test, coeffs[0], coeffs[1], coeffs[2], coeffs[3])
plt.plot(x, y, 'o')
plt.plot(x_test, y_test, 'k-')
plt.xlabel('Number of Training Images, n')
plt.ylabel('Training Time, t (s)')
plt.title('Approximating Training Time for\nMNIST Classification on a VGP Model')
fig = plt.gcf()
fig.savefig('vgp-train-time.png')
plt.show()