import numpy as np
import matplotlib.pyplot as plt

def power_fit(x, powers, index):
    p = powers.size
    if p == index + 1:
        return powers[0]
    return powers[index]*x**(p - index - 1) + power_fit(x, powers, index + 1)

images_per_digit = np.array([10, 25, 75, 150])
num_digits = 10
total_images = images_per_digit * num_digits

train_time = np.array([5.1, 164, 2140.8, 12682.5])

plt.plot(total_images, train_time, 'o')

square_fit = np.polyfit(total_images, train_time, 2)
cube_fit = np.polyfit(total_images, train_time, 3)

print('Square fit:', square_fit)
print('Cube fit:', cube_fit)

x_test = np.arange(0, 1501)
# square_test = square_fit[0]*x_test**2 + square_fit[1]*x_test + square_fit[2]
# cube_test = cube_fit[0]*x_test**3 + cube_fit[1]*x_test**2 + cube_fit[2]*x_test + cube_fit[3]
square_test = power_fit(x_test, square_fit, 0)
cube_test = power_fit(x_test, cube_fit, 0)

square_est = power_fit(total_images, square_fit, 0)
cube_est = power_fit(total_images, cube_fit, 0)

square_ss_res = np.sum((square_est - train_time)**2)
cube_ss_res = np.sum((cube_est - train_time)**2)

average_time = np.average(train_time)
ss_tot = np.sum((train_time - average_time)**2)

print(square_ss_res)
print(cube_ss_res)

square_r = square_ss_res/ss_tot
cube_r = cube_ss_res/ss_tot

print('square r^2:', square_r)
print('cube r^2:', cube_r)

plt.plot(x_test, square_test, label='Square')
plt.plot(x_test, cube_test, label='Cube')
plt.title('Training Time vs. Number of Training Images')
plt.ylabel('Training Time (s)')
plt.xlabel('Total Number of Training Images')
plt.legend(loc='lower right')
plt.show()

est = cube_fit[0]*10000**3 + cube_fit[1]*10000**2 + cube_fit[2]*10000 + cube_fit[0]


print('Estimated time for {} images: {:.0f} seconds ({:.2f} hours)'.format(10000, est, est/3600))

