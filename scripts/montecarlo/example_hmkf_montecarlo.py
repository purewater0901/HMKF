import math
import numpy as np

sample_num = 1000* 10000
dt = 0.1

# initial state
ini_mean = np.array([0.54675, 0.315666])
ini_cov = np.array([[0.194073, 0.102455], [0.102455, 0.0757687]])

# control info
controls = [1.0]

# system noise
wv_lambda = 5.0
upper_wtheta = (np.pi/10.0)
lower_wtheta = -(np.pi/10.0)

# get samples
wv_samples = np.random.exponential(wv_lambda, sample_num)
wtheta_samples = np.random.normal(np.pi/6.0, np.pi/30, sample_num)
state_samples = np.random.multivariate_normal(ini_mean, ini_cov, sample_num)

# measurement noise
upper_lambda = 30
lower_lambda = 0.0
mr_samples = np.random.uniform(lower_lambda, upper_lambda, sample_num)

sum_x = 0.0
sum_y = 0.0
sum_x2 = 0.0
sum_y2 = 0.0
sum_x1_y1 = 0.0
sum_x3 = 0.0
sum_y3 = 0.0
sum_x2_y1 = 0.0
sum_x1_y2 = 0.0
sum_x4 = 0.0
sum_y4 = 0.0
sum_x2_y2 = 0.0
next_x_samples = []
next_y_samples = []
for (state, wv, wtheta) in zip(state_samples, wv_samples, wtheta_samples):
    curr_x = state[0]
    curr_y = state[1]
    v = controls[0]
    next_x = curr_x + (v + wv) * np.cos(wtheta) * dt
    next_y = curr_y + (v + wv) * np.sin(wtheta) * dt
    next_x_samples.append(next_x)
    next_y_samples.append(next_y)
    sum_x += next_x
    sum_y += next_y
    sum_x2 += next_x * next_x
    sum_y2 += next_y * next_y
    sum_x1_y1 += next_x * next_y
    sum_x3 += next_x ** 3
    sum_y3 += next_y ** 3
    sum_x2_y1 += next_x**2 * next_y
    sum_x1_y2 += next_x * next_y**2
    sum_x4 += next_x ** 4
    sum_y4 += next_y ** 4
    sum_x2_y2 += next_x**2 * next_y**2

print('E[X]: ', sum_x/sample_num)
print('E[Y]: ', sum_y/sample_num)
print('E[X^2]: ', sum_x2/sample_num)
print('E[Y^2]: ', sum_y2/sample_num)
print('E[X*Y]: ', sum_x1_y1/sample_num)
print('E[X^3]: ', sum_x3/sample_num)
print('E[Y^3]: ', sum_y3/sample_num)
print('E[X^2*Y]: ', sum_x2_y1/sample_num)
print('E[X*Y^2]: ', sum_x1_y2/sample_num)
print('E[X^4]: ', sum_x4/sample_num)
print('E[Y^4]: ', sum_y4/sample_num)
print('E[X^2Y^2]: ', sum_x2_y2/sample_num)

sum_mr = 0.0
sum_mr_square = 0.0
measurement_r_samples = []
for (x, y, mr) in zip(next_x_samples, next_y_samples, mr_samples):
    measurement_r = x**2 + y**2 + mr
    sum_mr += measurement_r
    sum_mr_square += measurement_r * measurement_r
    measurement_r_samples.append(measurement_r)

print('E[R]: ', sum_mr/sample_num)
print('E[R^2]: ', sum_mr_square/sample_num)

sum_x1_mr1 = 0.0
sum_y1_mr1 = 0.0
for (x, y, mr) in zip(next_x_samples, next_y_samples, measurement_r_samples):
    sum_x1_mr1 += x * mr
    sum_y1_mr1 += y * mr

mean_mr = sum_mr/sample_num
mean_x = sum_x / sample_num
mean_y = sum_y / sample_num
print('V[XR]: ', sum_x1_mr1/sample_num - mean_mr * mean_x)
print('V[YR]: ', sum_y1_mr1/sample_num - mean_mr * mean_y)
print('E[XR]: ', sum_x1_mr1/sample_num)
print('E[YR]: ', sum_y1_mr1/sample_num)
