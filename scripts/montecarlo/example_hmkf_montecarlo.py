import math
import numpy as np

sample_num = 10000 * 1000
dt = 0.1

# initial state
ini_mean = np.array([5.0, np.pi/2.0])
ini_cov = np.array([[1.0**2, 0.0], [0.0, (np.pi/30)**2]])

# control info
controls = [1.0*dt, 0.1*dt]

# system noise
wx_lambda = 1.0
upper_wtheta = (np.pi/10.0)
lower_wtheta = -(np.pi/10.0)

# get samples
wx_samples = np.random.exponential(wx_lambda, sample_num)
wtheta_samples = np.random.uniform(upper_wtheta, lower_wtheta, sample_num)
state_samples = np.random.multivariate_normal(ini_mean, ini_cov, sample_num)

# measurement noise
mr_lambda = 2.0
upper_mtheta = (np.pi/20.0)
lower_mtheta = -(np.pi/20.0)
mr_samples = np.random.exponential(mr_lambda, sample_num)
mtheta_samples = np.random.uniform(upper_mtheta, lower_mtheta, sample_num)

sum_x = 0.0
sum_yaw = 0.0
sum_x2 = 0.0
sum_yaw2 = 0.0
sum_x1_yaw1 = 0.0
sum_x3 = 0.0
sum_x2_yaw1 = 0.0
sum_x4 = 0.0
next_x_samples = []
next_yaw_samples = []
for (state, wx, wtheta) in zip(state_samples, wx_samples, wtheta_samples):
    curr_x = state[0]
    curr_yaw = state[1]
    v = controls[0]
    u = controls[1]
    next_x = curr_x + v * np.cos(curr_yaw) + wx
    next_yaw = curr_yaw + u + wtheta
    next_x_samples.append(next_x)
    next_yaw_samples.append(next_yaw)
    sum_x += next_x
    sum_yaw += next_yaw
    sum_x2 += next_x * next_x
    sum_yaw2 += next_yaw * next_yaw
    sum_x1_yaw1 += next_x * next_yaw
    sum_x3 += next_x ** 3
    sum_x2_yaw1 += next_x**2 * next_yaw
    sum_x4 += next_x ** 4

print('E[X]: ', sum_x/sample_num)
print('E[YAW]: ', sum_yaw/sample_num)
print('E[X^2]: ', sum_x2/sample_num)
print('E[YAW^2]: ', sum_yaw2/sample_num)
print('E[X*YAW]: ', sum_x1_yaw1/sample_num)
print('E[X^3]: ', sum_x3/sample_num)
print('E[X^2*YAW]: ', sum_x2_yaw1/sample_num)
print('E[X^4]: ', sum_x4/sample_num)

sum_mr = 0.0
sum_myaw = 0.0
sum_mr_square = 0.0
sum_myaw_square = 0.0
sum_mr_myaw = 0.0
for (x, yaw, mr, myaw) in zip(next_x_samples, next_yaw_samples, mr_samples, mtheta_samples):
    measurement_r = x*x + mr
    measurement_yaw = yaw + myaw
    sum_mr += measurement_r
    sum_myaw += measurement_yaw
    sum_mr_square += measurement_r * measurement_r
    sum_myaw_square += measurement_yaw * measurement_yaw
    sum_mr_myaw += measurement_r * measurement_yaw

print('E[R]: ', sum_mr/sample_num)
print('E[YAW]: ', sum_myaw/sample_num)
print('E[R^2]: ', sum_mr_square/sample_num)
print('E[YAW^2]: ', sum_myaw_square/sample_num)
print('E[R*YAW]: ', sum_mr_myaw/sample_num)
