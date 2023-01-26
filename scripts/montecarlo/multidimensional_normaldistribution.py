import math
import numpy as np

sample_num = 1000 * 10000

# initial state
ini_mean = np.array([0.2, 0.5])
ini_cov = np.array([[0.1**2, 0.05**2],
                    [0.05**2, 0.1**2]])
samples = np.random.multivariate_normal(ini_mean, ini_cov, sample_num)

sum_x5 = 0
sum_y5 = 0
sum_x1_y4 = 0
sum_x4_y1 = 0
sum_x2_y3 = 0
sum_x3_y2 = 0
sum_x6 = 0
sum_y6 = 0
sum_x3_y3 = 0
sum_x2_y4 = 0
sum_x4_y2 = 0
sum_x3_y4 = 0
sum_x4_y3 = 0
sum_x4_y4 = 0
for sample in samples:
    x = sample[0]
    y = sample[1]
    sum_x5 += x**5
    sum_y5 += y**5
    sum_x1_y4 += x * y**4
    sum_x4_y1 += x**4 * y
    sum_x2_y3 += x**2 * y**3
    sum_x3_y2 += x**3 * y**2

    sum_x6 += x**6
    sum_y6 += y**6
    sum_x2_y4 += x**2 * y**4
    sum_x4_y2 += x**4 * y**2
    sum_x3_y3 += x**3 * y**3

    sum_x3_y4 += x**3 * y**4
    sum_x4_y3 += x**4 * y**3
    sum_x4_y4 += x**4 * y**4

print('E[X^5]: ', sum_x5/sample_num)
print('E[Y^5]: ', sum_y5/sample_num)
print('E[XY^4]: ', sum_x1_y4/sample_num)
print('E[X^4Y]: ', sum_x4_y1/sample_num)
print('E[X^2Y^3]: ', sum_x2_y3/sample_num)
print('E[X^3Y^2]: ', sum_x3_y2/sample_num)

print('E[X^6]: ', sum_x6/sample_num)
print('E[Y^6]: ', sum_y6/sample_num)
print('E[X^2Y^4]: ', sum_x2_y4/sample_num)
print('E[X^4Y^2]: ', sum_x4_y2/sample_num)
print('E[X^3Y^3]: ', sum_x3_y3/sample_num)

print('E[X^3Y^4]: ', sum_x3_y4/sample_num)
print('E[X^4Y^3]: ', sum_x4_y3/sample_num)
print('E[X^4Y^4]: ', sum_x4_y4/sample_num)
