import math
import numpy as np

sample_num = 1000 * 10000
dt = 0.1

# initial state
ini_mean = np.array([2.0, 1.0, np.pi/3.0])
ini_cov = np.array([[1.0**2, 0.1, 0.01],
                    [0.1, 1.0**2, 0.2],
                    [0.01, 0.2, (np.pi/10)**2]])

# control info
controls = [1.0*dt, 0.1*dt]

# system noise
wv_lambda = 1.0 / 1.0
upper_wu = (np.pi/10.0) * dt
lower_wu = -(np.pi/10.0) * dt

# get samples
wv_samples = np.random.exponential(wv_lambda, sample_num)
wu_samples = np.random.uniform(upper_wu, lower_wu, sample_num)
state_samples = np.random.multivariate_normal(ini_mean, ini_cov, sample_num)

# measurement noise
mr_lambda = 1.0
upper_mtheta = (np.pi/20.0)
lower_mtheta = -(np.pi/20.0)
mr_samples = np.random.exponential(mr_lambda, sample_num)
mtheta_samples = np.random.uniform(upper_mtheta, lower_mtheta, sample_num)

sum_x = 0.0
sum_y = 0.0
sum_yaw = 0.0
sum_cos = 0.0
sum_sin = 0.0
sum_x2 = 0.0
sum_y2 = 0.0
sum_yaw2 = 0.0
sum_cos2 = 0.0
sum_sin2 = 0.0
sum_x1_y1= 0.0
sum_x1_yaw1 = 0.0
sum_y1_yaw1 = 0.0
sum_x1_cos1 = 0.0
sum_y1_cos1 = 0.0
sum_x1_sin1 = 0.0
sum_y1_sin1 = 0.0
sum_cos1_sin1 = 0.0
sum_yaw1_cos1 = 0.0
sum_yaw1_sin1 = 0.0

sum_x1_cos2 = 0.0
sum_y1_cos2 = 0.0
sum_x1_sin2 = 0.0
sum_y1_sin2 = 0.0
sum_x2_cos1 = 0.0
sum_y2_cos1 = 0.0
sum_x2_sin1 = 0.0
sum_y2_sin1 = 0.0
sum_x1_y1_cos1 = 0.0
sum_x1_y1_sin1 = 0.0
sum_x1_cos1_sin1 = 0.0
sum_y1_cos1_sin1 = 0.0
sum_x1_yaw1_cos1 = 0.0
sum_x1_yaw1_sin1 = 0.0
sum_y1_yaw1_cos1 = 0.0
sum_y1_yaw1_sin1 = 0.0

sum_x2_cos2 = 0.0
sum_y2_cos2 = 0.0
sum_x2_sin2 = 0.0
sum_y2_sin2 = 0.0
sum_xy_cos2 = 0.0
sum_xy_sin2 = 0.0
sum_x2_cos1_sin1 = 0.0
sum_y2_cos1_sin1 = 0.0
sum_xy_cos1_sin1 = 0.0

next_x_samples = []
next_y_samples = []
next_yaw_samples = []
for (state, wv, wu) in zip(state_samples, wv_samples, wu_samples):
    curr_x = state[0]
    curr_y = state[1]
    curr_yaw = state[2]
    v = controls[0]
    u = controls[1]
    next_x = curr_x + (v + wv) * np.cos(curr_yaw)
    next_y = curr_y + (v + wv) * np.sin(curr_yaw)
    next_yaw = curr_yaw + (u + wu)
    next_x_samples.append(next_x)
    next_y_samples.append(next_y)
    next_yaw_samples.append(next_yaw)
    sum_x += next_x
    sum_y += next_y
    sum_yaw += next_yaw
    sum_cos += math.cos(next_yaw)
    sum_sin += math.sin(next_yaw)

    # second order
    """
    sum_x2 += next_x * next_x
    sum_y2 += next_y * next_y
    sum_yaw2 += next_yaw * next_yaw
    sum_cos2 += math.cos(next_yaw) * math.cos(next_yaw)
    sum_sin2 += math.sin(next_yaw) * math.sin(next_yaw)
    sum_x1_y1 += next_x * next_y
    sum_x1_yaw1 += next_x * next_yaw
    sum_y1_yaw1 += next_y * next_yaw
    sum_x1_cos1 += next_x * math.cos(next_yaw)
    sum_y1_cos1 += next_y * math.cos(next_yaw)
    sum_x1_sin1 += next_x * math.sin(next_yaw)
    sum_y1_sin1 += next_y * math.sin(next_yaw)
    sum_cos1_sin1 += math.cos(next_yaw) * math.sin(next_yaw)
    sum_yaw1_cos1 += next_yaw * math.cos(next_yaw)
    sum_yaw1_sin1 += next_yaw * math.sin(next_yaw)
    """

    # third order
    sum_x1_cos2 += next_x * math.cos(next_yaw)**2
    sum_y1_cos2 += next_y * math.cos(next_yaw)**2
    sum_x1_sin2 += next_x * math.sin(next_yaw)**2
    sum_y1_sin2 += next_y * math.sin(next_yaw)**2
    sum_x2_cos1 += next_x**2 * math.cos(next_yaw)
    sum_y2_cos1 += next_y**2 * math.cos(next_yaw)
    sum_x2_sin1 += next_x**2 * math.sin(next_yaw)
    sum_y2_sin1 += next_y**2 * math.sin(next_yaw)
    sum_x1_y1_cos1 += next_x * next_y * math.cos(next_yaw)
    sum_x1_y1_sin1 += next_x * next_y * math.sin(next_yaw)
    sum_x1_cos1_sin1 += next_x * math.cos(next_yaw) * math.sin(next_yaw)
    sum_y1_cos1_sin1 += next_y * math.cos(next_yaw) * math.sin(next_yaw)
    sum_x1_yaw1_cos1 += next_x * next_yaw * math.cos(next_yaw)
    sum_x1_yaw1_sin1 += next_x * next_yaw * math.sin(next_yaw)
    sum_y1_yaw1_cos1 += next_y * next_yaw * math.cos(next_yaw)
    sum_y1_yaw1_sin1 += next_y * next_yaw * math.sin(next_yaw)

    # fouth order
    sum_x2_cos2 += next_x**2 * math.cos(next_yaw) **2
    sum_y2_cos2 += next_y**2 * math.cos(next_yaw) **2
    sum_x2_sin2 += next_x**2 * math.sin(next_yaw) **2
    sum_y2_sin2 += next_y**2 * math.sin(next_yaw) **2
    sum_xy_cos2 += next_x * next_y * math.cos(next_yaw) **2
    sum_xy_sin2 += next_x * next_y * math.sin(next_yaw) **2
    sum_x2_cos1_sin1 += next_x**2 * math.cos(next_yaw) * math.sin(next_yaw)
    sum_y2_cos1_sin1 += next_y**2 * math.cos(next_yaw) * math.sin(next_yaw)
    sum_xy_cos1_sin1 += next_x * next_y * math.cos(next_yaw) * math.sin(next_yaw)
"""
print('E[X]: ', sum_x/sample_num)
print('E[Y]: ', sum_y/sample_num)
print('E[YAW]: ', sum_yaw/sample_num)
print('E[cos]: ', sum_cos/sample_num)
print('E[sin]: ', sum_sin/sample_num)
print('E[X^2]: ', sum_x2/sample_num)
print('E[Y^2]: ', sum_y2/sample_num)
print('E[YAW^2]: ', sum_yaw2/sample_num)
print('E[cos^2]: ', sum_cos2/sample_num)
print('E[sin^2]: ', sum_sin2/sample_num)
print('E[X*Y]: ', sum_x1_y1/sample_num)
print('E[X*YAW]: ', sum_x1_yaw1/sample_num)
print('E[Y*YAW]: ', sum_y1_yaw1/sample_num)
print('E[xcos]: ', sum_x1_cos1/sample_num)
print('E[ycos]: ', sum_y1_cos1/sample_num)
print('E[xsin]: ', sum_x1_sin1/sample_num)
print('E[ysin]: ', sum_y1_sin1/sample_num)
print('E[cossin]: ', sum_cos1_sin1/sample_num)
print('E[yawcos]: ', sum_yaw1_cos1/sample_num)
print('E[yawsin]: ', sum_yaw1_sin1/sample_num)
"""

print('E[xcos^2]: ', sum_x1_cos2/sample_num)
print('E[ycos^2]: ', sum_y1_cos2/sample_num)
print('E[xsin^2]: ', sum_x1_sin2/sample_num)
print('E[ysin^2]: ', sum_y1_sin2/sample_num)
print('E[x^2cos]: ', sum_x2_cos1/sample_num)
print('E[y^2cos]: ', sum_y2_cos1/sample_num)
print('E[x^2sin]: ', sum_x2_sin1/sample_num)
print('E[y^2sin]: ', sum_y2_sin1/sample_num)
print('E[xycos]: ', sum_x1_y1_cos1/sample_num)
print('E[xysin]: ', sum_x1_y1_sin1/sample_num)
print('E[xcossin]: ', sum_x1_cos1_sin1/sample_num)
print('E[ycossin]: ', sum_y1_cos1_sin1/sample_num)
print('E[xyawcos]: ', sum_x1_yaw1_cos1/sample_num)
print('E[xyawsin]: ', sum_x1_yaw1_sin1/sample_num)
print('E[yyawcos]: ', sum_y1_yaw1_cos1/sample_num)
print('E[yyawsin]: ', sum_y1_yaw1_sin1/sample_num)

print('E[x^2cos^2]: ', sum_x2_cos2/sample_num)
print('E[y^2cos^2]: ', sum_y2_cos2/sample_num)
print('E[x^2sin^2]: ', sum_x2_sin2/sample_num)
print('E[y^2sin^2]: ', sum_y2_sin2/sample_num)
print('E[xycos^2]: ', sum_xy_cos2/sample_num)
print('E[xysin^2]: ', sum_xy_sin2/sample_num)
print('E[x^2cossin]: ', sum_x2_cos1_sin1/sample_num)
print('E[y^2cossin]: ', sum_y2_cos1_sin1/sample_num)
print('E[xycossin]: ', sum_xy_cos1_sin1/sample_num)

sum_mrcos = 0.0
sum_mrsin = 0.0
sum_mrcos_square = 0.0
sum_mrsin_square = 0.0
sum_mrcos_mrsin = 0.0
measurement_rcos_samples = []
measurement_rsin_samples = []

x_land = 1.0
y_land = 2.0
for (px, py, pyaw, mr, ma) in zip(next_x_samples, next_y_samples, next_yaw_samples, mr_samples, mtheta_samples):
    rcos_bearing = (x_land - px) * math.cos(pyaw) + (y_land - py) * math.sin(pyaw)
    rsin_bearing = (y_land - py) * math.cos(pyaw) - (x_land - px) * math.sin(pyaw)

    measurement_rcos =  mr * math.cos(ma) * rcos_bearing - mr * math.sin(ma) * rsin_bearing
    measurement_rsin =  mr * math.cos(ma) * rsin_bearing + mr * math.sin(ma) * rcos_bearing
    sum_mrcos += measurement_rcos
    sum_mrsin += measurement_rsin
    sum_mrcos_square += measurement_rcos * measurement_rcos
    sum_mrsin_square += measurement_rsin * measurement_rsin
    sum_mrcos_mrsin += measurement_rcos * measurement_rsin
    measurement_rcos_samples.append(measurement_rcos)
    measurement_rsin_samples.append(measurement_rsin)

print('E[RCOS]: ', sum_mrcos/sample_num)
print('E[RSIN]: ', sum_mrsin/sample_num)
print('E[RCOS^2]: ', sum_mrcos_square/sample_num)
print('E[RSIN^2]: ', sum_mrsin_square/sample_num)
print('E[RCOS*RSIN]: ', sum_mrcos_mrsin/sample_num)


sum_x_mrcos = 0.0
sum_x_mrsin = 0.0
sum_y_mrcos = 0.0
sum_y_mrsin = 0.0
sum_yaw_mrcos = 0.0
sum_yaw_mrsin = 0.0
for (x, y, yaw, mrcos, mrsin) in zip(next_x_samples, next_y_samples, next_yaw_samples, measurement_rcos_samples, measurement_rsin_samples):
    sum_x_mrcos += x * mrcos
    sum_x_mrsin += x * mrsin
    sum_y_mrcos += y * mrcos
    sum_y_mrsin += y * mrsin
    sum_yaw_mrcos += yaw * mrcos
    sum_yaw_mrsin += yaw * mrsin

mean_x_mrcos = sum_x_mrcos / sample_num
mean_x_mrsin = sum_x_mrsin / sample_num
mean_y_mrcos = sum_y_mrcos / sample_num
mean_y_mrsin = sum_y_mrsin / sample_num
mean_yaw_mrcos = sum_yaw_mrcos / sample_num
mean_yaw_mrsin = sum_yaw_mrsin / sample_num

print('E[X*MRCOS]: ', mean_x_mrcos)
print('E[X*MRSIN]: ', mean_x_mrsin)
print('E[Y*MRCOS]: ', mean_y_mrcos)
print('E[Y*MRSIN]: ', mean_y_mrsin)
print('E[YAW*MRCOS]: ', mean_yaw_mrcos)
print('E[YAW*MRSIN]: ', mean_yaw_mrsin)

mean_x = sum_x / sample_num
mean_y = sum_y / sample_num
mean_yaw = sum_yaw / sample_num
mean_mrcos = sum_mrcos / sample_num
mean_mrsin = sum_mrsin/ sample_num
print('V[X*MRCOS]: ', mean_x_mrcos - mean_x * mean_mrcos)
print('V[X*MRSIN]: ', mean_x_mrsin - mean_x * mean_mrsin)
print('V[Y*MRCOS]: ', mean_y_mrcos - mean_y * mean_mrcos)
print('V[Y*MRSIN]: ', mean_y_mrsin - mean_y * mean_mrsin)
print('V[YAW*MRCOS]: ', mean_yaw_mrcos - mean_yaw * mean_mrcos)
print('V[YAW*MRSIN]: ', mean_yaw_mrsin - mean_yaw * mean_mrsin)
