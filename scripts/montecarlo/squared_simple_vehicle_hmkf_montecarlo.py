import math
import numpy as np

sample_num = 10000 * 10000
dt = 0.1

# measurement noise
mean_wr = 1.0
cov_wr = 0.09 ** 2
mean_wa = 0.0
cov_wa = (math.pi/100.0)**2
mr_samples = np.random.normal(mean_wr, math.sqrt(cov_wr), sample_num)#np.random.exponential(mr_lambda, sample_num)
mtheta_samples = np.random.normal(mean_wa, math.sqrt(cov_wa), sample_num) #np.random.uniform(upper_mtheta, lower_mtheta, sample_num)

mean_x = 3.0045114484524555
mean_y = 1.5082653269460484
mean_yaw = 0.964168029865345
cov_x = 0.10000238971645992
cov_y = 0.10015079232158008
cov_yaw = 0.10870760813388891
cov_xy = 0.0009786270528993057
cov_xyaw = -0.000979207100381796
cov_yyaw = 0.00346968320345753

state_mean = np.array([mean_x, mean_y, mean_yaw])
state_cov = np.array([[cov_x, cov_xy, cov_xyaw],
                      [cov_xy, cov_y, cov_yyaw],
                      [cov_xyaw, cov_yyaw, cov_yaw]])
next_state_samples = np.random.multivariate_normal(state_mean, state_cov, sample_num)

sum_mrcos = 0.0
sum_mrsin = 0.0
sum_mrcos_square = 0.0
sum_mrsin_square = 0.0
sum_mrcos_mrsin = 0.0
measurement_rcos_samples = []
measurement_rsin_samples = []

x_land = 3.7629
y_land = -2.03092
for (state, mr, ma) in zip(next_state_samples, mr_samples, mtheta_samples):
    px = state[0]
    py = state[1]
    pyaw = state[2]
    rcos_bearing = (x_land - px) * math.cos(pyaw) + (y_land - py) * math.sin(pyaw)
    rsin_bearing = (y_land - py) * math.cos(pyaw) - (x_land - px) * math.sin(pyaw)

    measurement_rcos = mr * math.cos(ma) * rcos_bearing - mr * math.sin(ma) * rsin_bearing
    measurement_rsin = mr * math.cos(ma) * rsin_bearing + mr * math.sin(ma) * rcos_bearing
    measurement_rcos = measurement_rcos**2
    measurement_rsin = measurement_rsin**2
    sum_mrcos += measurement_rcos
    sum_mrsin += measurement_rsin
    sum_mrcos_square += measurement_rcos**2
    sum_mrsin_square += measurement_rsin**2
    sum_mrcos_mrsin += (measurement_rcos) * (measurement_rsin)
    measurement_rcos_samples.append(measurement_rcos)
    measurement_rsin_samples.append(measurement_rsin)

mean_mrcos = sum_mrcos/sample_num
mean_mrsin = sum_mrsin/sample_num
mean_mrcos_square = sum_mrcos_square/sample_num
mean_mrsin_square = sum_mrsin_square/sample_num
mean_mrcos_mrsin = sum_mrcos_mrsin / sample_num
print('E[RCOS^2]: ', mean_mrcos)
print('E[RSIN^2]: ', mean_mrsin)
print('E[RCOS^4]: ', mean_mrcos_square)
print('E[RSIN^4]: ', mean_mrsin_square)
print('E[RCOS^2*RSIN^2]: ', mean_mrcos_mrsin)
print('V[RCOS^4]: ', mean_mrcos_square - mean_mrcos*mean_mrcos)
print('V[RSIN^4]: ', mean_mrsin_square - mean_mrsin*mean_mrsin)
print('V[RCOS^2*RSIN^2]: ', mean_mrcos_mrsin - mean_mrsin*mean_mrcos)

sum_x_mrcos = 0.0
sum_x_mrsin = 0.0
sum_y_mrcos = 0.0
sum_y_mrsin = 0.0
sum_yaw_mrcos = 0.0
sum_yaw_mrsin = 0.0
for (state, mrcos, mrsin) in zip(next_state_samples, measurement_rcos_samples, measurement_rsin_samples):
    x = state[0]
    y = state[1]
    yaw = state[2]
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
print('V[X*MRCOS]: ', mean_x_mrcos - mean_x * mean_mrcos)
print('V[X*MRSIN]: ', mean_x_mrsin - mean_x * mean_mrsin)
print('V[Y*MRCOS]: ', mean_y_mrcos - mean_y * mean_mrcos)
print('V[Y*MRSIN]: ', mean_y_mrsin - mean_y * mean_mrsin)
print('V[YAW*MRCOS]: ', mean_yaw_mrcos - mean_yaw * mean_mrcos)
print('V[YAW*MRSIN]: ', mean_yaw_mrsin - mean_yaw * mean_mrsin)