import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy import stats
import platform
import os

if __name__ == '__main__':
    data_num = 30
    os.chdir('../../')
    path = os.getcwd()
    data = []
    for i in range(data_num):
        data.append(pd.read_csv(path + "/result/data/simulation_result/result_monte_carlo_" + str(i) +  ".csv"))

    ekf_xy_rmse = []
    ukf_xy_rmse = []
    mkf_xy_rmse = []
    hmkf_xy_rmse = []
    ekf_yaw_ame = []
    ukf_yaw_ame = []
    mkf_yaw_ame = []
    hmkf_yaw_ame = []
    for i in range(len(data[0])):
        ekf_xy_rmse_sum = 0.0
        ukf_xy_rmse_sum = 0.0
        mkf_xy_rmse_sum = 0.0
        hmkf_xy_rmse_sum = 0.0
        ekf_yaw_ame_sum = 0.0
        ukf_yaw_ame_sum = 0.0
        mkf_yaw_ame_sum = 0.0
        hmkf_yaw_ame_sum = 0.0
        for single_data in data:
            ekf_xy_rmse_sum += single_data['ekf_xy_error'][i]
            ukf_xy_rmse_sum += single_data['ukf_xy_error'][i]
            mkf_xy_rmse_sum += single_data['mkf_xy_error'][i]
            hmkf_xy_rmse_sum += single_data['hmkf_xy_error'][i]
            ekf_yaw_ame_sum += single_data['ekf_yaw_error'][i]
            ukf_yaw_ame_sum += single_data['ukf_yaw_error'][i]
            mkf_yaw_ame_sum += single_data['mkf_yaw_error'][i]
            hmkf_yaw_ame_sum += single_data['hmkf_yaw_error'][i]
        ekf_xy_rmse.append(ekf_xy_rmse_sum/len(data))
        ukf_xy_rmse.append(ukf_xy_rmse_sum/len(data))
        mkf_xy_rmse.append(mkf_xy_rmse_sum/len(data))
        hmkf_xy_rmse.append(hmkf_xy_rmse_sum/len(data))
        ekf_yaw_ame.append(ekf_yaw_ame_sum/len(data))
        ukf_yaw_ame.append(ukf_yaw_ame_sum/len(data))
        mkf_yaw_ame.append(mkf_yaw_ame_sum/len(data))
        hmkf_yaw_ame.append(hmkf_yaw_ame_sum/len(data))

    fig = plt.figure(figsize=(15.5,8.5))
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams["font.size"] = 28
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in
    plt.rcParams['font.family'] = 'Times New Roman'

    ax1 = fig.add_subplot(121)
    ax1.plot(hmkf_xy_rmse, color="black", linewidth=2.0, label="MKF", linestyle="solid")
    ax1.plot(mkf_xy_rmse, color="red", linewidth=2.0, label="MKF", linestyle="dotted")
    ax1.plot(ekf_xy_rmse, color="blue", linewidth=1.5, label="EKF", linestyle="dashed")
    ax1.plot(ukf_xy_rmse, color="green", linewidth=1.5, label="UKF", linestyle="dashdot")
    ax1.set_xlabel(r"time step", fontsize=40)
    ax1.set_ylabel(r"position error [m]", fontsize=40)
    ax1.legend(fontsize=25)
    ax1.set_title("Position RMSE")

    ax2 = fig.add_subplot(122)
    ax2.plot(hmkf_yaw_ame, color="black", linewidth=2.0, label="MKF", linestyle="solid")
    ax2.plot(mkf_yaw_ame, color="red", linewidth=2.0, label="MKF", linestyle="dotted")
    ax2.plot(ekf_yaw_ame, color="blue", linewidth=1.5, label="EKF", linestyle="dashed")
    ax2.plot(ukf_yaw_ame, color="green", linewidth=1.5, label="UKF", linestyle="dashdot")
    ax2.set_xlabel(r"time step", fontsize=40)
    ax2.set_ylabel(r"yaw angle error", fontsize=40)
    ax2.legend(fontsize=25)
    ax1.set_title("Yaw Angle AME")

    plt.show()

