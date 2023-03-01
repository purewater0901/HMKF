import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy import stats
import platform
import os

if __name__ == '__main__':
    dt = 0.2
    data_num = 20
    os.chdir('../../')
    path = os.getcwd()
    data = []
    for i in range(data_num):
        data.append(pd.read_csv(path + "/result/data/simulation_result/squared_result_monte_carlo_" + str(i) +  ".csv"))

    times = []
    target_data = data[13]
    for i in range(len(target_data)):
        times.append(target_data['time'][i])

    fig = plt.figure(figsize=(20.5,20.5))
    #plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams["font.size"] = 28
    plt.rcParams['xtick.labelsize'] = 35
    plt.rcParams['ytick.labelsize'] = 35
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.plot(target_data["x_true"], target_data["y_true"], color="purple", linewidth=3.0, label="True")
    plt.plot(target_data["hmkf_x"], target_data["hmkf_y"], color="black", linewidth=3.0, label="HMKF")
    plt.plot(target_data["mkf_x"], target_data["mkf_y"], color="red", linewidth=3.0, label="MKF",  linestyle="dotted")
    plt.plot(target_data["ekf_x"], target_data["ekf_y"], color="blue", linewidth=2.5, label="EKF", linestyle="dashed")
    plt.plot(target_data["ukf_x"], target_data["ukf_y"], color="green", linewidth=2.5, label="UKF", linestyle="dashdot")
    plt.xlabel(r"x[m]", fontsize=40)
    plt.ylabel(r"y[m]", fontsize=40)
    plt.legend(fontsize=25)
    plt.title("Estimation Result")

    plt.savefig(path + "/result/picture/simulation_result/simulation_result_trajectory.eps", bbox_inches="tight", pad_inches=0.05)
    plt.show()