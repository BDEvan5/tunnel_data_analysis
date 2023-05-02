import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import math

import os
import yaml 
from argparse import Namespace
from utils import *


def train_svr_model(x_train, y_train):
    svr_regr = SVR(kernel="rbf", epsilon=0.02894736842105263, C=7.63157894736842, gamma=1.3526315789473684)    # 1 day simulation with time (BEST)
    svr_regr.fit(x_train, y_train)
    
    return svr_regr


def run_simulation_benjamin(svr, scalery, x_test, true_temperatures):
    N = len(x_test)
    
    predicted_tempteratures = np.zeros((N-12,12))
    fan_predictions = np.zeros(N-12)
    tempterature_errors = np.zeros((N-12,12))
    
    for i in range(N-12): # steps through each element in test data
        input_sim = x_test[i,:].reshape(1,-1)
        
        for j in range(12): # predicts 12x5 minute steps ahead
            predicted_scaled_temp = svr.predict(input_sim)
            predicted_tempterature = scalery.inverse_transform(predicted_scaled_temp.reshape(1,-1))
            predicted_tempteratures[i,j] = predicted_tempterature
            tempterature_errors[i, j] = math.sqrt((true_temperatures[i+j] - predicted_tempterature)**2)
    
            fan_setting = fan_logic(input_sim[0, 3], predicted_tempterature)
            input_sim = np.array([x_test[i+j+1,0], x_test[i+j+1,1], predicted_scaled_temp[0], fan_setting]).reshape(1,-1)
        
        fan_predictions[i] = fan_setting * 5
    
    predicted_internal_temp = predicted_tempteratures[:, 11]
    
    return predicted_internal_temp, tempterature_errors, fan_predictions


def plot_temperatrues(true_temperatures, predicted_temperatures, X_test, fan_pred, run_data, mode="Testing"):
    N = len(true_temperatures)

    x_labels = np.arange(0,N,1)

    plt.plot(x_labels[0:N-12], true_temperatures[0:N-12], color='blue', alpha = 0.8, label='Actual Temperature')
    plt.plot(x_labels[0:N-12], predicted_temperatures, color='red', alpha=0.7, label='Predicted Temperature (1 hour ahead)')
    plt.plot(x_labels[0:N-12], X_test[:N-12,3]*5, color='black', alpha=0.8, label='True Fan State')
    plt.plot(x_labels[0:N-12], np.array(fan_pred)*0.8, color='green', label='Simulated Fan State')

    plt.xlabel("Time (5-minute Intervals)", fontsize=15)
    plt.ylabel("Temperature (deg. C)", fontsize=15)
    plt.ylim(0, 37)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.xticks(rotation=90)
    plt.grid(False)              
    plt.savefig(f'{run_data.path}nn_result_{mode}_1.svg')    
    plt.savefig(f'{run_data.path}nn_result_{mode}_1.pdf')   

def plot_prediction_errors(error_array, run_data, mode="Testing"):
    plt.figure()
    plt.boxplot(error_array, showfliers=False, medianprops={'color': 'red'})
    plt.xticks(np.arange(0, 12, 1), np.arange(0, 60, 5))
    plt.ylabel('Temperature')
    plt.xlabel('Time Step Ahead (min)')
    plt.title('5-min Ahead Predictions')
    plt.savefig(f'{run_data.path}Result_{mode}_2.svg')    
    plt.savefig(f'{run_data.path}Result_{mode}_2.pdf')    


    
if __name__ == "__main__":
    x_train, y_train, x_test, y_test, scalery = load_data("Data/training.csv", "Data/TestSet.csv")
    
    run_data = {"name": "StdSVR"}
    
    run_data["path"] = "RunData/" + run_data["name"] + "/"
    if not os.path.exists(run_data["path"]):
        os.mkdir(run_data["path"])
    
    with open(run_data["path"] + "run_data.yaml", "w") as f:
        yaml.dump(run_data, f)
    run_data = Namespace(**run_data)
    
    svr = train_svr_model(x_train, y_train)
    
    true_temperatures = scalery.inverse_transform(y_test.reshape(-1,1))
    predicted_temperatures, error_array, fan_pred = run_simulation_benjamin(svr, scalery, x_test, true_temperatures)
    print_metric_file(run_data, predicted_temperatures, true_temperatures[:-12])
    plot_temperatrues(true_temperatures, predicted_temperatures, x_test, fan_pred, run_data)
    plot_prediction_errors(error_array, run_data)
    
    
    