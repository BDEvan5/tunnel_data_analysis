import numpy as np 
import pandas as pd
import torch
from numba import njit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math

def MBE(y_true, y_pred):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Biais score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_true = y_true.reshape(len(y_true),1)
    
    y_pred = y_pred.reshape(-1, 1)   
    diff = (y_true-y_pred)
    mbe = diff.mean()
    
    return mbe
    # print('MBE = ', mbe)
   


def calculate_metrics(predicted_temperatures, true_temperatures):
    N = len(true_temperatures)
    svr_r2_sim = round(r2_score(true_temperatures, predicted_temperatures),2)
    svr_mse_sim = round(math.sqrt(mean_squared_error(true_temperatures, predicted_temperatures)),2)

    predict_mae = mean_absolute_error(true_temperatures, predicted_temperatures)
    print("MAE=" + str(predict_mae))
    MBE(true_temperatures, predicted_temperatures)
    print("R2=" + str(svr_r2_sim))
    print("RMSE: " + str(svr_mse_sim)) 

def print_metric_file(run_data, predicted_temperatures, true_temperatures, print_data=True):
    N = len(true_temperatures)
    svr_r2_sim = round(r2_score(true_temperatures, predicted_temperatures),2)
    svr_mse_sim = round(math.sqrt(mean_squared_error(true_temperatures, predicted_temperatures)),2)

    predict_mae = mean_absolute_error(true_temperatures, predicted_temperatures)
    mbe = MBE(true_temperatures, predicted_temperatures)
    
    if print_data:
        print("Metrics")
        print(f"RMSE: {svr_mse_sim}")
        print(f"MAE: {predict_mae}")
        print(f"MBE: {mbe}")
        print(f"R2: {svr_r2_sim}")
        print(f"--------------------------------------\n")

    with open(run_data.path + "Metrics.txt", 'w') as f:
        f.write("Metrics\n")
        f.write(f"RMSE: {svr_mse_sim}\n")
        f.write(f"MAE: {predict_mae}\n")
        f.write(f"MBE: {mbe}\n")
        f.write(f"R2: {svr_r2_sim}\n")
        f.write(f"--------------------------------------\n")

    
def load_data(train_data, test_data):
    data = pd.read_csv(train_data)
    scalerx = MinMaxScaler()
    scalery = MinMaxScaler()
    
    x_train = scalerx.fit_transform(data[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off']].to_numpy())
    y_train = scalery.fit_transform(data[['Actual Temperature Middle ((T[n])']].to_numpy())[:, 0]

    data = pd.read_csv(test_data)
    
    x_data = data[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off']].to_numpy()
    x_test = scalerx.transform(x_data)
    y_data = data[['Actual Temperature Middle ((T[n])']].to_numpy()
    y_test = scalery.transform(y_data)[:, 0]
        
    return x_train, y_train, x_test, y_test, scalery
    
def select_mini_batch(x_data, y_data, batch_size = 100):
    num_samples = x_data.shape[0]
    batch_indices = np.random.choice(num_samples, batch_size, replace=False)

    x = x_data[batch_indices]
    y = y_data[batch_indices]

    return x, y


def test_nn_model_loss(model, x, y):
    x_tensor = torch.from_numpy(x).float()
    predictions = model(x_tensor).detach()
    loss = (predictions.numpy() - y[:, None]) **2
    mean_loss = np.mean(loss)
    print(f"TEST --> Mean loss: {mean_loss:.6f}, std loss: {np.std(loss):.4f}")
    
    return np.mean(loss)

@njit(cache=True)
def fan_logic_array(previous_fans, temperatures):
    new_fans = np.zeros(len(previous_fans))
    for i in range(len(previous_fans)):
        new_fans[i] = fan_logic(previous_fans[i], temperatures[i])
        
    return new_fans

@njit(cache=True)
def fan_logic(previous_fan, temperature):
    if previous_fan == 1 and temperature > 22:
        new_fan_setting = 1
    elif previous_fan == 1 and temperature <= 22:
        new_fan_setting = 0

    if previous_fan == 0 and temperature < 30:
        new_fan_setting = 0
    elif previous_fan == 0 and temperature >= 30:
        new_fan_setting = 1
            
    return new_fan_setting