import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

# from keras.models import Sequential
# from keras.layers import Dense, LSTM, GRU

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
    print('MBE = ', mbe)
    
def load_train_data(filename):
    data = pd.read_csv(filename)
    scalerx = MinMaxScaler()
    scalery = MinMaxScaler()
    
    x = scalerx.fit_transform(data[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off']])
    y = scalery.fit_transform(data[['Actual Temperature Middle ((T[n])']])[:, 0]
    
    return x, y

def load_test_data(filename):
    data = pd.read_csv(filename)
    scalerx = MinMaxScaler()
    scalery = MinMaxScaler()
    
    x_data = data[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off']].to_numpy()
    x = scalerx.fit_transform(x_data)
    y_data = data[['Actual Temperature Middle ((T[n])']].to_numpy()
    y = scalery.fit_transform(y_data)[:, 0]
        
    return x, y, scalery


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x = x.view(-1, 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

def select_mini_batch(x_data, y_data, batch_size = 100):
    num_samples = x_data.shape[0]
    batch_indices = np.random.choice(num_samples, batch_size, replace=False)
    # print(batch_indices[0:8])

    x = x_data[batch_indices]
    y = y_data[batch_indices]

    return x, y

def test_nn_model_r2(model, x, y):
    x_tensor = torch.from_numpy(x).float()
    predictions = model(x_tensor).detach()
    loss = (predictions.numpy() - y) **2
    print(f"TEST --> Mean loss: {np.mean(loss):.4f}, std loss: {np.std(loss):.4f}")


def train_nn_model():
    X_train, train_y = load_train_data("Data/training.csv")
    X_test, test_y, scalery = load_test_data("Data/TestSet.csv")
    
    model = MyNet()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    losses = []
    
    # for i in range(20):
    for i in range(1000):
        x, y = select_mini_batch(X_train, train_y, 1000)
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float()
        outputs = model(x_tensor)
        
        out_np = outputs.clone().detach().numpy()
        loss = (out_np - y) **2
        print(f"Mean loss: {np.mean(loss):.4f}, std loss: {np.std(loss):.4f}")
        test_nn_model_r2(model, X_test, test_y)
        
        loss = criterion(outputs, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        print(f"Epoch {i+1} - loss: {loss.item():.4f}")

        losses.append(loss.item())

    plt.plot(losses)
    plt.grid(True)
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.savefig('DevelImgs/train_losses_rnn1.svg')

    return model

import torch

def r2_score(y_pred, y_true):
    """
    Computes R^2 score between y_true and y_pred.

    Args:
        y_pred (torch.Tensor): predicted values of shape (batch_size, 1)
        y_true (torch.Tensor): true values of shape (batch_size, 1)

    Returns:
        torch.Tensor: R^2 score
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_result = np.sum((y_pred - y_true) ** 2)
    r2 = 1 - ss_result / ss_total
    
    # ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    # ss_res = torch.sum((y_true - y_pred) ** 2)
    # r2 = 1 - ss_res / ss_tot
    return r2


def test_nn_model(model):
    X_test, test_y, scalery = load_test_data("Data/TestSet.csv")
    true_temperatures = scalery.inverse_transform(test_y.reshape(-1, 1))

    
    return X_test, true_temperatures, scalery


def run_simulation_benjamin(model, scalery, X_test):
    N = len(X_test)
    
    predicted_tempteratures = np.zeros((N-12,12))
    fan_predictions = np.zeros(N-12)
    tempterature_errors = np.zeros((N-12,12))
    
    for i in range(N-12): # steps through each element in test data
        input_sim = X_test[i,:].reshape(1,-1)
        
        for j in range(12): # predicts 12x5 minute steps ahead
            input_sim_tensor = torch.from_numpy(input_sim).float()
            predicted_scaled_temp = model(input_sim_tensor).detach().numpy()
            predicted_tempterature = scalery.inverse_transform(predicted_scaled_temp.reshape(1,-1))
            predicted_tempteratures[i,j] = predicted_tempterature
            tempterature_errors[i, j] = math.sqrt((true_temperatures[i+j] - predicted_tempterature)**2)
    
            fan_setting = fan_logic(input_sim[0, 3], predicted_tempterature)
            input_sim = np.array([X_test[i+j+1,0], X_test[i+j+1,1], predicted_scaled_temp[0, 0], fan_setting]).reshape(1,-1)
        
        fan_predictions[i] = fan_setting * 5
    
    predicted_internal_temp = predicted_tempteratures[:, 11]
    
    return predicted_internal_temp, tempterature_errors, fan_predictions

    
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

def calculate_metrics(predicted_temperatures, true_temperatures):
    N = len(true_temperatures)
    svr_r2_sim = round(r2_score(true_temperatures[0:N-12], predicted_temperatures),2)
    svr_mse_sim = round(math.sqrt(mean_squared_error(true_temperatures[0:N-12], predicted_temperatures)),2)

    predict_mae = mean_absolute_error(true_temperatures[0:N-12], predicted_temperatures)
    print("MAE=" + str(predict_mae))
    MBE(true_temperatures[0:N-12], predicted_temperatures)
    print("R2=" + str(svr_r2_sim))
    print("RMSE: " + str(svr_mse_sim))
    

def plot_temperatrues(true_temperatures, predicted_temperatures, X_test, fan_pred):
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
    plt.savefig('Imgs/lstm_result_1')    
    plt.show()

def plot_prediction_errors(error_array):
    plt.figure()
    plt.boxplot(error_array, showfliers=False, medianprops={'color': 'red'})
    plt.xticks(np.arange(0, 12, 1), np.arange(0, 60, 5))
    plt.ylabel('Temperature')
    plt.xlabel('Time Step Ahead (min)')
    plt.title('5-min Ahead Predictions')
    plt.savefig('Imgs/lstm_result_2')    

    plt.show()

    
if __name__ == "__main__":
    
    model = train_nn_model()
    X_test, true_temperatures, scalery = test_nn_model(model)
    
    
    # svr = train_svr_model()
    # X_test, true_temperatures, scalery = test_svr_model(svr)
    predicted_temperatures, error_array, fan_pred = run_simulation_benjamin(model, scalery, X_test)
    calculate_metrics(predicted_temperatures, true_temperatures)
    plot_temperatrues(true_temperatures, predicted_temperatures, X_test, fan_pred)
    plot_prediction_errors(error_array)
    
    
    