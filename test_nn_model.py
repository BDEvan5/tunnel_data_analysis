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
from numba import njit

from utils import *

torch.use_deterministic_algorithms(True)
torch.manual_seed(101)
np.random.seed(101)

    
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

# LAYER_SIZE = 500
LAYER_SIZE = 256 

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(4, LAYER_SIZE)
        self.fc2 = nn.Linear(LAYER_SIZE, LAYER_SIZE)
        self.fc_out = nn.Linear(LAYER_SIZE, 1)
        
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc_out(x))
        
        return x
    
    def test_model(self, x):
        x = torch.from_numpy(x).float()
        y = self.forward(x)
        
        return y.detach().numpy()
        
    def train_model(self, x, targets):
        x_tensor = torch.from_numpy(x).float()
        targets_tensor = torch.from_numpy(targets[:, None]).float()
        
        y_predicted = self.forward(x_tensor)
        loss = self.loss_function(y_predicted, targets_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return y_predicted.detach().numpy(), loss.item()
        


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


def train_nn_model(x_train, y_train, x_test, y_test, epochs):
    model = MyNet()
    
    train_losses, test_losses = [], []
    
    for i in range(epochs):
        x, y = select_mini_batch(x_train, y_train, 200)
        
        outputs, loss = model.train_model(x, y)
    
        print(f"Epoch {i+1} - loss: {loss:.6f}")
        train_losses.append(loss)
        test_loss = test_nn_model_loss(model, x_test, y_test)
        test_losses.append(test_loss)

    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.grid(True)
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig('DevelImgs/train_losses_dnn.svg')

    return model

def run_simulation_torch(model, scalery, x_test):
    N = len(x_test)
    
    predicted_tempteratures = np.zeros((N-12,12)) 
    fan_predictions = np.zeros(N-12)
    tempterature_errors = np.zeros((N-12,12))
    
    model_inputs = x_test[:N-12, :] # (N-12, 4)
    for i in range(12):
        outputs = model.test_model(model_inputs) # (N-12, 1)
        predicted_temperature_set = scalery.inverse_transform(outputs)[:, 0] # (N-12)
        tempterature_errors[:, i] = np.sqrt((true_temperatures[i:N-(12-i), 0] - predicted_temperature_set)**2)
        predicted_tempteratures[:, i] = predicted_temperature_set
        
        fan_modes = fan_logic_array(model_inputs[:, 3], predicted_temperature_set)
        model_inputs = x_test[i+1:N-(11-i), :]
        model_inputs[:, 2] = outputs[:, 0]
        model_inputs[:, 3] = fan_modes
        
    fan_predictions = fan_modes*5 # last prediction.
    predicted_internal_temp = predicted_temperature_set # last prediction.
    
    return predicted_internal_temp, tempterature_errors, fan_predictions

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
    plt.savefig('DevelImgs/nn_result_1.svg')    
    plt.show()

def plot_prediction_errors(error_array):
    plt.figure()
    plt.boxplot(error_array, showfliers=False, medianprops={'color': 'red'})
    plt.xticks(np.arange(0, 12, 1), np.arange(0, 60, 5))
    plt.ylabel('Temperature')
    plt.xlabel('Time Step Ahead (min)')
    plt.title('5-min Ahead Predictions')
    plt.savefig('DevelImgs/nn_result_2.svg')    

    plt.show()

    
if __name__ == "__main__":
    x_train, y_train, x_test, y_test, scalery = load_data("Data/training.csv", "Data/TestSet.csv")
    model = train_nn_model(x_train, y_train, x_test, y_test, 200)
    
    true_temperatures = scalery.inverse_transform(y_test.reshape(-1, 1))
    #! true temperatures are not being passes here...
    predicted_temperatures, error_array, fan_pred = run_simulation_torch(model, scalery, x_test)
    
    calculate_metrics(predicted_temperatures, true_temperatures)
    plot_temperatrues(true_temperatures, predicted_temperatures, x_test, fan_pred)
    plot_prediction_errors(error_array)
    
    
    