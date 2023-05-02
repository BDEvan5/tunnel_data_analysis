from argparse import Namespace
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
import os, yaml

torch.use_deterministic_algorithms(True)
torch.manual_seed(101)
np.random.seed(101)

    

# LAYER_SIZE = 500
LAYER_SIZE = 256 
# LAYER_SIZE = 100

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
        


def train_nn_model(x_train, y_train, x_test, y_test, run_data):
    model = MyNet()
    
    train_losses, test_losses = [], []
    
    for i in range(run_data.epochs):
        x, y = select_mini_batch(x_train, y_train, run_data.batch_size)
        
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
    plt.savefig(f'{run_data.path}train_losses_dnn.svg')
    plt.savefig(f'{run_data.path}train_losses_dnn.pdf')

    return model

def run_simulation_torch(model, scalery, x_test, true_temperatures):
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
    plt.savefig(f'{run_data.path}nn_result_{mode}_2.svg')    
    plt.savefig(f'{run_data.path}nn_result_{mode}_2.pdf')  


    
if __name__ == "__main__":
    x_train, y_train, x_test, y_test, scalery = load_data("Data/training.csv", "Data/TestSet.csv")
    run_data = {"name": "StdNN",
                "batch_size": 100,
                "epochs": 200}
    
    run_data["path"] = "RunData/" + run_data["name"] + "/"
    if not os.path.exists(run_data["path"]):
        os.mkdir(run_data["path"])
    
    with open(run_data["path"] + "run_data.yaml", "w") as f:
        yaml.dump(run_data, f)
    run_data = Namespace(**run_data)
    
    model = train_nn_model(x_train, y_train, x_test, y_test, run_data)
    
    true_temperatures = scalery.inverse_transform(y_train.reshape(-1, 1))
    predicted_temperatures, error_array, fan_pred = run_simulation_torch(model, scalery, x_train, true_temperatures)
    
    print_metric_file(run_data, predicted_temperatures, true_temperatures[:-12], mode="Training")
    plot_temperatrues(true_temperatures, predicted_temperatures, x_train, fan_pred, run_data, mode="Training")
    plot_prediction_errors(error_array, run_data, mode="Training")
    
    
    true_temperatures = scalery.inverse_transform(y_test.reshape(-1, 1))
    predicted_temperatures, error_array, fan_pred = run_simulation_torch(model, scalery, x_test, true_temperatures)
    
    print_metric_file(run_data, predicted_temperatures, true_temperatures[:-12], mode="Testing")
    plot_temperatrues(true_temperatures, predicted_temperatures, x_test, fan_pred, run_data, mode="Testing")
    plot_prediction_errors(error_array, run_data, mode="Testing")
    
    
    