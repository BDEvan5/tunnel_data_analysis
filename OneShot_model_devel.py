from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import os, yaml

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from numba import njit

from utils import *

torch.use_deterministic_algorithms(True)
torch.manual_seed(101)
np.random.seed(101)

#HYPERPARAMETERS

# LAYER_SIZE = 500
LAYER_SIZE = 256  


def modify_data(x, y):
    N = len(x) - 24
    N = N - (N%12)
    # N = 1000 #? debugging term
    print(f"X shape: {x.shape}")
    xp, yp = np.zeros((N, 72)), np.zeros((N, 1))
    for i in range(N):
        dx1 = x[i:(i+12), :]
        dx2 = x[(i+12):(i+24), 0:2]
        dx = np.hstack([dx1, dx2])
        xp[i] = dx.reshape(72)

        dy = y[(i+12)]
        yp[i] = dy
    
    return xp, yp



class MyNet(nn.Module):
    def __init__(self, input, output):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input, LAYER_SIZE)
        self.fc2 = nn.Linear(LAYER_SIZE, LAYER_SIZE)
        self.fc_out = nn.Linear(LAYER_SIZE, output)
        
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
        targets_tensor = torch.from_numpy(targets).float()
        
        y_predicted = self.forward(x_tensor)
        loss = self.loss_function(y_predicted, targets_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return y_predicted.detach().numpy(), loss.item()
        


def train_nn_model(x_train, y_train, x_test, y_test, run_data):
    model = MyNet(72, 1)
    
    train_losses, test_losses = [], []
    
    for i in range(run_data.epochs):
        x, y = select_mini_batch(x_train, y_train, run_data.batch_size)
        
        outputs, loss = model.train_model(x, y)
        # plot_data_tuple(x, y, outputs, 0)
    
        print(f"Epoch {i+1} - loss: {loss:.6f}")
        train_losses.append(loss)
        test_loss = test_nn_model_loss(model, x_test, y_test)
        test_losses.append(test_loss)

    plt.figure(2)
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

def run_simulation_LN(model, scalery, x_test, true_temperatures):
    N = len(x_test)
    
    outputs = model.test_model(x_test) # (N, 12)
    predicted_temperatures = scalery.inverse_transform(outputs) # (N, 12)

    predicted_internal_temp = predicted_temperatures[:, -1]
    tempterature_errors = np.abs(true_temperatures - predicted_temperatures)

    fan_predictions = np.ones(N)
    
    return predicted_internal_temp, tempterature_errors, fan_predictions


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


def plot_simulation(model, scalery, x, true_temperatures):
    N = len(x)
    outputs = model.test_model(x) # (N, 12)
    predicted_temperatures = scalery.inverse_transform(outputs) # (N, 12)

    plt.figure(1)
    for i in range(N):
        plt.clf()
        internal_temp = x[i, :].reshape(12, 6)[:, 2]
        internal_temp = scalery.inverse_transform(internal_temp.reshape(-1, 1))
        plt.plot(np.arange(12), internal_temp, label="InputX")
        plt.plot(np.arange(12, 24), true_temperatures[i], label="True Temp")
        plt.plot(np.arange(12, 24), predicted_temperatures[i], label="Prediction")

        plt.legend()
        plt.pause(0.1)

    

def plot_temperatrues(true_temperatures, predicted_temperatures, X_test, fan_pred, run_data, mode):
    N = len(true_temperatures)

    x_labels = np.arange(N)

    plt.plot(x_labels, true_temperatures, color='blue', alpha = 0.8, label='Actual Temperature')
    plt.plot(x_labels+1, predicted_temperatures, color='red', alpha=0.7, label='Predicted Temperature (1 hour ahead)')

    plt.xlabel("Time (5-minute Intervals)", fontsize=15)
    plt.ylabel("Temperature (deg. C)", fontsize=15)
    plt.ylim(8, 40)
    # plt.ylim(0, 37)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.xticks(rotation=90)
    plt.grid(False)              
    plt.savefig(f'{run_data.path}nn_result_{mode}_1.svg')    
    plt.savefig(f'{run_data.path}nn_result_{mode}_1.pdf')    

def plot_prediction_errors(error_array, run_data, mode):
    plt.figure()
    plt.boxplot(error_array, showfliers=False, medianprops={'color': 'red'})
    plt.xticks(np.arange(0, 12, 1), np.arange(0, 60, 5))
    plt.ylabel('Temperature')
    plt.xlabel('Time Step Ahead (min)')
    plt.title('5-min Ahead Predictions')
    plt.savefig(f'{run_data.path}nn_result_{mode}_2.svg')    
    plt.savefig(f'{run_data.path}nn_result_{mode}_2.pdf')    


def plot_data_tuple(x, y, predict_y, i):
    """
    Args:
        x (ndarray(72)): x data
        y ndarray(12): temperatures
    """
    x = x[i, :].reshape(12, 6)

    internal_temp = x[:, 2]
    
    # solar_radiation = x[i, 0:12]
    # outside_temp = x[i, 12:24]
    # internal_temp = x[i, 24:36]
    # fan_mode = x[i, 36:48]
    # y_solar = x[i, 48:60]
    # y_outside = x[i, 60:72]

    y_inside_temp = y[i, :]
    predict_y = predict_y[i, :]

    plt.figure(1)
    plt.clf()
    # print
    plt.plot(np.arange(12), internal_temp, label="InputX")
    plt.plot(np.arange(12, 24), y_inside_temp, label="True Temp")
    plt.plot(np.arange(12, 24), predict_y, label="Prediction")
    # plt.plot(np.arange(12), outside_temp, label="OutX")

    plt.legend()

    # plt.show()
    plt.pause(0.1)



if __name__ == "__main__":
    x_train, y_train, x_test, y_test, scalery = load_data("Data/training.csv", "Data/TestSet.csv")
    
    run_data = {"name": "DevelLN",
                "batch_size": 80,
                # "epochs": 10}
                "epochs": 60}
    
    run_data["path"] = "RunData/" + run_data["name"] + "/"
    if not os.path.exists(run_data["path"]):
        os.mkdir(run_data["path"])
    
    with open(run_data["path"] + "run_data.yaml", "w") as f:
        yaml.dump(run_data, f)
    run_data = Namespace(**run_data)
    
    x_train, y_train = modify_data(x_train, y_train)
    x_test, y_test = modify_data(x_test, y_test)

    model = train_nn_model(x_train, y_train, x_test, y_test, run_data)
    
    true_train_temperatures = scalery.inverse_transform(y_train)
    predicted_temperatures, error_array, fan_pred = run_simulation_LN(model, scalery, x_train, true_train_temperatures)
    # plot_simulation(model, scalery, x_train, true_train_temperatures)
    
    print_metric_file(run_data, predicted_temperatures, true_train_temperatures[:, -1], "Training")
    plot_temperatrues(true_train_temperatures[:, -1], predicted_temperatures, x_test, fan_pred, run_data, "Training")
    plot_prediction_errors(error_array, run_data, "Training")
    
    
    true_temperatures = scalery.inverse_transform(y_test)
    predicted_temperatures, error_array, fan_pred = run_simulation_LN(model, scalery, x_test, true_temperatures)
    
    print_metric_file(run_data, predicted_temperatures, true_temperatures[:, -1], "Testing")
    plot_temperatrues(true_temperatures[:, -1], predicted_temperatures, x_test, fan_pred, run_data, "Testing")
    plot_prediction_errors(error_array, run_data, "Testing")
    
    