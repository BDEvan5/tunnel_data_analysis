import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv("training.csv")

# Select relevant features
data = data[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off', 'Actual Temperature Middle ((T[n])']]
data_test = pd.read_csv("TestSet.csv")
data_test = data_test[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off', 'Actual Temperature Middle ((T[n])']]

# Scale the data
scaler = MinMaxScaler()
data_scaled_train = scaler.fit_transform(data)
data_scaled_test = scaler.transform(data_test)

# Split into training and testing sets
train_size = int(len(data_scaled_train))
test_size = len(data_scaled_test)
# train, test = data_scaled[0:train_size,:], data_scaled[train_size:len(data_scaled),:]

train = data_scaled_train
test = data_scaled_test

# Define function to create X and Y for time series dataset
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :4]
        X.append(a)
        Y.append(dataset[i + look_back, 4])
    return np.array(X), np.array(Y)

# Reshape data into time series format
look_back = 10 # number of previous time steps to use as input features
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)


model = Sequential()
# model.add(Dense(50, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(GRU(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add()
# model.add(Dense(100))
model.add(Dense(1))
# model.compile(loss='mean_absolute_error', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam')
# 
# history = model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test), verbose=2, shuffle=False)
history = model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test), verbose=2, shuffle=False)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert scaling to get actual temperature values
train_predict = scaler.inverse_transform(np.hstack((X_train[:, -1, :].reshape(-1, 4), train_predict)))
print(train_predict.shape)
Y_train = scaler.inverse_transform(np.hstack((X_train[:, -1, :].reshape(-1, 4), Y_train.reshape(-1, 1))))
test_predict = scaler.inverse_transform(np.hstack((X_test[:, -1, :].reshape(-1, 4), test_predict)))
Y_test = scaler.inverse_transform(np.hstack((X_test[:, -1, :].reshape(-1, 4), Y_test.reshape(-1, 1))))

# train_predict_it = scaler.inverse_transform(train_predict)
# Y_train_it = scaler.inverse_transform(Y_train.reshape(-1, 1))
# test_predict_it = scaler.inverse_transform(test_predict)
# Y_test_it = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Calculate mean squared error
train_score = mean_squared_error(Y_train[:, -1], train_predict[:, -1])
test_score = mean_squared_error(Y_test[:, -1], test_predict[:, -1])
print(f'Train MSE: {train_score} --> Train RMSE: {np.sqrt(train_score)}')
print(f'Test MSE: {test_score} --> Test RMSE: {np.sqrt(test_score)}')

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

train_score = MBE(Y_train[:, -1], train_predict[:, -1])
test_score = MBE(Y_test[:, -1], test_predict[:, -1])
print('Train MBE:', train_score)
print('Test MBE:', test_score)

train_score = mean_absolute_error(Y_train[:, -1], train_predict[:, -1])
test_score = mean_absolute_error(Y_test[:, -1], test_predict[:, -1])
print('Train MAE:', train_score)
print('Test MAE:', test_score)

train_score = r2_score(Y_train[:, -1], train_predict[:, -1])
test_score = r2_score(Y_test[:, -1], test_predict[:, -1])
print('Train R2:', train_score)
print('Test R2:', test_score)

# Use the model to predict the temperature of the wind tunnel at 5-minute intervals
# X_predict = data_scaled[-look_back:, :].reshape(1, look_back, 5)


