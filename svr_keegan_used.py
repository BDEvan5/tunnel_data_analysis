import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error


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
    
    
# Load data
train = pd.read_csv(r"training.csv")
test = pd.read_csv(r"TestSet.csv")

# Normalize data
scalerx = MinMaxScaler()
scalery = MinMaxScaler()
time_min = test['Minute']
actual_res = test['Actual Temperature Middle ((T[n])']
actual_res = np.array(actual_res)

X_train = scalerx.fit_transform(train[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off']])
train_y = scalery.fit_transform(train[['Actual Temperature Middle ((T[n])']])[:, 0]

X_test = scalerx.transform(test[['Solar Radiation (W/m^2)','Outside Temperature (T[n-1])', 'Inside Temperature Middle (T[n-1])', 'Fan on/off']])
test_y = scalery.transform(test[['Actual Temperature Middle ((T[n])']])[:, 0]

svr_regr = SVR(kernel="rbf", epsilon=0.02894736842105263, C=7.63157894736842, gamma=1.3526315789473684)    # 1 day simulation with time (BEST)
svr_regr.fit(X_train, train_y)

# testing
test_predict = svr_regr.predict(X_test)
test_predict = scalery.inverse_transform(np.array(test_predict).reshape(1,-1))
predict_score = svr_regr.score(X_test, test_y[0:len(test_y)])

prev_fan = 0
inside_temp_sim = []
x_test = X_test
input_sim = X_test[0,:]      
r2_list = []
param_list= []
inside_temp_sim.append(scalery.inverse_transform(test_y[0].reshape(1,-1)))
prev_delta_temp = 0
prev_sol = 0
prev_fan = 0
prev_inside_temp = 0
prev_outside_temp = 0
prev_time = 0
datapoint_no = 285

r2_hour_sim = 0
rmse_hour_sim = 0
pred_vec = []
count_col = 0
count_row = 0
def exponential_smoothing(data, alpha):
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[i-1])
    return smoothed
temp_array = []
error_array = np.zeros((len(time_min)-12,12))

j = 0
temp_pred = []
plot_array = []

# SIMULATE HOUR AHEAD EVERY 5 MINUTE INTERVAL
fan_pred = []

while len(time_min)-j > 12:
    pred_vec = []
    input_sim = X_test[j,:]
    
    for i in range(0,12):
        
        temp_fan = 0
        input_sim = input_sim.reshape(1,-1)
                               
        pred_vec.append(scalery.inverse_transform(svr_regr.predict(input_sim).reshape(1,-1)))
        temp_pred.append(scalery.inverse_transform(svr_regr.predict(input_sim).reshape(1,-1)))
        
        
        error_array[j][i] = math.sqrt((actual_res[j+i] - pred_vec[i])**2)
        
        
        if i == 11:  
            plot_array.append(pred_vec[i])

        input_sim = []
        if i == 11:
            fan_pred.append(prev_fan*5)
        temp_variable = np.array([X_test[j+i,0], X_test[j+i,1], scalery.transform(pred_vec[i-1].reshape(-1,1)), prev_fan], dtype='object')
    
        
        prev_sol = temp_variable[0]
        prev_outside_temp = temp_variable[1]
        prev_inside_temp = temp_variable[2]
        prev_fan = temp_variable[3]
        
        input_sim.append(prev_sol)
        input_sim.append(prev_outside_temp)
        input_sim.append(prev_inside_temp)
        input_sim.append(prev_fan)
        input_sim = np.array(input_sim, dtype='object')
        input_sim = input_sim.reshape(1,-1)
    
    
        temp_temp = scalery.inverse_transform(prev_inside_temp.reshape(-1,1))
        if prev_fan == 1 and temp_temp > 22:
            temp_fan = 1
        elif prev_fan == 1 and temp_temp <= 22:
            temp_fan = 0

        if prev_fan == 0 and temp_temp < 30:
            temp_fan = 0
        elif prev_fan == 0 and temp_temp >= 30:
            temp_fan = 1
        prev_fan = temp_fan
        
        count_col += 1
    j += 1

x_labels = np.arange(0,datapoint_no,1)
inside_temp_sim = np.array(inside_temp_sim).flatten()
predictions_svr = inside_temp_sim

# 12 AHEAD EVERY 5 MInutes

prev_fan = prev_fan*5
plot_array = np.array(plot_array).flatten()
svr_r2_sim = round(r2_score(actual_res[0:len(time_min)-12], plot_array),2)
svr_mse_sim = round(math.sqrt(mean_squared_error(actual_res[0:len(time_min)-12], plot_array)),2)

predict_mae = mean_absolute_error(actual_res[0:len(time_min)-12], plot_array)
print("MAE=" + str(predict_mae))
MBE(actual_res[0:len(time_min)-12], predictions_svr)
print("R2=" + str(svr_r2_sim))
print("RMSE: " + str(svr_mse_sim))

x_labels = np.arange(0,len(time_min),1)

plt.plot(x_labels[0:len(time_min)-12],actual_res[0:len(time_min)-12], color='blue', alpha = 0.8)
plt.plot(x_labels[0:len(time_min)-12],plot_array, color='red', alpha=0.7)
plt.plot(x_labels[0:len(time_min)-12], X_test[:len(time_min)-12,3]*5, color='black', alpha=0.8)
plt.plot(x_labels[0:len(time_min)-12], fan_pred, color='green')

plt.xlabel("Time (5-minute Intervals)", fontsize=15)
plt.ylabel("Temperature (deg. C)", fontsize=15)
plt.ylim(0, 37)
plt.legend(['Actual Temperature', 'Time Ahead Prediction', 'Actual Fan State', 'Simulated Fan State'], loc='lower left')
plt.xticks(rotation=90)
plt.grid(False)                  
plt.show()


fig, ax = plt.subplots()
bp = ax.boxplot(error_array, showfliers=False, medianprops={'color': 'red'})
ax.set_xticklabels(np.arrange(5, 5, 60))
# ax.set_xticklabels(['5 min', '10 min', '15 min', '20 min', '25 min', '30 min', '35 min', '40 min', '45 min', '50 min', '55 min', '60 min'])
ax.set_ylabel('Temperature')
ax.set_xlabel('Time Step Ahead (min)')
ax.set_title('5-min Ahead Predictions')

plt.show()