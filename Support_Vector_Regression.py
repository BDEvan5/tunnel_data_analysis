import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.utils import validation
import os
from numpy import absolute, savetxt, take_along_axis
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import math
import seaborn as sns
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
# from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
# from pykalman import KalmanFilter
fan_params = None
nofan_params = None


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
train = pd.read_csv(r"Data/training.csv")
test = pd.read_csv(r"Data/TestSet.csv")

# Normalize data
scalerx = MinMaxScaler()
scalery = MinMaxScaler()
time_min = test['Minute']
actual_res = test['Actual Temperature Middle ((T[n])']
actual_res = np.array(actual_res)

train_x = scalerx.fit_transform(train[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off']])
#train_x = scalerx.fit_transform(train[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])', 'Fan on/off']])
train_y = scalery.fit_transform(train[['Actual Temperature Middle ((T[n])']])
train_y = train_y.flatten()

test_x = scalerx.transform(test[['Solar Radiation (W/m^2)','Outside Temperature (T[n-1])', 'Inside Temperature Middle (T[n-1])', 'Fan on/off']])
#test_x = scalerx.transform(test[['Solar Radiation (W/m^2)','Outside Temperature (T[n-1])',  'Fan on/off']])
test_y = scalery.transform(test[['Actual Temperature Middle ((T[n])']])
test_y = test_y.flatten()

# reshape input to be [samples, time steps, features]
X_train = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
X_test = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
    

      
svr_regr = SVR(kernel="rbf", epsilon=0.02894736842105263, C=7.63157894736842, gamma=1.3526315789473684)    # 1 day simulation with time (BEST)
#svr_regr = SVR(kernel="rbf", epsilon=0.01, C=3, gamma = 0.5)           # No time as input variable
#svr_regr = SVR(kernel="rbf", epsilon=0.01, C=5, gamma = 0.9)           # For predictions with time
#svr_regr = SVR(kernel="rbf", epsilon=0.01, C=3, gamma=0.25555555554)   # 1 day simulation without time
#svr_regr = SVR(kernel="rbf", epsilon=0.02, C=10, gamma=1.5)            # 1 day simulation with time
#svr_regr = SVR(kernel="rbf", epsilon=0.01, C=6, gamma=1.344444444446)  # For 2 day simulation
#svr_regr = SVR(kernel="rbf", epsilon=0.03, C=1, gamma=1.5)    # 1 day simulation without time
#svr_regr = SVR(kernel="rbf", epsilon=0.08, C=1, gamma=0.72222222222222)    # 1 day simulation without time
#svr_regr = SVR(kernel="rbf", epsilon=0.04, C=3, gamma=1.344444)    # 1 day simulation without time
#svr_regr = SVR(kernel="rbf", epsilon=0.12, C=2, gamma=1.2)    # 1 day simulation without time
#svr_regr = SVR(kernel="linear", epsilon=0.01, C=2.8947368)    # 1 day simulation without time

#svr_regr = SVR(kernel="rbf", epsilon=0.1, C=6.8965517241379315, gamma=0.7827586206)    # No time or delta temperature, best results

       
svr_regr.fit(X_train, train_y)

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




# 10 AHEAD SIMULATION WITH HOURLY ACTUAL UPDATES 

'''
for i in range(0, len(time_min)):
    if time_min[i] == 5 or time_min[i] % 55 == 0:
        
        
        input_sim = X_test[i,:]
        pred_vec.append(scalery.inverse_transform(test_y[i].reshape(1,-1)))
        
        count_col = 0
        if i != 0:
            count_row += 1
    else:
        
        temp_fan = 0
        input_sim = input_sim.reshape(1,-1)
                               
        pred_vec.append(scalery.inverse_transform(svr_regr.predict(input_sim).reshape(1,-1)))
        
        
        error_array[count_row][count_col] = math.sqrt((actual_res[i] - pred_vec[i])**2)
        
        input_sim = []
            
        temp_variable = np.array([X_test[i,0], X_test[i,1], scalery.transform(pred_vec[i-1].reshape(-1,1)), prev_fan], dtype='object')
    
        #prev_time = temp_variable[0]
        prev_sol = temp_variable[0]
        prev_outside_temp = temp_variable[1]
        prev_inside_temp = temp_variable[2]
        prev_fan = temp_variable[3]
        #prev_delta_temp = temp_variable[4]
    
            
        #input_sim.append(prev_time)
        input_sim.append(prev_sol)
        input_sim.append(prev_outside_temp)
        input_sim.append(prev_inside_temp)
        input_sim.append(prev_fan)
        #input_sim.append(prev_delta_temp)
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
pred_vec = np.array(pred_vec).flatten().reshape(1,-1).flatten()

kf = KalmanFilter(transition_matrices=np.eye(1),
                  observation_matrices=np.eye(1),
                  transition_covariance=0.1,
                  observation_covariance=1,
                  initial_state_mean=pred_vec[0],
                  initial_state_covariance=1)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(pred_vec)


alpha = 0.5
smoothed = exponential_smoothing(pred_vec, alpha)
smoothed = np.array(smoothed)

r2_hour_sim = round(r2_score(actual_res.reshape(-1,1), pred_vec.reshape(-1,1)),2)
rmse_hour_sim = round(math.sqrt(mean_squared_error(actual_res.reshape(-1,1), pred_vec.reshape(-1,1))),2)
print("R2 - RMSE (No Smoothing)")
print("{:f} {:03f}".format(r2_hour_sim, rmse_hour_sim))

kal_r2_hour_sim = round(r2_score(actual_res.reshape(-1,1), smoothed_state_means.reshape(-1,1)),2)
kal_rmse_hour_sim = round(math.sqrt(mean_squared_error(actual_res.reshape(-1,1), smoothed_state_means.reshape(-1,1))),2)
print("R2 - RMSE (Kalman Smoothing)")
print("{:f} {:03f}".format(kal_r2_hour_sim, kal_rmse_hour_sim))

exp_r2_hour_sim = round(r2_score(actual_res.reshape(-1,1), smoothed.reshape(-1,1)),2)
exp_rmse_hour_sim = round(math.sqrt(mean_squared_error(actual_res.reshape(-1,1), smoothed.reshape(-1,1))),2)
print("R2 - RMSE (Exponential Smoothing)")
print("{:f} {:03f}".format(exp_r2_hour_sim, exp_rmse_hour_sim))
'''



# 24 HOUR AHEAD SIMULATION
'''
for i in range(1, datapoint_no):
                        
    temp_fan = 0
    input_sim = input_sim.reshape(1,-1)
                               
    inside_temp_sim.append(scalery.inverse_transform(svr_regr.predict(input_sim).reshape(1,-1)))
    input_sim = []
            
    #temp_variable = np.array([X_test[i,0], X_test[i,1], scalery.transform(inside_temp_sim[i-1].reshape(-1,1)), prev_fan], dtype='object')
    temp_variable = np.array([X_test[i,0], X_test[i,1],  prev_fan], dtype='object')
    #prev_time = temp_variable[0]
    prev_sol = temp_variable[0]
    prev_outside_temp = temp_variable[1]
    #prev_inside_temp = temp_variable[2]
    prev_fan = temp_variable[2]
    #prev_delta_temp = temp_variable[4]
    
            
    #input_sim.append(prev_time)
    input_sim.append(prev_sol)
    input_sim.append(prev_outside_temp)
    #input_sim.append(prev_inside_temp)
    input_sim.append(prev_fan)
    #input_sim.append(prev_delta_temp)
    input_sim = np.array(input_sim, dtype='object')
    input_sim = input_sim.reshape(1,-1)
    
    
    #temp_temp = scalery.inverse_transform(prev_inside_temp.reshape(-1,1))
    temp_temp = inside_temp_sim[i-1]
    if prev_fan == 1 and temp_temp > 22:
        temp_fan = 1
    elif prev_fan == 1 and temp_temp <= 22:
        temp_fan = 0

    if prev_fan == 0 and temp_temp < 30:
        temp_fan = 0
    elif prev_fan == 0 and temp_temp >= 30:
        temp_fan = 1
    prev_fan = temp_fan         
'''



# 24 HOUR AHEAD SIMULATION GRID SEARCH CROSS VALIDATION
'''
max_r2 = 0
best_predict = []
# For Hyperparameter Optimization
C = np.linspace(1,10,30)
epsilon = np.linspace(0.01,0.1,5)
#epsilon = 0.1
gamma = np.linspace(0.1, 10, 30)

for k in range(0, len(C)):
    print("Best R2 = " + str(round(max_r2,2)))
    for l in range(0, len(gamma)):
        for j in range(0,len(epsilon)):
            svr_regr = SVR(kernel="rbf", epsilon=epsilon[j], C=C[k], gamma=gamma[l])
            svr_regr.fit(X_train, train_y)

                    
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
            for i in range(1, datapoint_no):
                        
                temp_fan = 0
                input_sim = input_sim.reshape(1,-1)
                               
                inside_temp_sim.append(scalery.inverse_transform(svr_regr.predict(input_sim).reshape(1,-1)))
                input_sim = []
            
                temp_variable = np.array([ X_test[i,0], X_test[i,1], scalery.transform(inside_temp_sim[i-1].reshape(-1,1)), prev_fan], dtype='object')
    
                #prev_time = temp_variable[0]
                prev_sol = temp_variable[0]
                prev_outside_temp = temp_variable[1]
                prev_inside_temp = temp_variable[2]
                prev_fan = temp_variable[3]
                #prev_delta_temp = temp_variable[4]
    
            
                #input_sim.append(prev_time)
                input_sim.append(prev_sol)
                input_sim.append(prev_outside_temp)
                input_sim.append(prev_inside_temp)
                input_sim.append(prev_fan)
                #input_sim.append(prev_delta_temp)
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

        inside_temp_sim = np.array(inside_temp_sim).flatten()
        predictions_svr = inside_temp_sim
        svr_r2_sim = round(r2_score(scalery.inverse_transform(test_y[0:datapoint_no].reshape(-1,1)), predictions_svr),2)

        if svr_r2_sim > max_r2:
            max_r2 = svr_r2_sim
            best_predict = predictions_svr

        r2_list.append(svr_r2_sim)
        print('C = ' + str(C[k]) +', ' + 'epsilon = ' + str(epsilon[j]) +', ' + 'gamma = ' + str(gamma[l]) + ', ' + 'R2 = ' + str(svr_r2_sim))
        temp_text = 'C = ' + str(C[k]) +', epsilon = ' + str(epsilon[j]) +', gamma = ' + str(gamma[l]) + ', R2 = ' + str(svr_r2_sim)
        param_list.append(temp_text)
           
           
print(str(max(r2_list)) + ', ' + param_list[r2_list.index(max(r2_list))])  
predictions_svr = best_predict
#savetxt('fans.csv', predictions_svr, delimiter=',')                  
'''       

x_labels = np.arange(0,datapoint_no,1)
inside_temp_sim = np.array(inside_temp_sim).flatten()
predictions_svr = inside_temp_sim


#savetxt('simulated.csv', predictions_svr, delimiter=',')
#savetxt('prediction5-min.csv', test_predict, delimiter=',')
        
'''        
svr_r2_sim = round(r2_score(scalery.inverse_transform(test_y[0:datapoint_no].reshape(-1,1)), predictions_svr),2)
svr_mse_sim = round(math.sqrt(mean_squared_error(scalery.inverse_transform(test_y[0:datapoint_no].reshape(-1,1)), predictions_svr)),2)
#svr_rmse_pred = round(math.sqrt(mean_squared_error(y_test_fan[1:datapoint_no], test_predict[0,1:datapoint_no])),2)
predict_mae = mean_absolute_error(np.array(test_y[0:datapoint_no]), predictions_svr)
print("MAE=" + str(predict_mae))
MBE(np.array(test_y[0:datapoint_no]), predictions_svr)
print("R2=" + str(svr_r2_sim))
print("RMSE: " + str(svr_mse_sim))
'''



# GENERAL OUTPUT 

'''
plt.plot(x_labels,scalery.inverse_transform(test_y[0:datapoint_no].reshape(-1,1)), color='blue', marker='+',alpha = 0.5)
plt.plot(x_labels,predictions_svr[0:datapoint_no], color='red')
plt.plot(x_labels,scalerx.inverse_transform(test_x)[0:datapoint_no,3]*5, color='pink',alpha = 0.5)
plt.plot(x_labels,scalerx.inverse_transform(test_x)[0:datapoint_no,1], color='green',alpha = 0.5)
plt.plot(x_labels,test_predict.reshape(-1,1)[0:datapoint_no], color='black',alpha = 0.5)
#plt.plot(x_labels,predictions_svr[0:datapoint_no], color='red',alpha = 0.5)
#plt.plot(x_labels, test_x[1:datapoint_no, 2], color='purple')
plt.title("SVR 5-min R2 Simulated = " + str(svr_r2_sim) + " RMSE = " + str(svr_mse_sim))
        
plt.xlabel("Time", fontsize=15)
plt.ylabel("Temperature", fontsize=15)
plt.legend(['Actual Temperature', 'Simulated Temperature','Fan on/off', 'Outside Temperature', 'Predicted Temperature'])
plt.xticks(rotation=90)
plt.grid(False)                  
plt.show()
'''


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




# NO PREVIOUS TIME PLOTS
'''
plot_array = np.array(plot_array).flatten()
plt.plot(x_labels[0:len(time_min)-11],actual_res[0:len(time_min)-11], color='blue', alpha = 0.8)
plt.plot(x_labels[0:len(time_min)-11],plot_array, color='red', alpha=0.7)
plt.plot(x_labels, scalery.inverse_transform(test_y[0:datapoint_no].reshape(-1,1)), color='blue', marker='+',alpha = 0.5 )
plt.plot(x_labels, inside_temp_sim,color='red', alpha=0.7)

plt.xlabel("Time", fontsize=15)
plt.ylabel("Temperature", fontsize=15)
plt.legend(['Actual Temperature', 'Time Ahead Prediction'])
plt.xticks(rotation=90)
plt.grid(False)                  
plt.show()
'''



# 10 AHEAD PREDICTIONS HOURLY WITH SMOOTHING
'''      
plt.plot(x_labels,actual_res, color='blue', alpha = 0.8)
plt.plot(x_labels,smoothed_state_means, color='red', alpha=0.7)
plt.plot(x_labels,smoothed, color='orange', alpha=0.4)
plt.plot(x_labels,pred_vec, color='black',alpha = 0.2 )
plt.plot(x_labels,scalerx.inverse_transform(test_x)[:,3]*5, color='pink',alpha = 0.5)
plt.plot(x_labels,scalerx.inverse_transform(test_x)[:,1], color='green',alpha = 0.5)

plt.title("SVR 5-min R2 Simulated = " + str(r2_hour_sim) + " RMSE = " + str(rmse_hour_sim))
        
plt.xlabel("Time", fontsize=15)
plt.ylabel("Temperature", fontsize=15)
plt.legend(['Actual Temperature', 'Simulated Temperature (Kalman)','Simulated Temperature (Exp Smoothing)', 'Simulated Temperature (No Smoothing)', 'Fan on/off', 'Outside Temperature', 'Predicted Temperature'])
plt.xticks(rotation=90)
plt.grid(False)                  
plt.show()
'''


fig, ax = plt.subplots()
bp = ax.boxplot(error_array, showfliers=False, medianprops={'color': 'red'})
ax.set_xticklabels(['5 min', '10 min', '15 min', '20 min', '25 min', '30 min', '35 min', '40 min', '45 min', '50 min', '55 min', '60 min'])
ax.set_ylabel('Temperature')
ax.set_xlabel('Time Step Ahead')
ax.set_title('5-min Ahead Predictions')

# Add the median value as a label next to each box plot
'''
for i, med in enumerate([np.median(d) for d in error_array]):
    ax.annotate(str(med), xy=(i+1, med), xytext=(i+1.25, med),
                fontsize=10, color='red',
                arrowprops=dict(facecolor='red', shrink=0.05))
    '''

plt.show()