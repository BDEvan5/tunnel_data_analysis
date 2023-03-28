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
    
def load_train_data(filename):
    data = pd.read_csv(filename)
    scalerx = MinMaxScaler()
    scalery = MinMaxScaler()
    
    x = scalerx.fit_transform(data[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off']])
    y = scalery.fit_transform(data[['Actual Temperature Middle ((T[n])']])[:, 0]
    
    # n_samples = 500
    # x = x[:n_samples]
    # y = y[:n_samples]
    
    return x, y

def load_test_data(filename):
    data = pd.read_csv(filename)
    scalerx = MinMaxScaler()
    scalery = MinMaxScaler()
    
    x_data = data[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off']].to_numpy()
    x = scalerx.fit_transform(x_data)
    y_data = data[['Actual Temperature Middle ((T[n])']].to_numpy()
    y = scalery.fit_transform(y_data)[:, 0]
        
    # n_samples = 500
    # x = x[:n_samples]
    # y = y[:n_samples]
    
    return x, y, data, scalery

def train_svr_model():
    X_train, train_y = load_train_data("training.csv")
    
    svr_regr = SVR(kernel="rbf", epsilon=0.02894736842105263, C=7.63157894736842, gamma=1.3526315789473684)    # 1 day simulation with time (BEST)
    # svr_regr = SVR(kernel="rbf", epsilon=0.1, C=6.8965517241379315, gamma=0.7827586206)    # No time or delta temperature, best results
    svr_regr.fit(X_train, train_y)
    
    return svr_regr

def test_svr_model(svr):
    scalery = MinMaxScaler()
    X_test, test_y, df_test, scalery = load_test_data("TestSet.csv")
    
    test_true_times = df_test['Minute']
    test_true_result = np.array(df_test['Actual Temperature Middle ((T[n])'])

    # testing
    test_predict = svr.predict(X_test)
    test_predict = scalery.inverse_transform(np.array(test_predict).reshape(1,-1))
    predict_score = svr.score(X_test, test_y[0:len(test_y)])
    print("Prediction....")
    print(f"Prediction score: {predict_score}")

    inside_temp_sim = []
    input_sim = X_test[0,:]      
    prev_sol = 0
    prev_inside_temp = 0
    prev_outside_temp = 0
    datapoint_no = 285
    inside_temp_sim.append(scalery.inverse_transform(test_y[0].reshape(1,-1)))

    
    return test_true_times, test_true_result, X_test, scalery, inside_temp_sim


def run_simulation(svr, scalery, test_true_times, X_test, inside_temp_sim):
    pred_vec = []
    count_col = 0
    prev_fan = 0
    j = 0
    temp_pred = []
    plot_array = []
    error_array = np.zeros((len(test_true_times)-12,12))

    fan_pred = []

    while len(test_true_times)-j > 12:
        pred_vec = []
        input_sim = X_test[j,:]
        
        for i in range(0,12):
            
            temp_fan = 0
            input_sim = input_sim.reshape(1,-1)
                                
            pred_vec.append(scalery.inverse_transform(svr.predict(input_sim).reshape(1,-1)))
            temp_pred.append(scalery.inverse_transform(svr.predict(input_sim).reshape(1,-1)))
            
            
            error_array[j][i] = math.sqrt((test_true_result[j+i] - pred_vec[i])**2)
            
            if i == 11:  
                plot_array.append(pred_vec[i])

            input_sim = []
            if i == 11:
                fan_pred.append(prev_fan*5)
            input_val = pred_vec[i-1].reshape(-1,1)
            re_transformed_val = scalery.transform(input_val)
            temp_variable = np.array([X_test[j+i,0], X_test[j+i,1], re_transformed_val, prev_fan], dtype='object')
        
            
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

    predictions_svr = np.array(inside_temp_sim).flatten()

    plot_array = np.array(plot_array).flatten()
    svr_r2_sim = round(r2_score(test_true_result[0:len(test_true_times)-12], plot_array),2)
    svr_mse_sim = round(math.sqrt(mean_squared_error(test_true_result[0:len(test_true_times)-12], plot_array)),2)

    predict_mae = mean_absolute_error(test_true_result[0:len(test_true_times)-12], plot_array)
    print("MAE=" + str(predict_mae))
    MBE(test_true_result[0:len(test_true_times)-12], predictions_svr)
    print("R2=" + str(svr_r2_sim))
    print("RMSE: " + str(svr_mse_sim))
    
    return plot_array, error_array, fan_pred

def plot_data(test_true_times, test_true_result, plot_array, X_test, fan_pred, error_array):

    x_labels = np.arange(0,len(test_true_times),1)

    plt.plot(x_labels[0:len(test_true_times)-12],test_true_result[0:len(test_true_times)-12], color='blue', alpha = 0.8, label='Actual Temperature')
    plt.plot(x_labels[0:len(test_true_times)-12],plot_array, color='red', alpha=0.7, label='Time Ahead Prediction')
    plt.plot(x_labels[0:len(test_true_times)-12], X_test[:len(test_true_times)-12,3]*5, color='black', alpha=0.8, label='Actual Fan State')
    plt.plot(x_labels[0:len(test_true_times)-12], np.array(fan_pred)*0.8, color='green', label='Simulated Fan State')

    plt.xlabel("Time (5-minute Intervals)", fontsize=15)
    plt.ylabel("Temperature (deg. C)", fontsize=15)
    plt.ylim(0, 37)
    plt.legend(loc='lower left')
    plt.xticks(rotation=90)
    plt.grid(False)                  
    plt.show()

    # Use this later
    # fig, ax = plt.subplots()
    # bp = ax.boxplot(error_array, showfliers=False, medianprops={'color': 'red'})
    # ax.set_xticklabels(np.arange(5, 5, 60))
    # ax.set_ylabel('Temperature')
    # ax.set_xlabel('Time Step Ahead (min)')
    # ax.set_title('5-min Ahead Predictions')

    # plt.show()
    
if __name__ == "__main__":
    
    svr = train_svr_model()
    test_true_times, test_true_result, X_test, scalery, inside_temp_sim = test_svr_model(svr)
    plot_array, error_array, fan_pred = run_simulation(svr, scalery, test_true_times, X_test, inside_temp_sim)
    plot_data(test_true_times, test_true_result, plot_array, X_test, fan_pred, error_array)
    
    
    