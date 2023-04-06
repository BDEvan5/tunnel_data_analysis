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
    
    return x, y

def load_test_data(filename):
    data = pd.read_csv(filename)
    scalerx = MinMaxScaler()
    scalery = MinMaxScaler()
    
    x_data = data[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off']].to_numpy()
    x = scalerx.fit_transform(x_data)
    y_data = data[['Actual Temperature Middle ((T[n])']].to_numpy()
    y = scalery.fit_transform(y_data)[:, 0]
        
    return x, y, data, scalery

def train_svr_model():
    X_train, train_y = load_train_data("Data/training.csv")
    
    svr_regr = SVR(kernel="rbf", epsilon=0.02894736842105263, C=7.63157894736842, gamma=1.3526315789473684)    # 1 day simulation with time (BEST)
    # svr_regr = SVR(kernel="rbf", epsilon=0.1, C=6.8965517241379315, gamma=0.7827586206)    # No time or delta temperature, best results
    svr_regr.fit(X_train, train_y)
    
    return svr_regr

def test_svr_model(svr):
    scalery = MinMaxScaler()
    X_test, test_y, df_test, scalery = load_test_data("Data/TestSet.csv")
    
    test_true_times = df_test['Minute']
    test_true_result = np.array(df_test['Actual Temperature Middle ((T[n])'])

    # testing
    test_predict = svr.predict(X_test)
    test_predict = scalery.inverse_transform(np.array(test_predict).reshape(1,-1))
    predict_score = svr.score(X_test, test_y[0:len(test_y)])
    print(f"Prediction score: {predict_score}")

    inside_temp_sim = []
    inside_temp_sim.append(scalery.inverse_transform(test_y[0].reshape(1,-1)))

    
    return test_true_times, test_true_result, X_test, scalery, inside_temp_sim


    
def run_simulation_keegan(svr, scalery, test_true_times, X_test):
    prev_fan = 0
    j = 0
    predicted_internal_temp = []
    tempterature_errors = np.zeros((len(test_true_times)-12,12))
    fan_pred = []

    while len(test_true_times)-j > 12:
        pred_vec = []
        input_sim = X_test[j,:]
        
        for i in range(0,12):
            input_sim = input_sim.reshape(1,-1)
                                
            predicted_scaled_temp = svr.predict(input_sim)
            predicted_tempterature = scalery.inverse_transform(predicted_scaled_temp.reshape(1,-1))
            pred_vec.append(predicted_tempterature)
            
            tempterature_errors[j][i] = math.sqrt((test_true_result[j+i] - pred_vec[i])**2)
            
            if i == 11:  
                predicted_internal_temp.append(pred_vec[i])
                fan_pred.append(prev_fan*5)

            prev_sol = X_test[j+i,0]
            prev_outside_temp = X_test[j+i,1]
            prev_predicted_temp = pred_vec[i-1].reshape(-1,1)
            prev_inside_temp_scaled = scalery.transform(prev_predicted_temp)[0, 0]

            input_sim = np.array([prev_sol, prev_outside_temp, prev_inside_temp_scaled, prev_fan])
        
            temp_temp = prev_predicted_temp[0,0]
            prev_fan = fan_logic(prev_fan, temp_temp)
            
        j += 1

    predicted_internal_temp = np.array(predicted_internal_temp)[:, 0, 0]
    svr_r2_sim = round(r2_score(test_true_result[0:len(test_true_times)-12], predicted_internal_temp),2)
    svr_mse_sim = round(math.sqrt(mean_squared_error(test_true_result[0:len(test_true_times)-12], predicted_internal_temp)),2)

    predict_mae = mean_absolute_error(test_true_result[0:len(test_true_times)-12], predicted_internal_temp)
    print("MAE=" + str(predict_mae))
    MBE(test_true_result[0:len(test_true_times)-12], predicted_internal_temp)
    print("R2=" + str(svr_r2_sim))
    print("RMSE: " + str(svr_mse_sim))
    
    return predicted_internal_temp, tempterature_errors, fan_pred

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
    plt.savefig('Imgs/keegan_clean-result_1')    
    plt.show()

    fig, ax = plt.subplots()
    bp = ax.boxplot(error_array, showfliers=False, medianprops={'color': 'red'})
    ax.set_xticklabels(np.arange(5, 5, 60))
    ax.set_ylabel('Temperature')
    ax.set_xlabel('Time Step Ahead (min)')
    ax.set_title('5-min Ahead Predictions')
    plt.savefig('Imgs/keegan_clean-result_2')    

    plt.show()
    
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
    
if __name__ == "__main__":
    
    svr = train_svr_model()
    test_true_times, test_true_result, X_test, scalery, inside_temp_sim = test_svr_model(svr)
    plot_array, error_array, fan_pred = run_simulation_keegan(svr, scalery, test_true_times, X_test)
    plot_data(test_true_times, test_true_result, plot_array, X_test, fan_pred, error_array)
    
    
    