import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


def read_test_data():
    data = pd.read_csv("training.csv")
    solar_radiation = data['Solar Radiation (W/m^2)'].to_numpy()
    outside_temp = data["Outside Temperature (T[n-1])"].to_numpy()
    inside_previous_temp = data["Inside Temperature Middle (T[n-1])"].to_numpy()
    fan_status = data["Fan on/off"].to_numpy()
    
    day_time = data['Time'].to_numpy()
    day_time = np.array([int(t.split(":")[0])*60 + int(t.split(":")[1]) for t in day_time])
    
    actual_temp = data["Actual Temperature Middle ((T[n])"].to_numpy()
    
    return solar_radiation, outside_temp, inside_previous_temp, fan_status, actual_temp, day_time



def run_test_simulation_original(model):
    solar_radiation, outside_temp, inside_previous_temp, fan_status, actual_temp, day_time = read_test_data()
    
    
    

