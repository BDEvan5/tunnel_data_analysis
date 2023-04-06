import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


def read_data():
    data = pd.read_csv("Data/training.csv")
    solar_radiation = data['Solar Radiation (W/m^2)'].to_numpy()
    outside_temp = data["Outside Temperature (T[n-1])"].to_numpy()
    inside_previous_temp = data["Inside Temperature Middle (T[n-1])"].to_numpy()
    fan_status = data["Fan on/off"].to_numpy()
    
    day_time = data['Time'].to_numpy()
    day_time = np.array([int(t.split(":")[0])*60 + int(t.split(":")[1]) for t in day_time])
    
    actual_temp = data["Actual Temperature Middle ((T[n])"].to_numpy()
    
    return solar_radiation, outside_temp, inside_previous_temp, fan_status, actual_temp, day_time

def plot_temperatures(actual_temp, outside_temp):
    plt.figure(1)
    plt.plot(actual_temp)
    plt.plot(outside_temp)
    
    plt.show()
    
    
def plot_temperatures_zoom(actual_temp, outside_temp, solar_radiation):
    plt.figure(1)

    n = 800
    actual_temp = actual_temp[:n]
    outside_temp = outside_temp[:n]
    solar_radiation = solar_radiation[:n]

    plt.plot(actual_temp, label="Acutal temp")
    plt.plot(outside_temp, label="Outside Temp")
    plt.plot(solar_radiation / 20, label="Radiation /20")
    
    # plt.figure(2)
    # plt.plot(actual_temp - outside_temp)
    
    # plt.figure(3)
    # plt.plot(solar_radiation)
    
    plt.legend()
    plt.grid(True)
    
    plt.show()
    
def plot_day_results(day_time, indisde, outside):
    plt.figure(4)
    
    n = 12*24 # pts /day
    n_days = 20
    
    day_time = day_time/60 # convert to hours
    
    # plot n_days tempteratures on top of each other to acess the dependance on time of day
    # day_time = day_time[:(n_days * n)]
    # indisde = indisde[:(n_days * n)]
    # outside = outside[:(n_days * n)]
    
    pts = [i  for i in range(1, day_time.shape[0]) if day_time[i] < day_time[i-1]] 
    pts.insert(0, 0)
    print(pts)
    
    for i in range(n_days):
        plt.plot(day_time[pts[i]:pts[i+1]], indisde[pts[i]:pts[i+1]], label="Inside", alpha=0.5)
        plt.plot(day_time[pts[i]:pts[i+1]], outside[pts[i]:pts[i+1]], label="Outside")
        # plt.plot(day_time[i*n:(i+1)*n], indisde[i*n:(i+1)*n], label="Inside")
        # plt.plot(day_time[i*n:(i+1)*n], outside[i*n:(i+1)*n], label="Outside")
        
    # plt.legend()
    plt.grid(True)
    
    plt.show()
    
    
    
if __name__ == "__main__":
    solar_radiation, outside_temp, inside_previous_temp, fan_status, actual_temp, day_time = read_data()
    
    # plot_temperatures(actual_temp, outside_temp)
    # plot_temperatures_zoom(actual_temp, outside_temp, solar_radiation)

    plot_day_results(day_time, actual_temp, outside_temp)


