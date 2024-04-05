import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal,optimize
from scipy.signal import butter, lfilter

# Function #1
# This function is design to load data from a txt file with tile or other information
# The function can automatically find the numbers of lines and skip them
# The inforamtion should only be in the top of the file
# The core reading function is genfromtxt from numpy. 
def load_data(file_path):
    with open(file_path,'r') as file:
        for i, line in enumerate(file,start=1):
            if line.strip() and line.strip()[0].isdigit():
                skip_lines = i - 1
                break
                
    data = np.genfromtxt(file_path, skip_header=skip_lines)
    return data

# Function #2
# This function is designed to find the trapping signal in a long trapping time domian signal
# THe principle can be divided the whole signal into sveral sigemtation
# Each segementation has a length of scan_size.
# If one segementation's difference between its maximum and minimum is larger thant 2 times of the amplitude
# Regarding this segementation contain the trapping.
# The output is a vector, which can be the start point for the trapping data collection
def trapping_recognize(data):
    analysis_num = 10000
    data_temp = data[:1000]; data_min = min(data_temp); data_max = max(data_temp)
    amplitude = data_max - data_min
    # view_size can be changed, it's more like a scanning resolution,
    # small view_size will result in a accurate position,
    # but too small will loss the averge of amplitude
    scan_size = 40000
    moving_step = scan_size
    segmentaion_ave_set = []
    trapping_position_l = []; trapping_position_r = []
    for i in range(int((len(data)-scan_size)/moving_step)):
        segmentaion_temp = data[i*moving_step:i*moving_step+scan_size]
        segmentaion_min = np.min(segmentaion_temp)
        segmentaion_max = np.max(segmentaion_temp)
        delta = segmentaion_max - segmentaion_min
        if delta > 2*amplitude:
            trapping_position_l.append(i*moving_step)
            trapping_position_r.append(i*moving_step+scan_size)
    trapping_position = np.array([trapping_position_l, trapping_position_r])
    if trapping_position==[]:
        trapping_start_position = []
    else:
        trapping_start_position = np.max(trapping_position[:,1])
    return trapping_start_position

# Function #3
# This fuction is designed to plot data, which conclude the sampling rate, xlabel and ylabel
# This function donot contain plt.show() or plt.figure(), which means user can plot several curves on one figure
def data_plot(data):
    time_step = 0.00001
    num_data = len(data)
    time_variable = np.arange(start=0,stop=num_data, step=1)
    time_variable = time_variable * time_step
    plt.plot(time_variable,data)
    plt.xlabel('time/s')
    print(len(time_variable),len(data))
    plt.ylabel('normalized APD voltage')
    plt.ylim(min(data)*0.9,max(data)*1.1)
    plt.show()
# Function #3.1
# This funciton is a complemenary of the main function, which can not only offer the signal
# But also gives the filtered signal, and the two curves have in the same color, but difference in transparency
def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    # 获取滤波器的系数
    b, a = butter(4, normal_cutoff,'low')
    # 应用滤波器
    y = signal.filtfilt(b, a, data)
    return y

def data_plot_filted(data, cutoff=10, sampling_fs=100000, order=5,color = 'green'):
    time_step = 0.00001
    num_data = len(data)
    time_variable = np.arange(start=0,stop=num_data, step=1)
    time_variable = time_variable * time_step
    filtered_data = lowpass_filter(data, cutoff=cutoff, fs=sampling_fs, order=order)
    plt.plot(time_variable,data,linewidth=0.5,alpha=0.5,color=color)
    plt.plot(time_variable,filtered_data,linewidth=1,alpha=1,color=color)
    plt.xlabel('time/s')
    plt.ylabel('normalized APD voltage')
    plt.ylim(min(data)-(max(data)-min(data)),max(data)+(max(data)-min(data)))
    plt.show()
# Function #4
# This function is designed to cut the data, the result data string can be the trapping signal
# The input parameters are the data string and the trapping_start_positon, which can be obtain from Function #2
# Before using this function, trapping_start_position have to be checked
def trapping_select(data):
    trapping_start_position = trapping_recognize(data)
    new_data = data[trapping_start_position:]
    return new_data

def trapping_select_m(data,start_time,stop_time):
    dt = 0.00001
    start_p = int(start_time/dt)
    stop_p = int(stop_time/dt)
    trapping_data = data[start_p:stop_p]
    return trapping_data
# Function #5
# This function is designed to generate data string for Power Spectrum Density process
# As for the previous paper:
# https://pubs.acs.org/doi/full/10.1021/acsnanoscienceau.3c00045
# The time-length for the PSD process should be 5s. And this function can generate a 5s data string.
# The input parameter is trapping data, which can be obtain by using trapping_select()
def PSD_data_generator(trapping_data):
    sampling_rate = 0.00001
    time_length = 5
    num_sample = int(time_length/sampling_rate)+1
    PSD_data = trapping_data[:num_sample]
    return PSD_data

# Function #6
# This funciton is designed to calculate the PSD of the input data
# The normal parameter(fs/dt) from is certain, dt = 0.00001
# There are 2 return variables, the first is frequency and the second one is the PSD data
def PSD_cal(data):
    n = len(data)
    dt = 0.00001
    fs = 1/dt
    data_fft = np.abs(np.fft.fft(data)[:int(n/2)])
    data_psd = (1/(n*dt))*(np.abs(data_fft))**2
    #freq = np.arange(0,5*fs/2,5*fs/n)
    freq = np.fft.fftfreq(n,dt)[:int(n/2)]
    freq_resol = 1/(n*dt)
    #cut_p = int(5000/freq_resol)
    return freq[1:], data_psd[1:]

# Function #7
# This function is designed to plot PSD figure.
# The main input variables are frequency and PSD data from function #6.
# The next three parameters are the figure setting parameters, which are have a initial value
def PSD_plot(freq,data,cmap='b',marker = '.',s=0.5,fontsize=12):
    plt.loglog(freq,data,'.',color=cmap,marker = marker,markersize=2.5)
    plt.xlabel('Frequency(Hz)',fontsize=fontsize)
    plt.ylabel('Power Spectral Density($V^2$/Hz)',fontsize = fontsize)
    plt.show()


# Function #8
# This function is designed to calculate the RMSD for signal(laser or trapping)
# The input parameters are calculated data and the window size
# Window size have a initial value 5000, which can be changed
def RMSD_cal(data,window_size=5000):
    N = len(data)
    window_size = window_size
    num_windows = int(N/window_size)
    RMSD_set = []
    data_ave = np.average(data)
    for i in range(num_windows):
        start_p = i*window_size
        stop_p = (i+1)*window_size-1
        X = data[start_p:stop_p]
        X_ave = np.average(X)
        RMSD_temp = np.sqrt(np.sum((X-X_ave)**2)/window_size)/data_ave
        RMSD_set.append(RMSD_temp)
    RMSD = np.average(RMSD_set)
    return RMSD

# Function #9
# This function is designed to calculate the corner frequency
# The fisrt definition is the function used to conduct the curve-fitting
# The second dedinition is the main part of the function
# There are 3 input parameters, frequency of the PSD, value of PSD and the frequency cut
# Based on experience, the low frequency point usually has a very large value, 
# wich may cause the fitting in a wrong way. 
# So we need to cut the value before the cutting frequency(2~4Hz)
def func(x,A,fc):
    return A/(x**2+fc**2)
def fc_cal(freq,data,freq_delete=2):
    num_delete = int(freq_delete/0.2)
    x = freq[num_delete:]
    y = data[num_delete:]
    popt,pcov = optimize.curve_fit(func,x,y)
    A = popt[0]
    fc = np.abs(popt[1])
    return A,fc

# Function #10
# This function is designed to plot the curve to show the fc.
# 4 input parameters: freqency of PSD, value of PSD, A and fs from fc_cal()
def fc_plot(freq,data_PSD,A,fc):
    plt.loglog(freq,data_PSD,'.',color='b',marker = '.',markersize=2.5)
    plt.loglog(freq,A/(freq**2+fc**2),color = 'r')
    plt.xlabel('frequency(Hz)',fontsize=12)
    plt.ylabel('Power Spectral Density($V^2$/Hz)',fontsize = 12)
    plt.show()

# Function #11
# This function is designed to 
def signal_plot(data):
    dt = 0.00001
    t = np.arange(0,dt*len(data),dt)
    plt.plot(t,data)
    plt.xlabel('Time(s)',fontsize= 12)
    plt.ylabel('Normalized APD Voltage',fontsize=12)
    plt.show()

# Function #12
# This function is desgined to generate data file from a previous data file.
# In some cases, we get the data from the set up is too huge, and the period we really want to 
# foucuse on is a tiny window of the whole data set. And, laoding and test the whole data file can
# be really time consuming. So, this function can generate a new file which contain a small window 
# of the whole data file.
def data_file_gen(data_path,path,start_time,stop_time):
    print('Data file loading...')
    data = load_data(data_path)
    dt = 0.00001
    start_point = start_time*1/dt
    stop_point = stop_time*1/dt
    data_new = data[int(start_point):int(stop_point)]
    print('New data file generating...')
    np.savetxt(path,data_new)
    print('New file have been genertated.')
    

