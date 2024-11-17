from scipy.signal import butter, lfilter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal,optimize

'''
Class: Get_data is a function set to solve all the basic data analysis
'''
class Get_data:
    '''
    The initial function is loading the data, which can return the data, 
    the sampling rate and the time length of the data.
    '''
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            for i, line in enumerate(file, start=1):
                if line.strip() and line.strip()[0].isdigit():
                    skip_lines = i - 1
                    break
        self.data = np.genfromtxt(file_path, skip_header=skip_lines)
        self.sampling_rate = 100000
        self.time_length = len(self.data) / self.sampling_rate

    '''
    This function is used for select a certain interval of the data sequence.
    The selecting boundary should in the unit of second.
    '''
    def select(self,data ,start_time, stop_time):
        dt = 0.00001
        start_p = int(start_time / dt)
        stop_p = int(stop_time / dt)
        trapping_data = data[start_p:stop_p]
        return trapping_data

    '''
    This function is the basic plot function, which is used for ploting the original data sequence.
    '''
    def plot(self, data, plt_show='on', color='blue'):
        time_step = 0.00001
        num_data = len(data)
        time_variable = np.arange(start=0, stop=num_data, step=1)
        time_variable = time_variable * time_step
        plt.plot(time_variable, data, linewidth=0.5, color=color)
        plt.xlabel('time/s')
        print(len(time_variable), len(data))
        plt.ylabel('normalized APD voltage')
        plt.ylim(min(data) * 0.9, max(data) * 1.1)
        if plt_show == 'on':
            plt.show()

    '''
    The upgraded verison of the previous one, can plot the data sequence with a filtered curve
    '''
    def plot_filted(self, data, cutoff=10, sampling_fs=100000, order=4,color = 'blue'):
        def lowpass_filter(data, cutoff, fs, order):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, 'low')
            y = filtfilt(b, a, data)
            return y
        time_step = 0.00001
        num_data = len(data)
        time_variable = np.arange(start=0, stop=num_data, step=1)
        time_variable = time_variable * time_step
        filtered_data = lowpass_filter(data, cutoff=cutoff, fs=sampling_fs, order=order)
        plt.plot(time_variable, data, linewidth=0.5, alpha=0.5, color=color)
        plt.plot(time_variable, filtered_data, linewidth=1, alpha=1, color=color)
        plt.xlabel('time/s')
        plt.ylabel('normalized APD voltage')
        plt.ylim(min(data) - (max(data) - min(data)), max(data) + (max(data) - min(data)))
        plt.show()
    
    '''
    This function is for calculating the PSD and also plot it.
    '''
    def PSD(self,data,cmap='b',marker='.',fontsize=12):
        n = len(data)
        dt = 0.00001
        fs = 1/dt
        data_fft = np.abs(np.fft.fft(data)[:int(n/2)])
        data_psd = (1/(n*dt))*(np.abs(data_fft))**2
        freq = np.fft.fftfreq(n,dt)[:int(n/2)]
        freq_resol = 1/(n*dt)
        freq = freq[1:10000]
        data_psd = data_psd[1:10000]

        plt.loglog(freq,data_psd,'.',color=cmap,marker = marker,markersize=2.5)
        plt.xlabel('Frequency(Hz)',fontsize=fontsize)
        plt.ylabel('Power Spectral Density($V^2$/Hz)',fontsize = fontsize)
        plt.show()
        return freq,data_psd
    
    '''
    This function is used for fitting the Lorentzian function, which can output the corner frequency, and also plot it.
    The last input variable is used for deleting the first few points. The low frequency points sometimes are not accurate.
    '''
    def Lorentzian(self,data,freq_delete=2):
        n = len(data)
        dt = 0.00001
        fs = 1/dt
        data_fft = np.abs(np.fft.fft(data)[:int(n/2)])
        data_psd = (1/(n*dt))*(np.abs(data_fft))**2
        freq = np.fft.fftfreq(n,dt)[:int(n/2)]
        freq_resol = 1/(n*dt)
        freq = freq[1:10000]
        data_psd = data_psd[1:10000]

        def func(x,A,fc):
            return A/(x**2+fc**2)

        num_delete = int(freq_delete/0.2)
        x = freq[num_delete:]
        y = data_psd[num_delete:]
        popt,pcov = optimize.curve_fit(func,x,y)
        A = popt[0]
        fc = np.abs(popt[1])
        plt.loglog(freq,data_psd,'.',color='b',marker = '.',markersize=2.5)
        plt.loglog(freq,A/(freq**2+fc**2),color = 'r')
        plt.title('Corner frequency:'+str(round(fc,2)))
        plt.xlabel('frequency(Hz)',fontsize=12)
        plt.ylabel('Power Spectral Density($V^2$/Hz)',fontsize = 12)
        plt.show()
        return A,fc
    
    '''
    A RMSD calculate function. Window size can be changed to improve the calculation result.
    '''
    def RMSD(self,data,window_size=5000):
            N = len(data)
            num_window = int(N/window_size)
            RMSD_set = []
            data_ave = np.average(data)
            for i in range(num_window):
                start_p = i*window_size
                stop_p = (i+1)*window_size
                X = data[start_p:stop_p]
                X_ave = np.average(X)
                RMSD_temp = np.sqrt(np.sum((X-X_ave)**2)/window_size)/data_ave
                RMSD_set.append(RMSD_temp)
            RMSD = np.average(RMSD_set)
            return RMSD
        





    
