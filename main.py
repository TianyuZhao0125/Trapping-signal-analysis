# Data file separation
import numpy as np
import function_box as f

print('Program starting...')
path_1 = '../Dataset/data_new_7.txt'
print('Data Loading...')
data = f.load_data(path_1)[:,0]
f.data_plot_filted(data,10,color='red')
print('The number of the sample of the input data: ',len(data))
data_trapping = f.trapping_select_m(data,4,10)
print(data.size)
print(data_trapping.size)

data_psd = f.PSD_data_generator(data_trapping)
x,y = f.PSD_cal(data_psd)
f.PSD_plot(x,y)
#freq,data = f.PSD_cal(data_3_trapping)
A,fs = f.fc_cal(x,y,freq_delete=6)
print(A,fs)
f.fc_plot(x,y,A,fs)