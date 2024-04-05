# Total data file figure generating
import function_box as f
import numpy as np

print('Program starting...')
path = '../Dataset/data_9.txt'
data = f.load_data(path)[:,0]
time_step = 0.00001
print('Length of the data:\t',len(data))
print('Time length of the data:\t',len(data)*time_step,'s')
f.data_plot(data)
print('Program has done.')
