# New data file generating
import numpy as np
import function_box as f
path_1 = '../Dataset/data_9.txt'
path_2 = '../Dataset/data_new_9.txt'
start_time = 30
stop_time = 80
f.data_file_gen(path_1,path=path_2,start_time=start_time,stop_time=stop_time)


