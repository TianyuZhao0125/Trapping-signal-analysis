import numpy as np
import Function_box as f

# Giving the path of the input data
path = 'data_1.txt'

# Using function box, Get_data Class to create an instance
data_instance = f.Get_data(path)

# Plot the input data
data_instance.plot(data_instance.data)

# Using function: select, to select a certain period of the data sequence
data = data_instance.select(data_instance.data,10,15)

# Plot the certain period of data with 10Hz filtered data
data_instance.plot_filted(data)

# Using function: PSD, to generate the PSD figure
data_instance.PSD(data)

# Using function: Lorentzian, to generate the fitting curve 
data_instance.Lorentzian(data)

# Using function: RMSD, to calculate the RMSD of the certain period data
print(data_instance.RMSD(data))