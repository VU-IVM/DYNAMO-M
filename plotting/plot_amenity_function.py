import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import os

def plot_amenity_function():
    amenity_functions = pd.read_excel(os.path.join('DataDrive', 'AMENITIES', 'amenity_functions.xlsx'), sheet_name='dist2coast')
    amenity_functions = amenity_functions.set_index('distance')
    amenity_functions['premium'] = amenity_functions['premium'] * 100
    # create interpolater object
    interpolater = interpolate.interp1d(x = amenity_functions.index, y = amenity_functions['premium'])

    # initiate distance array
    dist_arr = np.arange(0, 12.5, step= 0.01)
    amenity_premium = interpolater(dist_arr)

    plt.figure(figsize= (7, 4))
    plt.plot(dist_arr, amenity_premium, c = 'k')
    plt.xlabel('Distance to coast [km]')
    plt.ylabel(r'Amenity premium [% of wealth]')
    plt.title('Coastal amenity function')
    plt.ylim([0, 100])
    plt.grid(axis='y')
    plt.savefig(os.path.join('DataDrive', 'AMENITIES', 'plotted_function.png'))

if __name__ == '__main__':
    plot_amenity_function()