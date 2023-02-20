from operator import index
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from string import ascii_lowercase as alc
from datetime import datetime
import pickle
import sys
import geopandas as gpd
# Add parent folder to directory
sys.path.insert(1, os.path.abspath('.'))

def plume_plot_paper(multirun_path, rcps, variables, settings, out_file, show=True, set_fontsize=False):
    """ 
    This function plots the results of a multirun.
    
    Args:
        multirun_path (str): path to the multirun folder
        rcps (dict): dictionary with the rcps to plot
        variables (dict): dictionary with the variables to plot
        settings (dict): dictionary with the settings to plot
        out_file (str): path to the output file
        show (bool): show the plot or not
    Returns:
        None 
        
        """	
    # Initiate figure
    fig, axs = plt.subplots(nrows = len(variables.keys()), ncols = len(rcps.keys()),  figsize=[16,  len(variables.keys()) * 5])#, dpi = 300)   
    
    if set_fontsize != False:
        import matplotlib
        matplotlib.rcParams['legend.fontsize'] = set_fontsize


    for setting in settings.keys():
        
        for i, rcp in enumerate(rcps.keys()):
            axs[0, i].set_title(rcps[rcp]['title'])
            
            for j, var_name in enumerate(variables.keys()):

                data = pd.read_csv(os.path.join(multirun_path, setting, rcp, f'{var_name}.csv')).set_index('year')
                
                # remove the spinup results
                data = data.iloc[1:]
                
                # extract data and years
                array = np.array(data) * variables[var_name]['scaling']
                
                years = data.index

                # extract lower, median, and upper bounds
                lower = np.percentile(array, 0, axis=1)
                median = np.median(array, axis=1)
                upper = np.percentile(array, 100, axis=1)

                axs[j, i].fill_between(x = years, y1 =  lower, y2 = upper, color = settings[setting]['color'],  alpha=.2)
                # axs[j, i].plot(years, median, color = settings[setting]['color'],  linestyle = settings[setting]['linestyle'], alpha=1, label = setting)
                axs[j, i].set_ylim(variables[var_name]['ylims'])
                axs[j, 0].set_ylabel(variables[var_name]['ylabel'])


    for setting in settings.keys():
        
        for i, rcp in enumerate(rcps.keys()):
            axs[0, i].set_title(rcps[rcp]['title'])
            
            for j, var_name in enumerate(variables.keys()):

                data = pd.read_csv(os.path.join(multirun_path, setting, rcp, f'{var_name}.csv')).set_index('year')
                
                # remove the spinup results
                data = data.iloc[1:]
                
                # extract data and years
                array = np.array(data) * variables[var_name]['scaling']
                
                years = data.index
                                
                # extract lower, median, and upper bounds
                lower = np.percentile(array, 0, axis=1)
                median = np.median(array, axis=1)
                upper = np.percentile(array, 100, axis=1)

                # axs[j, i].fill_between(x = years, y1 =  lower, y2 = upper, color = settings[setting]['color'],  alpha=.5)
                axs[j, i].plot(years, median, color = settings[setting]['color'],  linestyle = settings[setting]['linestyle'], alpha=1, label = setting)
                axs[j, i].set_ylim(variables[var_name]['ylims'])
                axs[j, 0].set_ylabel(variables[var_name]['ylabel'])


    axs[len(variables.keys())-1, 1].legend(loc='upper center', 
            bbox_to_anchor=(0.5, -0.15),fancybox=False, shadow=False, ncol=4)
        

    for i, ax in enumerate(axs.reshape(-1)): 
        ax.grid(axis = 'y')
        ax.text(0.05, 0.95, alc[i], horizontalalignment='left', verticalalignment='top', weight = 'bold', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='k', pad=10.0))

        
    for i in range(len(rcps.keys())):
        axs[0, i].set_xticklabels([])
    axs[0, 1].set_yticklabels([])
    axs[0, 2].set_yticklabels([])
    axs[1, 1].set_yticklabels([])
    axs[1, 2].set_yticklabels([])

    plt.savefig(os.path.join(multirun_path, out_file), bbox_inches='tight')
    
    if show: 
        plt.show()


def plume_plot(multirun_path, rcps, variables, settings, out_file, show=True):
    """ 
    This function plots the results of a multirun.
    
    Args:
        multirun_path (str): path to the multirun folder
        rcps (dict): dictionary with the rcps to plot
        variables (dict): dictionary with the variables to plot
        settings (dict): dictionary with the settings to plot
        out_file (str): path to the output file
        show (bool): show the plot or not
    Returns:
        None 
        
        """	
    # Initiate figure
    fig, axs = plt.subplots(nrows = len(variables.keys()), ncols = len(rcps.keys()),  figsize=[16,  len(variables.keys()) * 4])#, dpi = 300)   
    
    for setting in settings.keys():
        
        for i, rcp in enumerate(rcps.keys()):
            axs[0, i].set_title(rcps[rcp]['title'])
            
            for j, var_name in enumerate(variables.keys()):

                data = pd.read_csv(os.path.join(multirun_path, setting, rcp, f'{var_name}.csv')).set_index('year')
                
                # remove the spinup results
                data = data.iloc[1:]
                
                # extract data and years
                array = np.array(data) * variables[var_name]['scaling']
                
                years = data.index

                # extract lower, median, and upper bounds
                lower = np.percentile(array, 0, axis=1)
                median = np.median(array, axis=1)
                upper = np.percentile(array, 100, axis=1)

                axs[j, i].plot(years, median, color = settings[setting]['color'],  linestyle = settings[setting]['linestyle'], alpha=1, label = setting)
                axs[j, i].fill_between(x = years, y1 =  lower, y2 = upper, color = settings[setting]['color'],  alpha=0.1)
                axs[j, i].set_ylim(variables[var_name]['ylims'])
                axs[j, 0].set_ylabel(variables[var_name]['ylabel'])


    axs[len(variables.keys())-1, 1].legend(loc='upper center', 
            bbox_to_anchor=(0.5, -0.15),fancybox=False, shadow=False, ncol=4)
    
    for i, ax in enumerate(axs.reshape(-1)): 
        ax.grid(axis = 'y')
        ax.text(0.05, 0.95, alc[i], horizontalalignment='left', verticalalignment='top', weight = 'bold', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='k', pad=10.0))

    
    plt.savefig(os.path.join(multirun_path, out_file))
    
    if show: 
        plt.show()

def plume_plot_node(multirun_path, rcps, variables, settings, out_file, admin_name, show=True):
    ''''
    This function plots the results of a multirun for a specific node.
    
    Args:
        multirun_path (str): path to the multirun folder
        rcps (dict): dictionary with the rcps to plot
        variables (dict): dictionary with the variables to plot
        settings (dict): dictionary with the settings to plot
        out_file (str): path to the output file
        admin_name (str): name of the admin to plot
        show (bool): show the plot or not
    Returns:
        None
        '''
    fig, axs = plt.subplots(ncols = 3, nrows = len(variables.keys()),  figsize=[16, len(variables.keys())* 5])#, dpi = 300)   
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

    for setting in settings.keys():
        
        for i, rcp in enumerate(rcps.keys()):        
            axs[0, i].set_title(rcps[rcp]['title'])

            for j, var_name in enumerate(variables.keys()):          
               
                # Load pickle 
                fn = os.path.join(multirun_path, setting, rcp, 'individual_nodes', f'{var_name}.pkl')        
                
                with open(fn, 'rb') as f:
                    loaded_dict = pickle.load(f)

                # Filter admin and construct numpy array for plotting
                all_runs_filt = np.full((len(loaded_dict.keys()), len(loaded_dict[0][admin_name])-1), -1, dtype=np.float32)
                for run in loaded_dict.keys():
                    all_runs_filt[run] = loaded_dict[run][admin_name][1:] # skip spin up
                
                all_runs_filt *= variables[var_name]['scaling']
                
                # extract minimum from dict
                lower = np.percentile(all_runs_filt, 0, axis=0)
                median = np.mean(all_runs_filt, axis=0)
                upper = np.percentile(all_runs_filt, 100, axis=0)


                axs[j, i].plot(np.arange(2015, 2081), median, color = settings[setting]['color'],  linestyle = settings[setting]['linestyle'], alpha=1, label = setting)
                axs[j, i].fill_between(x = np.arange(2015, 2081), y1 =  lower, y2 = upper, color = settings[setting]['color'],  alpha=0.1)
                axs[j, i].set_ylim(variables[var_name]['ylims'])
                axs[j, 0].set_ylabel(variables[var_name]['ylabel'])
    
    axs[len(variables.keys())-1, 1].legend(loc='upper center', 
    bbox_to_anchor=(0.5, -0.15),fancybox=False, shadow=False, ncol=4)
    plt.show()


def plot_single_runs_supplement(report_folder, admin_idxs, variables, out_file, annotate_flood = True, show = False):

    '''This function plots the results of a single run for a specific admin.
    Args:
        report_folder (str): path to the report folder
        admin_idx (int): index of the admin to plot
        variables (dict): dictionary with the variables to plot
        out_file (str): path to the output file
        show (bool): show the plot or not
    Returns:
        None
       
    '''
    # load dictionary for translating
    gadm = gpd.read_file(os.path.join('DataDrive', 'SLR', 'GADM', 'GADM_2.shp')).set_index('GID_2')


    # create figure
    fig, axs = plt.subplots(ncols = 2, nrows = 3,  figsize=[13, 11])#, dpi = 300)   
    # fig.tight_layout() 
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

    for col, admin_idx in enumerate(admin_idxs):
    
        # Load flood occurances
        flood_tracker = pd.read_csv(os.path.join(report_folder, 'flood_tracker.csv'), index_col = 0)
        flood_tracker = flood_tracker[admin_idx]
        flood_tracker = flood_tracker[flood_tracker != 0]
        years_flood = [datetime.strptime(date, '%Y-%m-%d').year for date in flood_tracker.index]
        
        for i, variable in enumerate(variables.keys()):
            # Load data
            data = pd.read_csv(os.path.join(report_folder, f'{variable}.csv'), index_col = 0)
            data = data.iloc[1:]*variables[variable]['scaling']
            data = data[admin_idx]
            years = [datetime.strptime(date, '%Y-%m-%d').year for date in data.index]
            ylims = [0, max(data * 1.2)]
            axs[i, col].plot(years,np.array([data])[0])
            axs[i, col].set_title(variables[variable]['title'])
            # axs[i].set_ylim(variables[variable]['ylims'])
            axs[i, col].set_ylim(ylims)
            axs[i, col].set_ylabel(variables[variable]['ylabel'])
            if 'ylims' in variables[variable].keys():
                axs[i, col].set_ylim([variables[variable]['ylims'][col], ylims[1]])
            for j, flood in enumerate(years_flood):
                axs[i, col].axvline(x = flood, color = 'k', linestyle = '--')
                if annotate_flood and i == 0:
                    axs[i, col].text(x = flood, y = .1 * ylims[1] , s=f'RP of 1/ {flood_tracker.iloc[j]} years', rotation = -90, size = 8,
                            bbox=dict(facecolor='wheat', edgecolor='k', boxstyle='round', alpha = .8))
            print(f'{variable} in 2015: {data.iloc[0]}, {variable} in 2080: {data.iloc[-1]}')
        ax_reshaped = axs.reshape(-1)
        for i, j in enumerate([0, 2, 4, 1, 3, 5]):
            ax_reshaped[j].grid(axis = 'y')
            ax_reshaped[j].text(0.05, 0.9, alc[i], horizontalalignment='left', verticalalignment='top', weight = 'bold', transform=ax_reshaped[j].transAxes, bbox=dict(facecolor='white', edgecolor='k', pad=10.0))
        
        
        axs[0, col].set_title(gadm.loc[[admin_idx[:-12]]]['NAME_2'][0])

    plt.savefig(os.path.join(report_folder, out_file), dpi=800, bbox_inches='tight')
    if show: plt.show()


def plot_single_run(report_folder, admin_idx, variables, out_file, annotate_flood = True, show = True):

    '''This function plots the results of a single run for a specific admin.
    Args:
        report_folder (str): path to the report folder
        admin_idx (int): index of the admin to plot
        variables (dict): dictionary with the variables to plot
        out_file (str): path to the output file
        show (bool): show the plot or not
    Returns:
        None
       
    '''

    # create figure
    fig, axs = plt.subplots(ncols = len(variables.keys()), nrows = 1,  figsize=[len(variables.keys()) * 7, 5])#, dpi = 300)   
    # fig.tight_layout() 
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

    # Load flood occurances
    flood_tracker = pd.read_csv(os.path.join(report_folder, 'flood_tracker.csv'), index_col = 0)
    flood_tracker = flood_tracker[admin_idx]
    flood_tracker = flood_tracker[flood_tracker != 0]
    years_flood = [datetime.strptime(date, '%Y-%m-%d').year for date in flood_tracker.index]
    
    for i, variable in enumerate(variables.keys()):
        # Load data
        data = pd.read_csv(os.path.join(report_folder, f'{variable}.csv'), index_col = 0)
        data = data.iloc[1:]*variables[variable]['scaling']
        data = data[admin_idx]
        years = [datetime.strptime(date, '%Y-%m-%d').year for date in data.index]
        ylims = [0, max(data * 1.4)]
        axs[i].plot(years,np.array([data])[0])
        axs[i].set_title(variables[variable]['title'])
        # axs[i].set_ylim(variables[variable]['ylims'])
        axs[i].set_ylim(ylims)
        axs[i].set_ylabel(variables[variable]['ylabel'])
        for j, flood in enumerate(years_flood):
            axs[i].axvline(x = flood, color = 'k', linestyle = '--')
            if annotate_flood:
                axs[i].text(x = flood, y = .3 * ylims[1] , s=f'RP of 1/ {flood_tracker.iloc[j]} years', rotation = -90, size = 8,
                        bbox=dict(facecolor='wheat', edgecolor='k', boxstyle='round', alpha = .8))

    for i, ax in enumerate(axs.reshape(-1)): 
        ax.grid(axis = 'y')
        ax.text(0.1, 0.9, alc[i], horizontalalignment='left', verticalalignment='top', weight = 'bold', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='k', pad=10.0))

    axs[len(variables.keys())-1].legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15),
        fancybox=False, 
        shadow=False, 
        ncol=4)

    plt.savefig(os.path.join(report_folder, out_file), bbox_inches='tight')
    if show: plt.show()
