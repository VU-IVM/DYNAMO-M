import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    
def export_runs(iteration, rcp):    
    n_moved_out = pd.read_csv('report/n_moved_out_last_timestep.csv')
    n_moved_out = n_moved_out[11:]

    n_moved_in = pd.read_csv('report/n_moved_in_last_timestep.csv')
    n_moved_in = n_moved_in[11:]


    tot_moved_out = n_moved_out.sum(axis=0)
    tot_moved_in = n_moved_in.sum(axis=0)

    tot_moved_out = pd.DataFrame(tot_moved_out[1:])
    tot_moved_in = pd.DataFrame(tot_moved_in[1:])
    
    tot_moved_out[0] = pd.to_numeric(tot_moved_out[0])
    tot_moved_in[0] = pd.to_numeric(tot_moved_in[0])
    
    net_change = tot_moved_in[0] - tot_moved_out[0]
    
    
    
    if os.path.exists(f'runs/runs_{rcp}.csv'):
        # Check if runs file exists. If so, append run results to dataframe and save.
        net_change_csv = pd.read_csv(f'runs/runs_{rcp}.csv')
        net_change_csv[f'run_{iteration}'] = net_change.values
        
        # Overwrite runs file
        net_change_csv.to_csv(f'runs/runs_{rcp}.csv', index=False)
    else:
        # Create intial runs file.
        net_change = net_change.reset_index()
        net_change.columns = ['keys', f'run_{iteration}']
        net_change.to_csv(f'runs/runs_{rcp}.csv', index=False)


def plot_runs(n_iterations, rcp):
    runs_control = pd.read_csv(f'runs/runs_control.csv')
    runs_control = runs_control.set_index(['keys']).T

    runs_rcp8p5 = pd.read_csv(f'runs/runs_rcp8p5.csv')
    runs_rcp8p5 = runs_rcp8p5.set_index(['keys']).T

    n_regions = len(runs_control.columns)

    pos_box_0 = np.arange(1, n_regions*2, 2)
    pos_box_1 = np.arange(2, n_regions*2+2, 2)

    ax = runs_control.boxplot(positions = pos_box_0, return_type = 'axes', color = 'grey')
    runs_rcp8p5.boxplot(positions =pos_box_1, ax = ax, color = 'black')
    plt.ylim((-1_000, 1_000))
    plt.title(f'Distribution of simulated net migration under {rcp} ({n_iterations} runs)')
    plt.savefig(fname=f'runs/images/boxplot_{rcp}.png')
    plt.show()
