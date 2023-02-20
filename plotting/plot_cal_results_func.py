# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:22:40 2022

@author: ltf200
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def merge_tables(parent_directory):
    fn = os.path.join(parent_directory, 'runs_1', 'benchmark_output_part_1.csv')
    data_1 = pd.read_csv(fn, index_col='run')

    fn2 = os.path.join(parent_directory, 'runs_2', 'benchmark_output_part_2.csv')
    data_2 = pd.read_csv(fn2, index_col='run')

    fn3 = os.path.join(parent_directory, 'runs_3', 'benchmark_output_part_3.csv')
    data_3 = pd.read_csv(fn3, index_col='run')


    merged_tables = pd.concat([data_1, data_2, data_3]).reset_index(drop=True)
    merged_tables = merged_tables[merged_tables['max_risk'] < 12]
    return merged_tables
        
def plot_results(axs, pos, data, variable, values, label, colors, cal_lims):
       
    for i, value in enumerate(values):
        val_label = value# str(value)[:4]

        sns.distplot(data[data[variable] == value]['perc_adapted'], hist=False, kde=True, 
                      bins=int(180/5), color = colors[i], 
                     hist_kws={'edgecolor':'black'},
                     kde_kws={'linewidth': 2},
                     label = val_label,
                     ax=axs[pos])
    
    axs[pos].set_xlim([0, 100])
    # axs[pos].set_xlim([5000, 15000])

    axs[pos].legend(title = label)
    axs[pos].set_xlabel('Percentage adapted')

    axs[pos].axvline(x=cal_lims[0], color = 'k', linestyle = '--')
    axs[pos].axvline(x=cal_lims[1], color = 'k', linestyle = '--')
    axs[pos].axvspan(cal_lims[0], cal_lims[1], ymin=0.0, ymax=1, alpha=0.1, color='grey')


#%% General
# fn = r'C:\Users\ltf200\Documents\GitHub\COASTMOVE\DataDrive\SLR\calibration\cal_short_results.csv'
# data = pd.read_csv(fn, index_col='run')
data = merge_tables(r'DataDrive\BENCHMARK')
# data['expenditure_cap'] = np.round(np.array([data['expenditure_cap']]), 3)[0]
# data['interest_rate'] = np.round(np.array([data['interest_rate']]), 3)[0]
# data = data[data['max_risk'] > 1]
# data = data[data['loan_duration'] > 12]
data = data[data['expenditure_cap'] > 0.03]

# Filter low adaptation update (remove this)
# data = data[data['loan_duration'] == 20]
# data = data[data['expenditure_cap'] == 0.03]

# data = data[data['interest_rate'] == 0.040]
# data = data[data['loan_duration'] == 30]
# data = data[data['max_risk'] > 0]

# data = data[data['expenditure_cap'] == 0.04]

# initiate figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), dpi = 400)
# fig, axs = plt.subplots(nrows=2, ncols=2)

cal_lims = (9.63, 36.05)
# cal_lims = (9.63, 21.87)

colors = ['black', 'darkblue', 'darkred']

#%% Plot peak risk
pos = (0,0)
variable = 'max_risk'
label = 'Peak risk perception'
# values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
values = [1, 2, 4]

plot_results(axs=axs, pos=pos, data=data, variable=variable, values=values, label=label, colors=colors, cal_lims=cal_lims)

#%% Plot interest_rate
pos = (0,1)
variable = 'interest_rate'
label = 'Interest rate'
values = [0.03, 0.04, 0.05]
plot_results(axs=axs, pos=pos, data=data, variable=variable, values=values, label=label, colors=colors, cal_lims=cal_lims)

#%% Plot loan duration
pos = (1,0)
variable = 'loan_duration'
label = 'Loan duration'
values = [12, 16, 20]


plot_results(axs=axs, pos=pos, data=data, variable=variable, values=values, label=label, colors=colors, cal_lims=cal_lims)

#%% Plot expenditure cap
pos = (1,1)
variable = 'expenditure_cap'
label = 'Expenditure cap'
values = [0.04, 0.05, 0.06]


plot_results(axs=axs, pos=pos, data=data, variable=variable, values=values, label=label, colors=colors, cal_lims=cal_lims)
# plt.show()
plt.savefig(os.path.join('DataDrive', 'BENCHMARK', 'benchmark_plot.png'), bbox_inches='tight')
# plt.show()