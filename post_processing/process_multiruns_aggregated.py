import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_multiruns(path):
    population = {}
    ead = {}
    for behave in os.listdir(path):
        f = os.path.join(path, behave)
        if not os.path.isfile(f):           
            population[behave] = {}
            ead[behave] = {}
            for scenario in os.listdir(os.path.join(path, behave)):
                pop_df = pd.read_csv(os.path.join(path, behave, scenario, 'population_total.csv'), index_col='year')[1:]
                ead_df = pd.read_csv(os.path.join(path, behave, scenario, 'ead_total.csv'), index_col='year')[1:]
                population[behave][scenario] = pd.DataFrame({'pop_average_runs': pop_df.mean(axis = 1)}, index=pop_df.index)
                ead[behave][scenario] = pd.DataFrame({'ead_average_runs': ead_df.mean(axis = 1)}, index=pop_df.index)
    return population, ead


def export_table_population(population, path):
    population_df = {}
    for behave in population.keys():

        population_change = {}
        population_change[2015] = {}
        population_change[2080] = {}
        population_change['change'] = {}
        population_change['slr_effect'] = {}

        # test_full
        for rcp in population[behave].keys():
            population_change[2015][rcp] = round(population[behave][rcp].iloc[0][0])
            population_change[2080][rcp] = round(population[behave][rcp].iloc[-1][0])
            population_change['change'][rcp] = population_change[2080][rcp] - population_change[2015][rcp]
            population_change['slr_effect'][rcp] = population_change['change'][rcp] -  population_change['change']['control']

        population_df[behave] = pd.DataFrame(population_change)

    # write to excel sheet
    with pd.ExcelWriter(os.path.join(path, 'population_multiruns.xlsx')) as writer:  
        for behave in population_df.keys():
            population_df[behave].to_excel(writer, sheet_name=behave)


def export_table_ead(ead, path):
    ead_df = {}
    for behave in ead.keys():

        ead_change = {}
        ead_change[2015] = {}
        ead_change[2080] = {}
        ead_change['change'] = {}
        ead_change['slr_effect'] = {}

        # test_full
        for rcp in population[behave].keys():
            ead_change[2015][rcp] = round(ead[behave][rcp].iloc[0][0])
            ead_change[2080][rcp] = round(ead[behave][rcp].iloc[-1][0])
            ead_change['change'][rcp] = ead_change[2080][rcp] - ead_change[2015][rcp]
            ead_change['slr_effect'][rcp] = ead_change['change'][rcp] -  ead_change['change']['control']

        ead_df[behave] = pd.DataFrame(ead_change)

    # write to excel sheet
    with pd.ExcelWriter(os.path.join(path, 'ead_multiruns.xlsx')) as writer:  
        for behave in ead_df.keys():
            ead_df[behave].to_excel(writer, sheet_name=behave)


def barplots_per_behave(population, ead):
    # plot the population increase under different RCPs
    # test 'Full'
    population_growth = {}
    for rcp in population['Full'].keys():
        population_growth[rcp] = (population['Full'][rcp].iloc[-1] - population['Full'][rcp].iloc[0])[0]

    plt.bar(x=population_growth.keys(), height= population_growth.values())

if __name__ == '__main__':
    path = os.path.join('DataDrive', 'MULTIRUNS')
    population, ead = load_multiruns(path)
    export_table_population(population, path)
    export_table_ead(ead, path)
    barplots_per_behave(population=population, ead=ead)