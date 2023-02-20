import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import geopandas as gpd

def load_gadm(path):
    gadm = gpd.read_file(path)[['GID_2', 'NAME_2']]
    return gadm

def load_multiruns(path):
    population = {}
    ead = {}
    for behave in os.listdir(path):
        f = os.path.join(path, behave)
        if not os.path.isfile(f):     


            population[behave] = {}
            ead[behave] = {}

            for scenario in os.listdir(os.path.join(path, behave)):
                population[behave][scenario] = {}
                ead[behave][scenario] = {}
                
                with open(os.path.join(path, behave, scenario, 'individual_nodes', 'population_node.pkl'), 'rb') as f:
                    population[behave][scenario]  = pickle.load(f) 

                with open(os.path.join(path, behave, scenario, 'individual_nodes', 'ead_node.pkl'), 'rb') as f:
                    ead[behave][scenario] = pickle.load(f)            

    return population, ead

def export_table_population(population, path, gadm):
    # test with full
    population_full = population['Full']
    # iterate over rcps
    for rcp in population_full.keys():
        population_df = pd.DataFrame()
        population_rcp = population_full[rcp]
        
        # average run results
        for run in population_rcp.keys():
            if population_df.size == 0:
                population_df = pd.DataFrame(population_rcp[run])
            else:
                population_df += pd.DataFrame(population_rcp[run])
        population_df = round(population_df/len(population_rcp.keys()))
        
        # filter out floodplains and spinup
        population_df = population_df[[col for col in population_df if col.endswith('flood_plain')]].iloc[1:]
        
        # extract population in 2015 and 2080
        population_df = population_df.iloc[[0, -1]].transpose()
        population_df.columns = ['2015', f'2080_{rcp}']
        
        # translate index        
        dep_names = gadm.set_index('GID_2', drop = True).loc[[gadm_idx[:-12] for gadm_idx in population_df.index]]['NAME_2']
        population_df = population_df.set_index(dep_names, drop=True)
        
        if rcp == 'control':
            population_export = population_df
        else:
            population_export[f'2080_{rcp}'] = np.array(population_df[f'2080_{rcp}'])
    
    population_export['slr_effect_rcp4p5'] = population_export[f'2080_rcp4p5'] - population_export[f'2080_control']
    population_export['slr_effect_rcp8p5'] = population_export[f'2080_rcp8p5'] - population_export[f'2080_control']

    population_export.to_csv(os.path.join(path, 'population_nodes_multiruns.csv'), index = True, encoding = 'utf-8-sig')


def export_table_ead(ead, path, gadm):
    # test with full
    ead_full = ead['Full']
    # iterate over rcps
    for rcp in ead_full.keys():
        ead_df = pd.DataFrame()
        ead_rcp = ead_full[rcp]
        
        # average run results
        for run in ead_rcp.keys():
            if ead_df.size == 0:
                ead_df = pd.DataFrame(ead_rcp[run])
            else:
                ead_df += pd.DataFrame(ead_rcp[run])
        ead_df = round(ead_df/len(ead_rcp.keys()))
        
        # filter out floodplains and spinup
        ead_df = ead_df[[col for col in ead_df if col.endswith('flood_plain')]].iloc[1:]
        
        # extract population in 2015 and 2080
        ead_df = ead_df.iloc[[0, -1]].transpose()
        ead_df.columns = ['2015', f'2080_{rcp}']
        
        # translate index        
        dep_names = gadm.set_index('GID_2', drop = True).loc[[gadm_idx[:-12] for gadm_idx in ead_df.index]]['NAME_2']
        ead_df = ead_df.set_index(dep_names, drop=True)
        
        if rcp == 'control':
            ead_export = ead_df
        else:
            ead_export[f'2080_{rcp}'] = np.array(ead_df[f'2080_{rcp}'])
    
    ead_export['slr_effect_rcp4p5'] = ead_export[f'2080_rcp4p5'] - ead_export[f'2080_control']
    ead_export['slr_effect_rcp8p5'] = ead_export[f'2080_rcp8p5'] - ead_export[f'2080_control']

    ead_export.to_csv(os.path.join(path, 'ead_nodes_multiruns.csv'), index = True, encoding = 'utf-8-sig')


    
if __name__ == '__main__':
    path = os.path.join('DataDrive', 'SLR', 'GADM', 'GADM_2.shp')
    gadm = load_gadm(path)

    path = os.path.join('DataDrive', 'MULTIRUNS')
    population, ead = load_multiruns(path)

    export_table_population(population, path, gadm)
    export_table_ead(ead, path, gadm)