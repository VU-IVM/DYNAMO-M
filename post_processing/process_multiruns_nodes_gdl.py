import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from plotting.map_slr_migration import main as map_slr
from plotting.create_barplots import create_barplots_risk, create_barplots_migration, create_barplots_government_spendings, create_grouped_barplots_household_spendings
from post_processing.process_migration import export_regions_migration, export_country_migration, export_n_persons_moving_in_and_out
from post_processing.process_adaptation_uptake import export_country_adaptation
from post_processing.process_adaptation_costs import export_country_adaptation_costs, export_country_government_investment_costs, export_country_government_investment_costs_relative_to_gdp
from post_processing.process_ead import export_region_ead, export_country_ead
import subprocess

def merge_countries_in_regions(path_multiruns, behaves, n_runs=25, scenarios=None, remove_files=True):
    
    if remove_files:
        search_path = path_multiruns
        command = ["find", search_path, "-type", "f", "-name", "*_merged.csv", "-delete"]
        try:
            subprocess.run(command, check=True)
            print("Files deleted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while deleting files: {e}")

    
    # merge BGD and IND into asia
    path = path_multiruns
    countries = ['BGD', 'CHN', 'IND']
    # check if already ran:
    if os.path.exists(os.path.join(path, 'maintain_fps', 'asia', 'rcp4p5_ssp2', 'individual_runs', 'run_00', 'population_merged.csv')):
        print('Already merged countries in regions')
        return
    
    if scenarios is not None:
        rcps_ssps  = []
        scenarios_orig = scenarios.copy()
        for scenario in scenarios_orig:
            if scenario == 'rcp4p5':
                rcps_ssps.append('rcp4p5_ssp2')
                rcps_ssps.append('control_ssp2')
            elif scenario == 'rcp8p5':
                rcps_ssps.append('rcp8p5_ssp5')
                rcps_ssps.append('control_ssp5')

    

    runs_sorted = [f'run_{i:02d}' for i in range(n_runs)]

    for country in countries:
        print('Merging:', country)
        # iterate over strategies
        for strategy in behaves:
            # get folder to iterate over
            if scenarios is None:
                rcps_ssps = os.listdir(os.path.join(path, strategy, 'asia'))
            for scenario in rcps_ssps:
                for run in runs_sorted:
                    #check if run in country folder
                    path_country_run = os.path.join(path, strategy, country, scenario, 'individual_runs', run)
                    assert os.path.exists(path_country_run), 'Path does not exist: ' + path_country_run
                    # get all files that end with .csv in dir
                    results_to_merge = [file for file in os.listdir(path_country_run) if file.endswith('.csv')]                           
                    for file in results_to_merge:
                        # append to asia
                        fp_asia = os.path.join(os.path.join(path, strategy, 'asia', scenario, 'individual_runs', run, file))
                        if os.path.exists(fp_asia.replace('.csv', '_merged.csv')):
                            fp_asia = fp_asia.replace('.csv', '_merged.csv')
                        fp_country = os.path.join(os.path.join(path, strategy, country, scenario, 'individual_runs', run, file))
                        if os.path.exists(fp_asia) and os.path.exists(fp_country):
                            data_asia = pd.read_csv(fp_asia, index_col=0)
                            if len(data_asia.columns) == 1:
                                continue
                            data_country = pd.read_csv(fp_country, index_col=0)
                            data_asia = pd.concat([data_asia, data_country], axis = 1)
                            if not fp_asia.endswith('merged.csv'):
                                data_asia.to_csv(fp_asia.replace('.csv', '_merged.csv'))
                            else: 
                                data_asia.to_csv(fp_asia)
                    # now merge aggregated data
                    results_to_merge = ['population_in_flood_plain.csv', 'total_ead_nodes.csv']
                    for file in results_to_merge:
                        fp_asia = os.path.join(os.path.join(path, strategy, 'asia', scenario, 'individual_runs', run, file))
                        if os.path.exists(fp_asia.replace('.csv', '_merged.csv')):
                            fp_asia = fp_asia.replace('.csv', '_merged.csv')                            
                        fp_country = os.path.join(os.path.join(path, strategy, country, scenario, 'individual_runs', run, file))
                        if os.path.exists(fp_asia) and os.path.exists(fp_country):
                            data_asia = pd.read_csv(fp_asia, index_col=0)
                            data_country = pd.read_csv(fp_country, index_col=0)
                            data_asia = data_asia + data_country
                            if not fp_asia.endswith('merged.csv'):
                                data_asia.to_csv(fp_asia.replace('.csv', '_merged.csv'))
                            else: 
                                data_asia.to_csv(fp_asia)


def aggregate_results(behaves, areas, rcps):
    # read in datas
    dict_migration = {}
    dict_risk = {}
    dict_population = {}
    dict_population_current = {}
    dict_population_time = {}
    dict_risk_time = {}

    for rcp in rcps:
        if rcp == 'rcp4p5':
            ssp = 'ssp2'
        elif rcp == 'rcp8p5':
            ssp = 'ssp5'
        dict_migration[rcp] = {}
        dict_risk[rcp] = {}
        dict_population[rcp] = {}
        dict_population_current[rcp] = {}
        dict_population_time[rcp] = {}
        dict_risk_time[rcp] = {}
        for region in areas:
            region_df = pd.DataFrame()
            countries_in_regions = os.listdir(os.path.join('DataDrive', 'processed_results', region, 'individual_countries'))
            dict_migration[rcp][region] = {}
            dict_risk[rcp][region] = {}
            dict_population[rcp][region] = {}
            dict_population_current[rcp][region] = {}

            for behavior in behaves:
                df_behavior = pd.DataFrame()
                path_region = os.path.join('DataDrive', 'processed_results', region, behavior)
                # first get migration in region
                pd_migration = pd.read_csv(os.path.join(path_region, 'migration', f'min_mean_max_migration_{rcp}_{region}.csv'), index_col=0)
                dict_migration[rcp][region][behavior] = pd_migration.iloc[-1]['mean']
                # then do ead
                pd_risk = pd.read_csv(os.path.join(path_region, 'risk', f'ead_{rcp}_{ssp}_{region}.csv'), index_col=0)
                dict_risk[rcp][region][behavior] = pd_risk.iloc[-1]['mean']
                # then do population
                pd_population = pd.read_csv(os.path.join(path_region, 'population', f'pop_fp_{rcp}_{ssp}_{region}.csv'), index_col=0)
                dict_population[rcp][region][behavior] = pd_population.iloc[-1].values[0]
                dict_population_current[rcp][region][behavior] = pd_population.iloc[15].values[0]
                if behavior not in dict_population_time[rcp]:
                    dict_population_time[rcp][behavior] = pd_population
                    dict_risk_time[rcp][behavior] = pd_risk
                else:
                    dict_population_time[rcp][behavior] += pd_population
                    dict_risk_time[rcp][behavior] += pd_risk
                # now create tables with migration for each region
                for country in countries_in_regions:
                    path_country = os.path.join('DataDrive', 'processed_results', region, 'individual_countries', country, behavior, 'migration', f'min_mean_max_migration_{rcp}_{country}.csv')
                    if os.path.exists(path_country):
                        pd_migration_country = pd.read_csv(path_country, index_col=0)
                        migration_country_2080 = pd_migration_country.iloc[-1]['mean']
                    else:
                        migration_country_2080 = np.nan
                    df_migration = pd.DataFrame({country: migration_country_2080}, index=[behavior]).transpose()
                    df_behavior = pd.concat([df_behavior, df_migration])
                region_df = pd.concat([region_df, df_behavior], axis=1)
            # export region df to region folder
            path_for_export = os.path.join('DataDrive', 'processed_results', region)
            os.makedirs(path_for_export, exist_ok=True)
            region_df.to_csv(os.path.join(path_for_export, f'migration_{region}_{rcp}.csv')) 

        # store in csv for world
        migration = pd.DataFrame(dict_migration[rcp]).transpose()
        migration.loc['world'] = migration.sum(axis=0)

        risk  = pd.DataFrame(dict_risk[rcp]).transpose()
        risk.loc['world'] = risk.sum(axis=0)

        population = pd.DataFrame(dict_population[rcp]).transpose()
        population.loc['world'] = population.sum(axis=0)

        population_current = pd.DataFrame(dict_population_current[rcp]).transpose()
        population_current.loc['world'] = population_current.sum(axis=0)
        
        path_for_export = os.path.join('DataDrive', 'processed_results', 'world')
        os.makedirs(path_for_export, exist_ok=True)
        migration.to_csv(os.path.join(path_for_export, f'migration_{rcp}.csv'))
        risk.to_csv(os.path.join(path_for_export, f'risk_{rcp}.csv')) 
        population.to_csv(os.path.join(path_for_export, f'population_{rcp}.csv'))
        population_current.to_csv(os.path.join(path_for_export, f'population_current_{rcp}.csv'))

        # export population development
        for strategy in dict_population_time[rcp].keys():
            dict_population_time[rcp][strategy].to_csv(os.path.join(path_for_export, f'population_time_{rcp}_{strategy}.csv'))
            dict_risk_time[rcp][strategy].to_csv(os.path.join(path_for_export, f'risk_time_{rcp}_{strategy}.csv'))

def rank_countries(behaves, areas, rcps):
    for rcp in rcps:
        data_migration_countries = pd.DataFrame()
        data_ead_countries = pd.DataFrame()

        for region in areas:
            # get countries
            countries = os.listdir(os.path.join('DataDrive', 'processed_results', region, 'individual_countries'))

            # iterate over strategies
            for behave in behaves:
                # do houshold spendings (stored under regions)
                # iterate over countries
                for country in countries:
                    # load ead
                    path = os.path.join('DataDrive', 'processed_results', region, 'individual_countries', country, behave, 'risk', f'min_mean_max_ead_{rcp}_{country}.csv')
                    ead = pd.read_csv(path, index_col=0)
                    # take ead at 2015
                    ead_2015 = ead.loc['2015-01-01 00:00:00']['mean']
                    # take ead at end
                    ead_2081 = ead.iloc[-1]['mean']

                    # store in dict and add to DF
                    data = {'region': region, 'strategy': behave, 'ead_2015': ead_2015, 'ead_2081': ead_2081}
                    data = pd.DataFrame(data, index=[country])
                    data_ead_countries = pd.concat([data_ead_countries, data])
        
                    # do the same for migration
                    path = os.path.join('DataDrive', 'processed_results', region, 'individual_countries', country, behave, f'min_mean_max_migration_{rcp}_{country}.csv')
                    if not os.path.exists(path):
                        print('Migration file does not exist:', path)
                        continue
                    migration = pd.read_csv(path, index_col=0)
                    # take migration at 2081
                    migration_2081 = migration.iloc[-1]['mean']
                    # store in dict and add to DF
                    data = {'region': region, 'strategy': behave, 'migration_2081': migration_2081}
                    data = pd.DataFrame(data, index=[country])
                    data_migration_countries = pd.concat([data_migration_countries, data])

        os.makedirs(os.path.join('DataDrive', 'processed_results', 'world'), exist_ok=True)
        data_ead_countries.to_csv(os.path.join('DataDrive', 'processed_results', 'world', f'ead_countries_{rcp}.csv'))
        data_migration_countries.to_csv(os.path.join('DataDrive', 'processed_results', 'world', f'migration_countries_{rcp}.csv'))

        # also export regional totals
        df_global_export = pd.DataFrame()
        for region in areas:
            data_region = data_migration_countries[data_migration_countries['region' ]== region].groupby('strategy').sum()[['migration_2081']]
            data_region['region'] = region
            df_global_export = pd.concat([df_global_export, data_region])
        world = df_global_export.groupby('strategy').sum()[['migration_2081']]
        world['region'] = 'world'
        df_global_export = pd.concat([df_global_export, world])
        df_global_export.to_csv(os.path.join('DataDrive', 'processed_results', 'world', f'total_migration_{rcp}_regions.csv'))

def main():
    path_multiruns = os.path.join('DataDrive', 'MULTIRUNS_GLOBAL')
    path_processed_results = os.path.join('DataDrive', 'processed_results')
    n_runs = 10

    if os.path.exists(path_processed_results):
        print('Already processed results')
        # return
    behaves = ['proactive_government', 'maintain_fps', 'no_government', 'no_adaptation']  
    areas = ['africa', 'europe', 'oceania', 'south_america', 'central_america', 'northern_america', 'asia']
    rcps = ['rcp4p5', 'rcp8p5']

    merge_countries_in_regions(path_multiruns, behaves, n_runs=n_runs, scenarios=rcps)

    for area in areas:
        for rcp in rcps:
            print('Processing:', area, rcp)
            # export_n_persons_moving_in_and_out(path_multiruns, area, behaves, rcp, n_runs=n_runs)
            # export_country_government_investment_costs_relative_to_gdp(path_multiruns, area, behaves, rcp)
            # export_country_adaptation_costs(path_multiruns, area, behaves, rcp)
            # export_country_government_investment_costs(path_multiruns, area, behaves, rcp)
            export_country_adaptation(path_multiruns, area, behaves, rcp, n_runs=n_runs)
            export_regions_migration(path_multiruns, area, behaves, rcp, n_runs=n_runs)
            export_country_migration(path_multiruns, area, behaves, rcp, n_runs=n_runs)
            export_region_ead(path_multiruns, area, behaves, rcp, n_runs=n_runs)
            export_country_ead(path_multiruns, area, behaves, rcp, n_runs=n_runs)

    # create global tables
    # areas = ['africa', 'europe', 'oceania', 'south_america', 'central_america', 'northern_america', 'asia']

    aggregate_results(behaves, areas, rcps)
    
    # country level results
    rank_countries(behaves, areas, rcps)

    # order strategies for mapping
    #     colors = {
    #         'Proactive upkeep of FPS': 'blue',
    #         'Maintain current FPS':'green',
    #         'No upkeep of FPS':'darkred',
    #         'No adaptation': 'k'
    #         }

    # # create plots
    # map_in_and_outmigration(path_multiruns, behaves, areas, rcps)
    # create_grouped_barplots_household_spendings(rcps)
    # create_barplots_risk(behaves, rcps)
    # create_barplots_migration(behaves, rcps)
    # create_barplots_government_spendings(rcps)
    # map_slr(rcps, areas, behaves)


if __name__ == '__main__':
    main()
