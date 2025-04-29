import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def export_country_adaptation(path, area, behaves, rcp, plot=True, n_runs=None):
    if rcp == 'rcp4p5':
        ssp = 'ssp2'
    elif rcp == 'rcp8p5':
        ssp = 'ssp5'
    else:
        raise ValueError('rcp not recognized')

    for behave in behaves:
        results = {}
        adaptation_per_gdl = {}

        # get path from multirun to load individual runs
        path_control = os.path.join(path, behave, area, f'control_{ssp}', 'individual_runs')
        path_rcp = os.path.join(path, behave, area, f'{rcp}_{ssp}', 'individual_runs')
        if not os.path.exists(path_control) or not os.path.exists(path_rcp):
            print(f'Path does not exist: {path_control} or {path_rcp}')
            continue       

        if n_runs == None:
            runs = os.listdir(path_control)
            # get run ids
            ids = np.arange(np.max([int(run[-2:]) for run in runs])+1)
            base = 'run_'
            runs_sorted = [base + str(id).zfill(2) for id in ids]
        else:
            runs_sorted = [f'run_{i:02d}' for i in range(n_runs)]


        # iterate over runs and store individual countries
        for run_folder in runs_sorted:
            path_run_n_agents = os.path.join(path_rcp, run_folder, 'n_household_agents.csv')
            path_n_adapted = os.path.join(path_rcp, run_folder, 'n_households_adapted.csv')

            if (os.path.exists(path_run_n_agents) and os.path.exists(path_n_adapted)):

                # load population in floodplain
                # check if merged filename exists, if so, use that one
                if os.path.exists(path_run_n_agents.replace('.csv', '_merged.csv')):
                    n_agents = pd.read_csv(path_run_n_agents.replace('.csv', '_merged.csv'), index_col=0)
                    n_adapted = pd.read_csv(path_n_adapted.replace('.csv', '_merged.csv'), index_col=0)
                    print(f'using merged files [export_country_adaptation]')
                else:
                    n_agents = pd.read_csv(path_run_n_agents, index_col=0)
                    n_adapted = pd.read_csv(path_n_adapted, index_col=0)
                # iterate over countries and only consider floodplains
                for country in np.unique([region[:3] for region in n_agents.columns]):
                    # subset dataframe based on country and floodplain
                    cols = [col for col in n_adapted.columns if col.endswith('flood_plain') and col.startswith(country)]
                    n_agents_country = n_agents[cols]
                    n_adapted_country = n_adapted[cols]

                    perc_adapted_country = n_adapted_country.sum(axis=1) / n_agents_country.sum(axis=1)
                    perc_adapted_regions = n_adapted_country / n_agents_country
                    # for now just sum whole country
                    if country not in results.keys():
                        results[country] = {}
                        results[country]['national'] = {}
                        results[country]['regional'] = {}
                        results[country]['national_n_adapted'] = {}
                        results[country]['national'][run_folder] = perc_adapted_country
                        results[country]['regional'][run_folder] = perc_adapted_regions
                        results[country]['national_n_adapted'][run_folder] = n_adapted_country.sum(axis=1)
                    else:
                        results[country]['national'][run_folder] = perc_adapted_country
                        results[country]['regional'][run_folder] = perc_adapted_regions
                        results[country]['national_n_adapted'][run_folder] = n_adapted_country.sum(axis=1)

        # now iterate over countries and export migration of individual runs
        for country in results.keys():
            country_df = pd.DataFrame()#({'runs': runs_sorted, 'perc_adapted': np.nan}).set_index('runs')
            country_n_adapted = pd.DataFrame()
            adaptation_per_gdl_df = pd.DataFrame()

            for run in results[country]['national'].keys():
                country_df[run]= results[country]['national'][run]
                country_n_adapted[run] = results[country]['national_n_adapted'][run]
                adaptation_per_gdl_df = pd.concat([adaptation_per_gdl_df, results[country]['regional'][run]], axis=0)
            
            # do some quick statistics to summarize
            statistics = pd.DataFrame()
            statistics['minimum'] = country_df.min(axis = 1)
            statistics['mean'] = country_df.mean(axis = 1)
            statistics['maximum'] = country_df.max(axis = 1)
            # do some quick statistics to summarize
            statistics_n_adapt = pd.DataFrame()
            statistics_n_adapt['minimum'] = country_n_adapted.min(axis = 1)
            statistics_n_adapt['mean'] = country_n_adapted.mean(axis = 1)
            statistics_n_adapt['maximum'] = country_n_adapted.max(axis = 1)

            # take the average of all runs for gdl regions in country
            average_adaptation_regions = adaptation_per_gdl_df.groupby(adaptation_per_gdl_df.index).mean().loc[statistics.index]

            # export
            path_for_export = os.path.join('DataDrive', 'processed_results', area, 'individual_countries', country, behave, 'adaptation')
            os.makedirs(path_for_export, exist_ok=True)
            country_df.to_csv(os.path.join(path_for_export, f'adaptation_uptake_{rcp}_{country}.csv'))
            statistics_n_adapt.to_csv(os.path.join(path_for_export, f'min_mean_max_n_adapt_{rcp}_{country}.csv'))
            statistics.to_csv(os.path.join(path_for_export, f'min_mean_max_adaptation_uptake_{rcp}_{country}.csv'))
            average_adaptation_regions.to_csv(os.path.join(path_for_export, f'average_adaptation_gdl_regions_{rcp}_{country}.csv'))



if __name__ == '__main__':
    path = 'DataDrive/MULTIRUNS_GLOBAL'
    areas = ['asia', 'africa', 'europe', 'oceania', 'south_america', 'central_america', 'northern_america']
    behaves = ['proactive_government']#, 'maintain_fps', 'no_government', 'no_adaptation']
    # behaves = ['maintain_fps']
    areas = ['asia']
    rcps = ['rcp4p5', 'rcp8p5']
    for rcp in rcps:
        for area in areas:
            export_country_adaptation(path, area, behaves, rcp, plot=True)